"""Replay objective nonlinear contact loads through native linear elastic recovery."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanAllContactSystem
from .replay_resolved_grasp import collect_forces
from .sample_load_fields import common_physical_probes, locate_probes
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def cauchy(bulk, f):
    j = np.linalg.det(f)
    return (bulk.mu * (f @ f.swapaxes(-1, -2) - np.eye(3)) + bulk.lam * np.log(j)[..., None, None] * np.eye(3)) / j[
        ..., None, None
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.reference)
    assert int((args.reference / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads((args.reference / "solution/result.json").read_text())
    assert result["material"]["body_material_model"] == "neo_hookean"
    references = []
    for name, path, young, poisson, density in (
        ("body", args.mesh, 1e10, 0.3, 2000),
        ("left_shoe", args.reference / "solution/left_shoe_mesh.npz", 1e6, 0, 1000),
        ("right_shoe", args.reference / "solution/right_shoe_mesh.npz", 1e6, 0, 1000),
    ):
        with np.load(path) as saved:
            references.append(
                (name, SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density))
            )
    assert result["material"]["pad_young_Pa"] == references[1][1].young
    system = NeoHookeanAllContactSystem(references, result["dhat_m"], 1e6, result["friction"], 1e-8)
    mapping = sparse.load_npz(args.reference / "solution/coordinate_map.npz")
    original_mapping = sparse.load_npz(args.reference / "solution/original_coordinate_map.npz")
    with np.load(args.reference / "solution/fields.npz") as saved:
        np.testing.assert_array_equal(saved["rest_positions"], system.rest)
        previous, applied = saved["displacement"].copy(), saved["applied_force_N"].copy()
    guides = np.flatnonzero(mapping.getnnz(axis=1) == 0)
    u, gradient, force, _, _ = system.solve_updated_friction(
        guides,
        np.zeros(len(guides)),
        1,
        initial_displacement=previous,
        coordinate_map=mapping,
        applied_force=applied,
        initial_force=applied,
        phase="objective_constant_load_hold",
        failure_path=args.output / "hold_failure.npz",
    )
    full_residual = float(np.linalg.norm(original_mapping.T @ gradient))
    assert full_residual <= 1e-9
    normal, current = system.normal_collisions(u)
    assert not ipctk.has_intersections(system.mesh, current)
    tangent = ipctk.TangentialCollisions()
    tangent.build(system.mesh, current, normal, system.barrier, result["friction"])
    velocity = system.mesh.vertices((u - previous).reshape(-1, 3))
    fn, normal_sources, normal_rows = collect_forces(system, normal, system.barrier, current, False)
    ft, shear_sources, shear_rows = collect_forces(system, tangent, system.friction_potential, velocity, True)
    np.testing.assert_allclose(fn + ft, force.reshape(-1, 3), rtol=1e-10, atol=1e-12)
    body = references[0][1]
    count = body.basis.N
    assert np.linalg.norm(gradient[:count]) <= 1e-9
    world_u = np.empty(count)
    world_u[system.parts[0].permutation] = u[:count]
    rotation, translation, local_u = finite_frame(body, world_u)
    material_force = np.zeros(count)
    material_force[body.component_dofs] = force[:count].reshape(-1, 3) @ rotation
    reference_solution = body.solve_free_link(material_force)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=args.mesh, method="pcg", output_mode="full", max_iterations=20000)
    model = StressModel(options)
    state = model.create_state(1, "full")
    permutation = body.native_dof_permutation(qd_to_numpy(model.info.vertices, transpose=True, copy=True))
    state.force.from_numpy(material_force[permutation].reshape(-1, 1, 3))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.fill(0)
    model.recover(omega, state, options)
    assert bool(qd_to_numpy(state.valid, transpose=True)[0])
    native_u = np.empty(count)
    native_u[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
    native_tensor, native_vm = body.corner_stress(native_u)
    raw = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
    np.testing.assert_allclose(raw, native_tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5)
    rhs = body.balanced_rhs(material_force, np.zeros(3))
    budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
    residual = float(np.linalg.norm(body.stiffness @ native_u - rhs))
    assert residual <= budget
    strict_difference = float(
        np.linalg.norm(native_tensor - reference_solution.tensor_pa) / np.linalg.norm(reference_solution.tensor_pa)
    )
    assert strict_difference < 1e-8
    bulk = system.body_bulk
    f = bulk.deformation_gradients(world_u).transpose(2, 3, 0, 1)
    nonlinear = rotation.T @ cauchy(bulk, f) @ rotation
    # Use the same independent quadrature points as the nonlinear bulk.
    from skfem import Basis

    evaluation = Basis(body.mesh, body.element, quadrature=(bulk.basis.X, bulk.basis.W))
    grad = evaluation.interpolate(native_u).grad.transpose(2, 3, 0, 1)
    strain = (grad + grad.swapaxes(-1, -2)) / 2
    lam, mu = body.lame
    linear = lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) + 2 * mu * strain
    norm = lambda values: float(np.sqrt(np.sum(bulk.basis.dx * np.sum(values**2, axis=(-2, -1)))))
    physical_difference = norm(linear - nonlinear) / norm(nonlinear)
    corners, vm, upper = bulk.stress_bounds(world_u)
    probes = common_physical_probes(result["geometry"])
    elements, valid, barycentric = locate_probes(body.mesh, probes)
    corner_f = bulk.corner_deformation_gradients(world_u)
    point_f = np.einsum("nc,ncij->nij", barycentric, corner_f[elements[valid]])
    probe_nonlinear = rotation.T @ cauchy(bulk, point_f) @ rotation
    probe_linear, _ = body.stress_at_points(native_u, probes[valid])
    np.savez_compressed(
        args.output / "objective_replay.npz",
        displacement_world=u,
        previous_displacement_world=previous,
        applied_force_world_N=applied,
        contact_force_world_N=force,
        rotation_world_from_material=rotation,
        translation_world_m=translation,
        body_objective_displacement_m=local_u,
        native_displacement_material_m=native_u,
        native_force_material_N=material_force,
        native_tensor_material_Pa=native_tensor,
        nonlinear_corner_tensor_world_Pa=corners,
        probe_coordinates_m=probes,
        probe_valid=valid,
        probe_nonlinear_material_Pa=probe_nonlinear,
        probe_native_material_Pa=probe_linear,
        source_normal_world_N=normal_sources,
        source_shear_world_N=shear_sources,
    )
    report = {
        "scope": __doc__,
        "reference": str(args.reference),
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "hold_original_coordinate_residual_N": full_residual,
        "native_complete_budget_fraction": residual / budget,
        "native_vs_independent_linear_tensor_relative_difference": strict_difference,
        "native_vs_nonlinear_volume_tensor_relative_difference": physical_difference,
        "native_vs_nonlinear_probe_tensor_relative_difference": float(
            np.linalg.norm(probe_linear - probe_nonlinear) / np.linalg.norm(probe_nonlinear)
        ),
        "native_raw_peak_Pa": float(native_vm.max()),
        "nonlinear_global_peak_lower_Pa": float(vm.max()),
        "nonlinear_global_peak_upper_Pa": float(upper.max()),
        "deformation_induced_reference_moment_difference_Nm": np.cross(
            local_u[body.component_dofs], material_force[body.component_dofs]
        )
        .sum(axis=0)
        .tolist(),
        "rotation_world_from_material": rotation.tolist(),
        "normal_stencils": normal_rows,
        "friction_stencils": shear_rows,
        "goal_completion": False,
        "limitations": "Same-load native versus independent linear gates remain strict. Difference from objective nonlinear Cauchy stress is a linearization/geometry approximation diagnostic, not an algebra failure. Resolved forces remain unchanged apart from the explicit frame rotation; their deformed/reference moment difference is retained. One hold is not spatial/regularization/history convergence or actual Franka matching.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("normal_stencils", "friction_stencils")}), flush=True
    )


if __name__ == "__main__":
    main()
