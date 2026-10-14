"""Audit a constant-load contact hold and replay its resolved nodal loads natively."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .contact_reference import IndependentContactSystem
from .profile_reference_sensitivity import tensor_l2
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def collect_forces(system, collisions, potential, attributes, friction):
    """Use actual stencil membership for pad provenance and exact local gradients."""
    complete = np.zeros((len(system.rest), 3))
    sources = np.zeros((2, len(system.rest), 3))
    rows = []
    ids_full = system.mesh.to_full_vertex_id()
    for index in range(len(collisions)):
        collision = collisions[index]
        ids = np.array(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
        ids = ids[ids >= 0]
        nodes = ids_full[ids]
        groups = system.groups[nodes]
        assert set(groups) in ({0, 1}, {0, 2})
        shoe = int(groups.max()) - 1
        dofs = collision.dof(attributes, system.mesh.edges, system.mesh.faces)
        force = -potential.gradient(collision, dofs).reshape(-1, 3)
        np.add.at(complete, nodes, force)
        np.add.at(sources[shoe], nodes, force)
        resultant = force[groups == 0].sum(axis=0)
        row = {
            "index": index,
            "shoe": shoe,
            "nodes": nodes.tolist(),
            "weight": collision.weight,
            "body_resultant_N": resultant.tolist(),
        }
        magnitude = np.linalg.norm(resultant)
        if magnitude > 0:
            body_mask = groups == 0
            coefficients = force[body_mask] @ resultant / magnitude**2
            row["reference_application_point_m"] = (coefficients @ system.rest[nodes[body_mask]]).tolist()
            row["stencil_traction_magnitude_Pa"] = float(magnitude / abs(collision.weight))
            if not friction:
                row["current_application_point_m"] = (coefficients @ attributes[ids[body_mask]]).tolist()
                row["collinearity_relative_difference"] = float(
                    np.linalg.norm(force[body_mask] - coefficients[:, None] * resultant) / magnitude
                )
        if friction:
            limit = collision.mu_s * abs(collision.weight) * collision.normal_force_magnitude
            violation = np.linalg.norm(resultant) - limit
            assert violation <= 1e-10 * max(limit, 1), (index, violation, limit)
            row.update(weighted_friction_bound_N=limit, bound_violation_N=float(violation))
        rows.append(row)
        # The C++ collision is borrowed from its owning collision set.
        del collision
    expected = -system.mesh.to_full_dof(potential.gradient(collisions, system.mesh, attributes)).reshape(-1, 3)
    np.testing.assert_allclose(complete, expected, rtol=1e-10, atol=1e-12)
    return complete, sources, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.reference)
    assert int((args.reference / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads((args.reference / "solution/result.json").read_text())
    references = []
    for name, path, young, poisson, density in (
        (
            "body",
            args.mesh,
            result["material"]["body_young_Pa"],
            result["material"]["body_poisson"],
            result["material"]["body_density_kg_m3"],
        ),
        (
            "left_shoe",
            args.reference / "solution/left_shoe_mesh.npz",
            result["material"]["pad_young_Pa"],
            result["material"]["pad_poisson"],
            result["material"]["pad_density_kg_m3"],
        ),
        (
            "right_shoe",
            args.reference / "solution/right_shoe_mesh.npz",
            result["material"]["pad_young_Pa"],
            result["material"]["pad_poisson"],
            result["material"]["pad_density_kg_m3"],
        ),
    ):
        with np.load(path) as saved:
            references.append(
                (name, SkfemElasticReference(saved["vertices"], saved["tetrahedra"], young, poisson, density))
            )
    system = IndependentContactSystem(references, result["dhat_m"], 1e6, result["friction"], 1e-8)
    coordinate_map = sparse.load_npz(args.reference / "solution/coordinate_map.npz")
    with np.load(args.reference / "solution/fields.npz") as saved:
        np.testing.assert_array_equal(system.rest, saved["rest_positions"])
        np.testing.assert_array_equal(system.faces, saved["surface_triangles"])
        previous = saved["displacement"].copy()
        external = saved["applied_force_N"].copy()
        original_tensor = saved["body_tensor_Pa"].copy()
    guides = np.flatnonzero(coordinate_map.getnnz(axis=1) == 0)
    np.testing.assert_array_equal(previous[guides], 0)
    displacement, gradient, contact_force, energy, iterations = system.solve_updated_friction(
        guides,
        np.zeros(len(guides)),
        1,
        initial_displacement=previous,
        coordinate_map=coordinate_map,
        applied_force=external,
        initial_force=external,
        phase="constant_load_hold",
        failure_path=args.output / "hold_failure.npz",
    )
    normal, current = system.normal_collisions(displacement)
    tangential = ipctk.TangentialCollisions()
    tangential.build(system.mesh, current, normal, system.barrier, result["friction"])
    velocity = system.mesh.vertices((displacement - previous).reshape(-1, 3))
    normal_force, source_normal, normal_rows = collect_forces(system, normal, system.barrier, current, False)
    shear_force, source_shear, shear_rows = collect_forces(
        system, tangential, system.friction_potential, velocity, True
    )
    np.testing.assert_allclose(normal_force + shear_force, contact_force.reshape(-1, 3), rtol=1e-10, atol=1e-12)
    assert not ipctk.has_intersections(system.mesh, current)
    body = references[0][1]
    body_count = body.basis.N
    body_force = np.empty(body_count)
    body_force[system.parts[0].permutation] = contact_force[:body_count]
    body_u = np.empty(body_count)
    body_u[system.parts[0].permutation] = displacement[:body_count]
    tensor, vm = body.corner_stress(body_u)
    vertices, tets = body.mesh.p.T, body.mesh.t.T
    volumes = abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
    parts = []
    for part in system.parts:
        count = part.reference.basis.N
        section = slice(3 * part.first_node, 3 * part.first_node + count)
        vector = np.empty(count)
        vector[part.permutation] = displacement[section]
        elastic = part.reference.align_rigid_gauge(vector)
        gradients = part.reference.corner_basis.interpolate(vector).grad.transpose(2, 3, 0, 1)
        strain = (gradients + gradients.swapaxes(-1, -2)) / 2
        parts.append(
            {
                "name": part.name,
                "maximum_elastic_displacement_m": float(
                    np.linalg.norm(elastic[part.reference.component_dofs], axis=1).max()
                ),
                "maximum_symmetric_strain_norm": float(np.linalg.norm(strain, axis=(-2, -1)).max()),
                "minimum_deformation_jacobian": float(np.linalg.det(gradients + np.eye(3)).min()),
            }
        )
    np.savez_compressed(
        args.output / "contact_hold.npz",
        rest_positions=system.rest,
        previous_displacement=previous,
        displacement=displacement,
        contact_force_N=contact_force,
        applied_force_N=external,
        source_normal_force_N=source_normal,
        source_shear_force_N=source_shear,
        body_force_N=body_force,
        body_displacement=body_u,
        body_tensor_Pa=tensor,
        body_vm_Pa=vm,
    )
    omega = np.zeros(3)
    rhs = body.balanced_rhs(body_force, omega)
    gravity = np.zeros(body_count)
    gravity[body.component_dofs] = result["gravity_m_per_s2"]
    static_rhs = body_force + body.mass @ gravity
    static_relief_difference = float(np.linalg.norm(rhs - static_rhs))
    assert np.linalg.norm(body.stiffness @ body_u - rhs) <= 1e-9
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=args.mesh, method="pcg", output_mode="full", max_iterations=20000)
    model = StressModel(options)
    state = model.create_state(1, "full")
    native_coordinates = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
    permutation = body.native_dof_permutation(native_coordinates)
    state.force.from_numpy(body_force[permutation].reshape(-1, 1, 3))
    native_omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    native_omega.fill(0)
    model.recover(native_omega, state)
    native_u = np.empty(body_count)
    native_u[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
    native_tensor, native_vm = body.corner_stress(native_u)
    native_raw_tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
    native_raw_vm = qd_to_numpy(state.von_mises, transpose=True, copy=True)[0, ..., 0]
    expected_components = tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    np.testing.assert_allclose(native_raw_tensor, expected_components, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(native_raw_vm, vm, rtol=1e-4, atol=1e-3)
    reconstructed_components = native_tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    raw_postprocessor_difference = float(
        np.linalg.norm(native_raw_tensor - reconstructed_components) / np.linalg.norm(reconstructed_components)
    )
    assert raw_postprocessor_difference < 1e-9
    native_peak = float(qd_to_numpy(state.peak, transpose=True)[0])
    np.savez_compressed(
        args.output / "native_nodal_replay.npz",
        displacement=native_u,
        tensor_Pa=native_tensor,
        vm_Pa=native_vm,
        peak_Pa=native_peak,
        valid=qd_to_numpy(state.valid, transpose=True, copy=True),
        raw_native_tensor_Pa=native_raw_tensor,
        raw_native_vm_Pa=native_raw_vm,
    )
    assert bool(qd_to_numpy(state.valid, transpose=True)[0])
    complete_residual = float(np.linalg.norm(body.stiffness @ native_u - rhs))
    budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
    assert complete_residual <= budget
    tensor_difference = tensor_l2(native_tensor - tensor, volumes) / tensor_l2(tensor, volumes)
    assert tensor_difference <= 1e-4
    np.testing.assert_allclose(native_peak, vm.max(), rtol=1e-4, atol=1e-3)
    element, corner = np.unravel_index(np.argmax(vm), vm.shape)
    report = {
        "scope": __doc__,
        "reference": str(args.reference),
        "reference_mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "hold_complete_residual_N": float(np.linalg.norm(coordinate_map.T @ gradient)),
        "hold_body_residual_N": float(np.linalg.norm(gradient[:body_count])),
        "constant_load_hold_tensor_relative_change": tensor_l2(tensor - original_tensor, volumes)
        / tensor_l2(original_tensor, volumes),
        "hold_maximum_nodal_increment_m": float(np.linalg.norm((displacement - previous).reshape(-1, 3), axis=1).max()),
        "static_to_inertia_relief_rhs_difference_N": static_relief_difference,
        "native_complete_budget_fraction": complete_residual / budget,
        "native_full_tensor_relative_difference": tensor_difference,
        "raw_native_tensor_postprocessor_relative_difference": raw_postprocessor_difference,
        "native_peak_Pa": native_peak,
        "reference_raw_peak_Pa": float(vm.max()),
        "reference_raw_peak_position_m": body.corner_positions[element, corner].tolist(),
        "part_small_strain_diagnostics": parts,
        "normal_stencils": normal_rows,
        "friction_stencils": shear_rows,
        "potential_J": float(energy),
        "iterations": [asdict(i) for i in iterations],
        "limitations": "One additional constant-load increment probes the displacement-smoothed friction history; it is not temporal or spatial convergence. Direct nodal replay isolates elastic recovery. Finite-footprint/load-model replay, rigid contact feedback, spatial/regularization tails and actual Franka matching remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "hold_complete_residual_N",
                    "constant_load_hold_tensor_relative_change",
                    "native_complete_budget_fraction",
                    "native_full_tensor_relative_difference",
                    "part_small_strain_diagnostics",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
