"""Replay solved inertial Panda loads and quantify full-inertia versus rigid relief."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from skfem import Basis

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .audit_objective_frame import finite_frame
from .dynamic_franka_reference import InertialRigidFingerContactSystem
from .franka_contact_reference import build_from_capture
from .franka_rigid_mode_coordinates import coordinate_map
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) == 0
    config = json.loads((args.run / "results/configuration.json").read_text())
    closure = Path(config["closure_run"])
    original_config = json.loads((closure / "results/configuration.json").read_text())
    static, _ = build_from_capture(Path(original_config["capture"]), original_config["dhat"])
    system = InertialRigidFingerContactSystem(static, config["dt"])
    system.q_target[:] = 0.018
    body, count = system.body, system.body.basis.N
    step = config["steps"]
    with np.load(args.run / f"results/step{step:04d}.npz") as saved:
        u, v, a, f, external = (
            saved[key].copy()
            for key in ("displacement", "velocity", "acceleration", "contact_force_N", "applied_force_N")
        )
        np.testing.assert_array_equal(saved["body_component_permutation"], system.parts[0].permutation)
    with np.load(args.run / f"results/step{step - 1:04d}.npz") as saved:
        previous, previous_v = saved["displacement"].copy(), saved["velocity"].copy()
    system.predicted_displacement = previous + config["dt"] * previous_v
    _, g, _, normal, contact_g = system.evaluate_updated_friction(u, previous, external, False)
    np.testing.assert_allclose(-contact_g, f, rtol=1e-10, atol=1e-12)
    dynamic_budget = float(np.linalg.norm(coordinate_map(system).T @ g))
    assert dynamic_budget <= 1e-9 and np.linalg.norm(g[:count]) <= 1e-9
    world_u, world_previous = np.empty(count), np.empty(count)
    permutation = system.parts[0].permutation
    world_u[permutation] = u[:count]
    world_previous[permutation] = previous[:count]
    frame, _translation, local_u = finite_frame(body, world_u)
    previous_frame, _, _ = finite_frame(body, world_previous)
    omega_world = Rotation.from_matrix(frame @ previous_frame.T).as_rotvec() / config["dt"]
    omega_material = frame.T @ omega_world
    material_f = np.zeros(count)
    material_f[body.component_dofs] = f[:count].reshape(-1, 3) @ frame
    inertial_force = system.inertial_mass @ a
    material_full_rhs = np.zeros(count)
    material_full_rhs[body.component_dofs] = (f + external - inertial_force)[:count].reshape(-1, 3) @ frame
    linear = body.solve(material_f, omega_material)
    full_inertia_linear = body.solve(material_full_rhs)
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = args.output / "body_mesh.npz"
    np.savez_compressed(mesh, vertices=body.mesh.p.T, tetrahedra=body.mesh.t.T)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=mesh, method="pcg", output_mode="full", max_iterations=20000)
    model = StressModel(options)
    state = model.create_state(1, "full")
    native_permutation = body.native_dof_permutation(qd_to_numpy(model.info.vertices, transpose=True, copy=True))
    state.force.from_numpy(material_f[native_permutation].reshape(-1, 1, 3))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.from_numpy(omega_material[None])
    model.recover(omega, state, options)
    assert bool(qd_to_numpy(state.valid, transpose=True)[0])
    native_u = np.empty(count)
    native_u[native_permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
    native_tensor, native_vm = body.corner_stress(native_u)
    raw = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
    np.testing.assert_allclose(raw, native_tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5)
    peak = float(qd_to_numpy(state.peak)[0])
    np.testing.assert_allclose(peak, native_vm.max(), rtol=1e-4, atol=1e-3)
    rhs = body.balanced_rhs(material_f, omega_material)
    residual = float(np.linalg.norm(body.stiffness @ native_u - rhs))
    budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
    assert residual <= budget
    algebra = float(np.linalg.norm(native_tensor - linear.tensor_pa) / np.linalg.norm(linear.tensor_pa))
    assert algebra < 1e-8
    bulk = system.bulk
    deform = bulk.deformation_gradients(world_u).transpose(2, 3, 0, 1)
    j = np.linalg.det(deform)
    b = deform @ deform.swapaxes(-1, -2)
    cauchy = (bulk.mu * (b - np.eye(3)) + bulk.lam * np.log(j)[..., None, None] * np.eye(3)) / j[..., None, None]
    nonlinear = frame.T @ cauchy @ frame
    evaluation = Basis(body.mesh, body.element, quadrature=(bulk.basis.X, bulk.basis.W))

    def tensor(displacement):
        grad = evaluation.interpolate(displacement).grad.transpose(2, 3, 0, 1)
        strain = (grad + grad.swapaxes(-1, -2)) / 2
        return bulk.lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) + 2 * bulk.mu * strain

    def norm(values):
        return float(np.sqrt(np.sum(bulk.basis.dx * np.sum(values**2, axis=(-2, -1)))))

    native_field, full_field = tensor(native_u), tensor(full_inertia_linear.displacement)
    corners, vm, upper = bulk.stress_bounds(world_u)
    np.savez_compressed(
        args.output / "replay_fields.npz",
        native_tensor_material_Pa=native_tensor,
        raw_native_tensor_material_Pa=raw,
        native_displacement_m=native_u,
        contact_force_material_N=material_f,
        full_inertia_rhs_material_N=material_full_rhs,
        reference_nonlinear_corner_tensor_world_Pa=corners,
        reference_nonlinear_vm_Pa=vm,
        rotation_world_from_material=frame,
        body_objective_displacement_m=local_u,
        full_inertia_linear_displacement_m=full_inertia_linear.displacement,
    )
    output = {
        "scope": __doc__,
        "reference": str(args.run),
        "dynamic_reference_complete_residual_N": dynamic_budget,
        "native_complete_budget_fraction": residual / budget,
        "actual_native_vs_independent_linear_tensor_relative_difference": algebra,
        "native_rigid_relief_vs_dynamic_nonlinear_reference_volume_tensor_difference": norm(native_field - nonlinear)
        / norm(nonlinear),
        "full_inertia_linear_vs_dynamic_nonlinear_reference_volume_tensor_difference": norm(full_field - nonlinear)
        / norm(nonlinear),
        "rigid_relief_vs_full_inertia_linear_reference_volume_tensor_difference": norm(native_field - full_field)
        / norm(full_field),
        "native_raw_global_peak_Pa": peak,
        "nonlinear_global_peak_lower_Pa": float(vm.max()),
        "nonlinear_global_peak_upper_Pa": float(upper.max()),
        "omega_material_rad_s": omega_material.tolist(),
        "deformation_induced_reference_moment_difference_Nm": np.cross(
            local_u[body.component_dofs], material_f[body.component_dofs]
        )
        .sum(axis=0)
        .tolist(),
        "objective_elastic_displacement_max_m": float(np.linalg.norm(local_u[body.component_dofs], axis=1).max()),
        "normal_stencils": len(normal),
        "goal_completion": False,
        "limitations": "Same-nodal-load native algebra gates use actual raw tensors and the global peak. Reference-volume nonlinear comparisons are one coarse solved dynamic state, not finite-footprint mapping, spatial/contact convergence or full-arm matching. The full-inertia diagnostic retains elastic acceleration but still uses reference geometry and linear constitutive response; its difference is not an exact causal decomposition. Fitted angular velocity is a backward finite rotation rate. Actual contact forces are rotated without changing their values; deformation-induced application moments are retained.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
