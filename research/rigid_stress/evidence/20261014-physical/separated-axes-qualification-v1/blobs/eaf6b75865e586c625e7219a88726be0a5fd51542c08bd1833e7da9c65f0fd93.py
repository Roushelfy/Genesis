"""Independently reconstruct every recorded original held initializer's static forces and complete nonlinear fields."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .audit_articulated_contact import quadrature_fields
from .audit_original_history_fields import wide_volume_bounds
from .qualify_articulated_precision import tensor_from_f
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def contact_fields(static, displacement, previous):
    normal, current = static.normal_collisions(displacement)
    positions = static.contact_node_map @ (static.rest + displacement.reshape(-1, 3))
    surface = np.zeros_like(positions)
    for index in range(len(normal)):
        collision = normal[index]
        ids = np.asarray(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
        ids = ids[ids >= 0]
        assert set(static.contact_surface_groups[ids]) in ({0, 1}, {0, 2})
        local = np.asarray(positions[ids] - positions[ids[0]], dtype=float).reshape(-1)
        force = -static.barrier.gradient(collision, local).reshape(-1, 3)
        np.add.at(surface, ids, force)
        del collision
    normal_surface = surface.copy()
    tangential = ipctk.TangentialCollisions()
    if static.friction:
        tangential.build(static.mesh, current, normal, static.barrier, static.friction)
    delta = np.asarray(static.contact_node_map @ (displacement - previous).reshape(-1, 3), dtype=float)
    # The held initializer's original mollifier is1e-8 m. Its static law has
    # neither dynamic dt scaling nor a velocity/PD damping term.
    friction = ipctk.FrictionPotential(1e-8)
    violations = []
    for index in range(len(tangential)):
        collision = tangential[index]
        ids = np.asarray(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
        ids = ids[ids >= 0]
        force = -friction.gradient(collision, collision.dof(delta, static.mesh.edges, static.mesh.faces)).reshape(-1, 3)
        np.add.at(surface, ids, force)
        bound = collision.mu_s * abs(collision.weight) * collision.normal_force_magnitude
        violation = float(np.linalg.norm(force[static.contact_surface_groups[ids] == 0].sum(axis=0)) - bound)
        assert violation <= 1e-12
        violations.append(violation)
        del collision
    force = static.mesh.to_full_dof(surface.reshape(-1))
    return (
        force,
        normal_surface,
        surface - normal_surface,
        {
            "normal_stencils": len(normal),
            "friction_stencils": len(tangential),
            "maximum_coulomb_violation_N": max(violations, default=0.0),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) == 0
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    static, config = restore_static(args.run)
    assert config["cold_closure"] and config["control_steps"] == 64
    configuration = args.run / "results/configuration.json"
    retain(configuration)
    source_summary = args.run / "results/summary.json"
    retain(source_summary)
    original_rows = json.loads(source_summary.read_text())["closure"]
    assert len(original_rows) == 64
    mapping, held = static.coordinate_map(True)
    map_path = args.run / "results/held_coordinate_map.npz"
    retain(map_path)
    recorded_map = sparse.load_npz(map_path)
    assert mapping.shape == recorded_map.shape
    for name in ["data", "indices", "indptr"]:
        np.testing.assert_array_equal(getattr(mapping, name), getattr(recorded_map, name))
    parameters = args.run / "results/held_initializer_inputs.npz"
    retain(parameters)
    gravity = static.gravity_force([0, 0, -9.81])
    permutation = static.parts[0].permutation
    with np.load(parameters) as saved:
        np.testing.assert_array_equal(held, saved["held_dofs"])
        np.testing.assert_array_equal(gravity, saved["applied_force_N"])
        np.testing.assert_array_equal(permutation, saved["body_component_permutation"])
    paths = sorted((args.run / "results/initializer_states").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert len(paths) == config["control_steps"] == 64
    args.output.mkdir(parents=True, exist_ok=False)
    bulk = WidePiolaBulk(static.body)
    n = static.body.basis.N
    previous = np.zeros(static.stiffness.shape[0])
    rows = []
    for step, path in enumerate(paths, 1):
        assert path.name == f"step{step:04d}.npz"
        retain(path)
        with np.load(path) as saved:
            u, history, original_gradient, original_force, applied, target = [
                saved[name].copy()
                for name in [
                    "displacement",
                    "previous_displacement",
                    "complete_gradient_N",
                    "contact_force_N",
                    "applied_force_N",
                    "q_target_m",
                ]
            ]
            np.testing.assert_array_equal(saved["body_component_permutation"], permutation)
            np.testing.assert_array_equal(saved["held_dofs"], held)
        np.testing.assert_array_equal(history, previous)
        np.testing.assert_array_equal(applied, gravity)
        np.testing.assert_array_equal(u[held], np.zeros(len(held)))
        expected_target = np.full(2, 0.04 + (0.018 - 0.04) * step / 64)
        np.testing.assert_array_equal(target, expected_target)
        np.testing.assert_array_equal(target, original_rows[step - 1]["target_m"])
        static.q_target[:] = target
        contact, normal, shear, row = contact_fields(static, u, history)
        np.testing.assert_allclose(contact, original_force, rtol=1e-10, atol=1e-12)
        vector = np.empty(n, dtype=np.longdouble)
        vector[permutation] = u[:n]
        gradient = np.zeros(len(u), dtype=np.longdouble)
        gradient[:n] = bulk.evaluate(vector, False)[1][permutation]
        gradient -= contact + gravity
        for index, (node, axis, q_reference) in enumerate(
            zip(static.finger_com_nodes, static.axes, static.q_reference, strict=True)
        ):
            dofs = np.arange(3 * node, 3 * node + 3)
            error = q_reference + u[dofs] @ axis - target[index]
            gradient[dofs] += np.clip(static.kp * error, -static.force_cap, static.force_cap) * axis
        reduced = mapping.T @ gradient
        original_reduced = mapping.T @ original_gradient
        residual = float(np.linalg.norm(reduced))
        original_residual = float(np.linalg.norm(original_reduced))
        gradient_difference = float(np.linalg.norm(gradient - original_gradient))
        assert residual <= 1e-9 and original_residual <= 1e-9 and gradient_difference <= 1e-9
        corner_f = bulk.corner_deformation_gradients(vector)
        corner_tensor = tensor_from_f(corner_f, static.body.lame)
        deviator = corner_tensor - np.trace(corner_tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        vm = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
        quadrature, energy = quadrature_fields(bulk, np.asarray(vector, dtype=float))
        barycentric = np.vstack((1 - bulk.basis.X.sum(axis=0), bulk.basis.X))
        # Interpolate H=F-I, then restore I exactly. Interpolating I with
        # rounded barycentric weights creates artificial volumetric stress.
        identity = np.eye(3, dtype=np.longdouble)
        interpolated_f = identity + np.einsum("cq,ecij->eqij", barycentric, corner_f - identity)
        from_corners = tensor_from_f(interpolated_f, static.body.lame)
        np.testing.assert_allclose(quadrature, from_corners, rtol=1e-10, atol=1e-6)
        lower = float(vm.max())
        determinant, upper = wide_volume_bounds(corner_f, bulk.mu)
        assert lower <= upper
        cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
        support = np.zeros_like(gradient)
        support[held] = -gradient[held]
        body_support = support[:n].reshape(-1, 3)
        current_body = static.rest[: n // 3] + u[:n].reshape(-1, 3)
        row.update(
            step=step,
            q_target_m=target.tolist(),
            independent_complete_held_residual_N=residual,
            producer_complete_held_residual_N=original_residual,
            full_gradient_reconstruction_difference_N=gradient_difference,
            maximum_contact_force_reconstruction_difference_N=float(abs(contact - original_force).max()),
            raw_global_peak_lower_Pa=lower,
            raw_global_peak_upper_Pa=upper,
            raw_peak_reference_position_m=static.body.corner_positions[cell, corner].tolist(),
            whole_volume_J_lower_bound=determinant,
            nonlinear_energy_J=energy,
            held_body_support_force_N=np.asarray(body_support.sum(axis=0), dtype=float).tolist(),
            held_body_support_moment_Nm=np.asarray(
                np.cross(current_body, body_support).sum(axis=0), dtype=float
            ).tolist(),
            full_quadrature_tensor_components_checked=int(quadrature.size),
            quadrature_vs_corner_deformation_maximum_tensor_difference_Pa=float(abs(quadrature - from_corners).max()),
        )
        np.savez_compressed(
            args.output / path.name,
            independent_complete_gradient_N=gradient,
            independent_complete_held_gradient_N=reduced,
            complete_contact_force_N=contact,
            normal_surface_force_N=normal,
            shear_surface_force_N=shear,
            complete_support_reaction_N=support,
            raw_corner_tensor_Pa=corner_tensor,
            raw_corner_vm_Pa=vm,
            complete_quadrature_tensor_Pa=quadrature,
        )
        rows.append(row)
        previous = u
        print(json.dumps(row), flush=True)
    closure = args.run / "results/closure_state.npz"
    retain(closure)
    with np.load(closure) as saved:
        np.testing.assert_array_equal(previous, saved["displacement"])
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "run": str(args.run),
        "states_checked": len(rows),
        "all_original64_initializer_states_checked": True,
        "rows": rows,
        "input_sha256": inputs,
        "new_equilibrium_states_solved": 0,
        "goal_completion": False,
        "limitations": "Independent static reconstruction uses every actual friction predecessor, the original held-coordinate map, constant gravity, static capped proportional jaw effort and original1e-8m friction mollifier. Complete1e-9N held-force gates remain unchanged; six body support reactions remain explicit and are not treated as free-body residuals. Fresh constitutive contraction, every corner/quadrature tensor and raw whole-volume bounds cover all64 controls. These are audits of recorded producer states, not new solves or accepted spatial/contact/pressure/history/model/feedback/cost convergence.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
