"""Measure force invariance and residual components at the unchanged failed spatial state."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np

from .franka_rigid_mode_coordinates import coordinate_map
from .wide_spatial_franka import WideSpatialInertia, restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    static, config = restore_static(args.run)
    system = WideSpatialInertia(static, config["dt"])
    system.q_target[:] = 0.018
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    with np.load(paths[-1]) as saved:
        previous = np.asarray(saved["displacement"], dtype=np.longdouble)
        velocity = np.asarray(saved["velocity"], dtype=np.longdouble)
    with np.load(args.run / "results/failure.npz") as saved:
        displacement = np.asarray(saved["displacement"], dtype=np.longdouble)
        np.testing.assert_array_equal(saved["previous_displacement"], previous)
    system.predicted_displacement = previous + config["dt"] * velocity
    external = system.gravity_force([0, 0, -9.81])
    evaluated = system.evaluate_updated_friction(displacement, previous, external, False)
    n = system.body.basis.N
    permutation = system.parts[0].permutation
    vector = np.empty(n, dtype=np.longdouble)
    vector[permutation] = displacement[:n]
    gradient = system.bulk.evaluate(vector, False)[1]
    positions = system.body.coordinates + vector[system.body.component_dofs]
    nodal_gradient = gradient[system.body.component_dofs]
    mapping = coordinate_map(system)
    reduced = mapping.T @ evaluated[1]
    result = {
        "scope": __doc__,
        "body_tetrahedra": system.body.mesh.nelements,
        "complete_reduced_residual_N": float(np.linalg.norm(reduced)),
        "body_residual_N": float(np.linalg.norm(evaluated[1][:n])),
        "nonbody_residual_N": float(np.linalg.norm(evaluated[1][n:])),
        "reduced_body_residual_N": float(np.linalg.norm(reduced[:-2])),
        "reduced_jaw_residual_N": float(np.linalg.norm(reduced[-2:])),
        "body_rigid_mode_residual_N_Nm": (system.body.rigid_modes[permutation].T @ evaluated[1][:n]).tolist(),
        "internal_bulk_resultant_N": nodal_gradient.sum(axis=0).tolist(),
        "internal_bulk_moment_Nm": np.cross(positions - system.body.com, nodal_gradient).sum(axis=0).tolist(),
        "basis_partition_gradient_max_m_inv": float(
            abs(np.sum(np.asarray(system.bulk.gradients, dtype=np.longdouble), axis=0)).max()
        ),
        "translation_tests": [],
        "goal_completion": False,
    }
    for offset in [[1e-6, -2e-6, 3e-6], [1e-4, -2e-4, 3e-4]]:
        shifted = vector.copy()
        shifted[system.body.component_dofs] += np.array(offset, dtype=np.longdouble)
        shifted_gradient = system.bulk.evaluate(shifted, False)[1]
        pure = np.zeros(n, dtype=np.longdouble)
        pure[system.body.component_dofs] = offset
        pure_gradient = system.bulk.evaluate(pure, False)[1]
        result["translation_tests"].append(
            {
                "offset_m": offset,
                "same_deformation_force_difference_N": float(np.linalg.norm(shifted_gradient - gradient)),
                "pure_translation_bulk_force_N": float(np.linalg.norm(pure_gradient)),
            }
        )
    normal, current = system.normal_collisions(displacement)
    precise_current = system.contact_node_map @ (
        np.asarray(system.rest, dtype=np.longdouble) + displacement.reshape(-1, 3)
    )
    np.testing.assert_allclose(current, precise_current, rtol=0, atol=1e-14)
    result["normal_stencil_precision"] = []
    for index in range(len(normal)):
        collision = normal[index]
        ids = np.array(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
        ids = ids[ids >= 0]
        groups = system.contact_surface_groups[ids]
        dofs = np.array(collision.dof(current, system.mesh.edges, system.mesh.faces))
        global_force = -system.barrier.gradient(collision, dofs).reshape(-1, 3)
        origin = np.asarray(system.contact_surface_rest[ids[groups == 0][0]], dtype=np.longdouble)
        local = np.asarray(precise_current[ids] - origin, dtype=float).reshape(dofs.shape)
        local_force = -system.barrier.gradient(collision, local).reshape(-1, 3)
        body_row = int(np.flatnonzero(groups == 0)[0])
        axis = int(np.argmax(abs(global_force[groups == 0].sum(axis=0))))
        perturbed = dofs.copy()
        points = perturbed.reshape(-1, 3)
        points[body_row, axis] = np.nextafter(points[body_row, axis], np.inf)
        changed = -system.barrier.gradient(collision, perturbed).reshape(-1, 3)
        row = {
            "kind": type(collision).__name__,
            "groups": groups.tolist(),
            "global_body_force_N": global_force[groups == 0].sum(axis=0).tolist(),
            "local_frame_body_force_N": local_force[groups == 0].sum(axis=0).tolist(),
            "local_vs_global_force_difference_N": float(np.linalg.norm(local_force - global_force)),
            "one_global_coordinate_ULP_force_change_N": float(np.linalg.norm(changed - global_force)),
            "global_coordinate_ULP_m": float(abs(points[body_row, axis] - dofs.reshape(-1, 3)[body_row, axis])),
        }
        if sum(groups == 0) == 1 and len(ids) == 4:
            triangle = precise_current[ids][groups != 0]
            direction = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            direction /= np.sqrt(np.sum(direction * direction))
            point = precise_current[ids][groups == 0][0]
            row["point_plane_gap_m"] = float(abs(np.sum((point - triangle[0]) * direction)))
        result["normal_stencil_precision"].append(row)
        del collision
    tangential = ipctk.TangentialCollisions()
    tangential.build(system.mesh, current, normal, system.barrier, system.friction)
    if len(tangential):
        collision = tangential[0]
        result["tangential_force_attributes"] = [name for name in dir(collision) if "force" in name or "basis" in name]
        value = collision.normal_force_magnitude
        try:
            collision.normal_force_magnitude = value
            result["tangential_normal_force_writable"] = True
        except (AttributeError, TypeError):
            result["tangential_normal_force_writable"] = False
        del collision
    np.savez_compressed(
        args.output / "forces.npz",
        body_gradient_N=evaluated[1][:n],
        full_gradient_N=evaluated[1],
        bulk_gradient_N=gradient,
        reduced_gradient_N=reduced,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, default=float))


if __name__ == "__main__":
    main()
