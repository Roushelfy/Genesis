"""Evaluate original mapped contact forces with centered stencil coordinates and direct increments."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np

from .franka_rigid_mode_coordinates import coordinate_map
from .wide_spatial_franka import WideSpatialInertia, restore_static


class PreciseStencilSpatial(WideSpatialInertia):
    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, original_contact = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        precise = self.contact_node_map @ (
            np.asarray(self.rest, dtype=np.longdouble) + np.asarray(displacement, dtype=np.longdouble).reshape(-1, 3)
        )
        surface_gradient = np.zeros(precise.shape)
        magnitudes = {}
        for index in range(len(normal)):
            collision = normal[index]
            ids = np.asarray(collision.vertex_ids(self.mesh.edges, self.mesh.faces))
            ids = ids[ids >= 0]
            groups = self.contact_surface_groups[ids]
            assert set(groups) in ({0, 1}, {0, 2})
            centered = np.asarray(precise[ids] - precise[ids[0]], dtype=float).reshape(-1)
            local = self.barrier.gradient(collision, centered).reshape(-1, 3)
            np.add.at(surface_gradient, ids, local)
            assert collision.weight != 0
            magnitudes[tuple(ids)] = self.barrier.force_magnitude(collision.compute_distance(centered), collision.dmin)
            del collision
        contact = self.mesh.to_full_dof(surface_gradient.reshape(-1))
        if self.friction > 0:
            for index in range(len(tangential)):
                collision = tangential[index]
                ids = np.asarray(collision.vertex_ids(self.mesh.edges, self.mesh.faces))
                ids = ids[ids >= 0]
                collision.normal_force_magnitude = magnitudes[tuple(ids)]
                del collision
            delta = np.asarray(
                self.contact_node_map
                @ (np.asarray(displacement, dtype=np.longdouble) - np.asarray(previous, dtype=np.longdouble)).reshape(
                    -1, 3
                ),
                dtype=float,
            )
            contact += self.mesh.to_full_dof(self.friction_potential.gradient(tangential, self.mesh, delta))
        gradient = gradient + contact - original_contact
        # The original tangent is retained as a numerical Newton approximation;
        # acceptance uses only the newly reconstructed complete force residual.
        return energy, gradient, matrix, normal, contact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    static, config = restore_static(args.run)
    original, system = WideSpatialInertia(static, config["dt"]), PreciseStencilSpatial(static, config["dt"])
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    with np.load(paths[-1]) as saved:
        previous = np.asarray(saved["displacement"], dtype=np.longdouble)
        velocity = np.asarray(saved["velocity"], dtype=np.longdouble)
    with np.load(args.run / "results/failure.npz") as saved:
        displacement = np.asarray(saved["displacement"], dtype=np.longdouble)
        np.testing.assert_array_equal(saved["previous_displacement"], previous)
    for candidate in [original, system]:
        candidate.q_target[:] = 0.018
        candidate.predicted_displacement = previous + config["dt"] * velocity
    mapping = coordinate_map(system)
    external = system.gravity_force([0, 0, -9.81])
    before = original.evaluate_updated_friction(displacement, previous, external, False)
    after = system.evaluate_updated_friction(displacement, previous, external, True)
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "body_tetrahedra": system.body.mesh.nelements,
        "original_complete_residual_N": float(np.linalg.norm(mapping.T @ before[1])),
        "precise_complete_residual_N": float(np.linalg.norm(mapping.T @ after[1])),
        "contact_force_difference_N": float(np.linalg.norm(before[4] - after[4])),
        "directional_checks": [],
        "goal_completion": False,
        "limitations": "Unchanged geometry/material/contact law and original force budgets. Centered per-stencil arithmetic and direct relative increments change force evaluation precision only. The original full tangent is a numerical approximation whose directional errors are retained. Successful same-state recovery still requires an independent precision-aware force audit and full remaining history; no spatial/contact convergence is implied.",
    }
    direction = np.asarray(mapping @ np.random.default_rng(405).normal(size=mapping.shape[1]), dtype=np.longdouble)
    direction /= np.linalg.norm(direction)
    actual = after[2] @ np.asarray(direction, dtype=float)
    for h in [1e-11, 3e-12, 1e-12]:
        plus = system.evaluate_updated_friction(displacement + h * direction, previous, external, False)[1]
        minus = system.evaluate_updated_friction(displacement - h * direction, previous, external, False)[1]
        numerical = (plus - minus) / (2 * h)
        error = np.linalg.norm(mapping.T @ (numerical - actual)) / np.linalg.norm(mapping.T @ actual)
        result["directional_checks"].append({"step_m": h, "relative_reduced_tangent_difference": float(error)})
    (args.output / "before_recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.solve:
        try:
            u, gradient, force, _energy, _iterations = system.solve_updated_friction(
                np.array([], dtype=int),
                np.array([]),
                1,
                tolerance=1e-10,
                initial_displacement=displacement,
                initial_previous_displacement=previous,
                coordinate_map=mapping,
                applied_force=external,
                initial_force=external,
                phase="precise_stencil_recovery",
                failure_path=args.output / "failure.npz",
            )
            result.update(
                recovered=True,
                recovered_complete_residual_N=float(np.linalg.norm(mapping.T @ gradient)),
                recovered_body_residual_N=float(np.linalg.norm(gradient[: system.body.basis.N])),
            )
            assert result["recovered_complete_residual_N"] <= 1e-9 and result["recovered_body_residual_N"] <= 1e-9
            np.savez_compressed(
                args.output / "recovered.npz",
                displacement=u,
                contact_force_N=force,
                previous_displacement=previous,
                previous_velocity=velocity,
            )
        except RuntimeError as error:
            result.update(recovered=False, error=str(error))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
