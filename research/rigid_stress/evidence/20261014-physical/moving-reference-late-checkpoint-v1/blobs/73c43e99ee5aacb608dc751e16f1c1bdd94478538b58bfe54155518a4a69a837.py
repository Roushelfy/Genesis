"""Measure force invariance and residual components at the unchanged failed spatial state."""

import argparse
import json
from pathlib import Path

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
