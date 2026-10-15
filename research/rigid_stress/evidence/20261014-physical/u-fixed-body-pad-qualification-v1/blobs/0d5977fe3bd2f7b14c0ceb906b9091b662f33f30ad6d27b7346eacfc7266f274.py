"""Fit a finite rigid body frame using independent consistent-mass integration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .neohookean_reference import NeoHookeanBulk
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def finite_frame(reference, displacement):
    """Minimize the integrated squared distance to a proper rigid transform."""
    scalar_mass = reference.mass[reference.component_dofs[:, 0]][:, reference.component_dofs[:, 0]]
    weights = np.asarray(scalar_mass.sum(axis=1)).ravel()
    current = reference.coordinates + displacement[reference.component_dofs]
    centre = weights @ current / reference.total_mass
    rest_relative = reference.coordinates - reference.com
    current_relative = current - centre
    covariance = current_relative.T @ (scalar_mass @ rest_relative)
    left, _, right = np.linalg.svd(covariance)
    proper = np.eye(3)
    proper[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ proper @ right
    translation = centre - rotation @ reference.com
    local = np.zeros(reference.basis.N)
    local[reference.component_dofs] = current_relative @ rotation - rest_relative
    return rotation, translation, local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "solution/configuration.json").read_text())
        with np.load(config["mesh"]) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        with np.load(run / "solution/fields.npz") as saved:
            u = np.empty(reference.basis.N)
            u[saved["body_component_permutation"]] = saved["displacement"][: reference.basis.N]
        known_angle = 0.7
        known = np.array(
            [[np.cos(known_angle), -np.sin(known_angle), 0], [np.sin(known_angle), np.cos(known_angle), 0], [0, 0, 1]]
        )
        rigid = np.zeros(reference.basis.N)
        rigid[reference.component_dofs] = reference.coordinates @ (known - np.eye(3)).T + [0.004, -0.002, 0.001]
        fitted, shift, local = finite_frame(reference, rigid)
        np.testing.assert_allclose(fitted, known, rtol=0, atol=1e-12)
        np.testing.assert_allclose(shift, [0.004, -0.002, 0.001], rtol=0, atol=1e-14)
        assert np.linalg.norm(local[reference.component_dofs], axis=1).max() < 1e-14
        rotation, translation, local = finite_frame(reference, u)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-12)
        # The fitted finite transform must not create or remove bulk energy.
        bulk = NeoHookeanBulk(reference)
        raw_energy, _, _ = bulk.evaluate(u, False)
        local_energy, _, _ = bulk.evaluate(local, False)
        np.testing.assert_allclose(raw_energy, local_energy, rtol=1e-8, atol=1e-15)
        local_gradients = reference.corner_basis.interpolate(local).grad.transpose(2, 3, 0, 1)
        strain = (local_gradients + local_gradients.swapaxes(-1, -2)) / 2
        infinitesimal = reference.align_rigid_gauge(u)
        np.savez_compressed(
            args.output / f"{run.name}_body_frame.npz",
            rotation_world_from_material=rotation,
            translation_world_m=translation,
            objective_displacement_material_m=local,
            infinitesimal_gauge_displacement_m=infinitesimal,
        )
        rows.append(
            {
                "run": str(run),
                "finite_rotation_angle_rad": float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))),
                "rotation_world_from_material": rotation.tolist(),
                "translation_world_m": translation.tolist(),
                "objective_maximum_nodal_displacement_m": float(
                    np.linalg.norm(local[reference.component_dofs], axis=1).max()
                ),
                "infinitesimal_maximum_nodal_displacement_m": float(
                    np.linalg.norm(infinitesimal[reference.component_dofs], axis=1).max()
                ),
                "maximum_material_symmetric_strain_norm": float(np.linalg.norm(strain, axis=(-2, -1)).max()),
                "raw_nonlinear_bulk_energy_J": raw_energy,
                "fitted_frame_nonlinear_bulk_energy_J": local_energy,
                "saved_field_sha256": hashlib.sha256((run / "solution/fields.npz").read_bytes()).hexdigest(),
            }
        )
    summary = {
        "scope": __doc__,
        "rows": rows,
        "goal_completion": False,
        "limitations": "A proper finite rigid-frame fit preserves the saved physical geometry and objective bulk energy. It does not turn a saved linear state into a solved nonlinear contact state. Native force/stress transforms, complete source moments and nonlinear common probes must use the corresponding frame and retain deformation-induced geometry/moment differences.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
