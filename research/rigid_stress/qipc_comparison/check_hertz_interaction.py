"""Test one cheap far-field curvature/center correction on retained two-region hard contact."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .hertz_reference import ellipse, ellipse_pressure, halfspace_stress


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = args.study / "result.json"
    original = json.loads(receipt.read_text())
    assert original["passed"]
    rows = []
    for case in original["rows"]:
        if case["case"] != "two_regions":
            continue
        path = Path(case["arrays"]["path"])
        assert file_id(path) == case["arrays"]
        with np.load(path) as saved:
            points, reference_force, observation, reference_stress = (
                saved[k] for k in ("points_m", "reference_force_n", "observation_points_m", "reference_stress_pa")
            )
        width = np.unique(points[:, 0])[1] - points[0, 0]
        modulus, load = 1e10 / (1 - 0.3**2), case["load_n"]
        curvature = np.array(case["curvature_per_m"])
        axes, _, _, _ = ellipse(load / 2, modulus, curvature)
        centers = np.array([[-1.5 * axes[0], 0], [1.5 * axes[0], 0]])
        gauss, weights = np.polynomial.legendre.leggauss(8)
        offset = np.array(np.meshgrid(gauss, gauss, indexing="ij")).reshape(2, -1).T * width / 2
        sample = (points[:, None] + offset[None]).reshape(-1, 2)
        weights = (weights[:, None] * weights[None, :]).reshape(-1) / 4
        force = np.zeros(len(points))
        changes = []
        for i in range(2):
            # The other finite patch is approximated by its resultant in the
            # Boussinesq surface kernel. Only one Taylor update is applied.
            relative = centers[i] - centers[1 - i]
            distance = np.linalg.norm(relative)
            coefficient = load / (2 * np.pi * modulus)
            gradient = -coefficient * relative / distance**3
            hessian = coefficient * (3 * np.outer(relative, relative) / distance**5 - np.eye(2) / distance**3)
            effective = curvature + hessian
            shift = -np.linalg.solve(effective, gradient)
            a, direction, _, peak = ellipse(load / 2, modulus, effective)
            pressure = ellipse_pressure(sample - centers[i] - shift, a, direction, peak)
            part = pressure.reshape(len(points), -1) @ weights * width**2
            part *= load / (2 * part.sum())
            force += part
            changes.append({"shift_m": shift.tolist(), "axes_m": a.tolist(), "curvature_per_m": effective.tolist()})
        stress = halfspace_stress(observation, points, force)
        path = args.output / f"interaction-{case['grid']}.npz"
        np.savez_compressed(path, points_m=points, force_n=force, stress_pa=stress)
        rows.append(
            {
                "grid": case["grid"],
                "pressure_area_l2_relative": float(
                    np.linalg.norm(force - reference_force) / np.linalg.norm(reference_force)
                ),
                "subsurface_sample_tensor_relative": float(
                    np.linalg.norm(stress - reference_stress) / np.linalg.norm(reference_stress)
                ),
                "changes": changes,
                "arrays": file_id(path),
                "reference": case["arrays"],
            }
        )
    assert len(rows) == 2
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "source": file_id(receipt),
            "rows": rows,
            "scope": "One far-field point-resultant Taylor correction of centers and relative curvature, using two known symmetric physical-region loads. No fit to reference pressure, extra contact solve, shell correction, asymmetric load allocation or general multiple-region acceptance.",
        },
    )


if __name__ == "__main__":
    main()
