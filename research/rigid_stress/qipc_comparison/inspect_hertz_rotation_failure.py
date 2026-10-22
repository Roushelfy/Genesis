"""Read failed rotation arrays without rerunning the native recovery."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_hertz_residual import elliptic_stress, field_difference


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base", type=Path, required=True)
    cli.add_argument("--rotated", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.base) as data:
        base = {key: data[key] for key in data.files}
    with np.load(args.rotated) as data:
        rotated = {key: data[key] for key in data.files}
    rotation = rotated["frames"][0] @ base["frames"][0].T
    translation = rotated["centers_m"][0] - rotation @ base["centers_m"][0]
    point_error = float(np.max(np.abs(base["positions_m"] @ rotation.T + translation - rotated["positions_m"])))
    expected = rotation @ base["tensor_pa"] @ rotation.T
    expected_corners = rotation @ base["corner_tensor_pa"] @ rotation.T
    rows = []
    nv = len(base["volume_weights_m3"])
    for env in range(4):
        field = float(np.linalg.norm(rotated["tensor_pa"][env] - expected[env]) / max(np.linalg.norm(expected[env]), 1))
        corners = float(
            np.linalg.norm(rotated["corner_tensor_pa"][env] - expected_corners[env])
            / max(np.linalg.norm(expected_corners[env]), 1)
        )
        volume = (
            float(field_difference(rotated["tensor_pa"][env, :nv], expected[env, :nv], base["volume_weights_m3"]))
            if np.linalg.norm(expected[env, :nv])
            else 0.0
        )
        rows.append(
            {
                "env": env,
                "native_field_rotation_relative": field,
                "native_corner_rotation_relative": corners,
                "native_volume_rotation_relative": volume,
            }
        )
    local = base["positions_m"] - base["centers_m"][0]
    transformed = (rotated["positions_m"] - rotated["centers_m"][0]) @ rotation
    axes, peak = base["hertz_axes_m"][0], base["hertz_pressure_pa"][0]
    mask = (local[:, 2] == 0) & (np.sum((local[:, :2] / axes) ** 2, axis=1) < 0.99)
    actual_coordinates = transformed[mask].copy()
    actual_coordinates[:, 2] = np.maximum(actual_coordinates[:, 2], 0)
    actual = elliptic_stress(actual_coordinates, axes, peak=peak, order=16)
    limit = elliptic_stress(local[mask], axes, peak=peak, order=16)
    difference = np.linalg.norm(actual - limit, axis=(-1, -2)) / peak
    np.savez_compressed(
        args.output / "surface-reference.npz",
        canonical=local[mask],
        transformed=transformed[mask],
        reference_transformed=actual,
        reference_surface=limit,
        difference_over_p0=difference,
    )
    passed = point_error < 1e-12 and max(row["native_field_rotation_relative"] for row in rows) < 1e-7
    write_json(
        args.output / "result.json",
        {
            "passed": passed,
            "identity": runtime_identity(require_gpu_inventory=False),
            "base": file_id(args.base),
            "rotated": file_id(args.rotated),
            "maximum_position_roundoff_m": point_error,
            "rows": rows,
            "nominal_surface_probe_count": int(mask.sum()),
            "positive_transformed_depth_count": int(np.count_nonzero(transformed[mask, 2] > 0)),
            "maximum_surface_reference_change_over_p0": float(difference.max()),
            "maximum_surface_depth_roundoff_m": float(np.abs(transformed[mask, 2]).max()),
            "scope": "Read-only native tensor covariance and nominal-surface reference-coordinate diagnostic; it does not qualify general contact physics.",
            "whole_goal_complete": False,
        },
    )
    assert passed, rows


if __name__ == "__main__":
    main()
