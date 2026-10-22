"""Compare complete curved-shell fields before and after remote edge quadrature."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_edge_fields import field_metrics
from .study_hertz_residual import field_difference


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.baseline), file_id(args.candidate)],
            "targets": {"complete_tensor_change": 0.005},
            "scope": "Compare identical curved-shell load configurations and sample locations under the remote quadrature approximation. Coarse volume integration and separate near-source probes retain their original resolution. This isolates implementation/model approximation change from independent physical error.",
        },
    )
    with np.load(args.baseline) as first, np.load(args.candidate) as second:
        np.testing.assert_array_equal(first["points_m"], second["points_m"])
        np.testing.assert_array_equal(first["volume_weights_m3"], second["volume_weights_m3"])
        np.testing.assert_array_equal(first["displacement_m"], second["displacement_m"])
        weights = first["volume_weights_m3"]
        rows = []
        for a, b in zip(first["complete_pa"], second["complete_pa"]):
            relative = np.linalg.norm(b - a, axis=(1, 2)) / np.maximum(np.linalg.norm(a, axis=(1, 2)), 1e-30)
            rows.append(
                {
                    "volume_tensor_relative": float(field_difference(b[: len(weights)], a[: len(weights)], weights)),
                    "common_probes": field_metrics(b[len(weights) :], a[len(weights) :]),
                    "all_samples": field_metrics(b, a),
                    "maximum_point_tensor_relative": float(relative.max()),
                }
            )
    passed = all(
        row["volume_tensor_relative"] < 0.005 and row["all_samples"]["tensor_relative"] < 0.005 for row in rows
    )
    write_json(
        args.output / "result.json",
        {"passed": passed, "cases": rows, "regular_displacement_identical": True, "whole_goal_complete": False},
    )
    assert passed


if __name__ == "__main__":
    main()
