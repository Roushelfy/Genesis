"""Independently inspect same-allocation serial/cooperative memory results."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--paired", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    limits = {"line_absolute": 1e-7, "coefficient_absolute": 1e-5, "force_residual": 1e-9, "slope_bound": 1 + 1e-10}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "limits": limits,
            "scope": "Read-only numerical layout and actual benchmark admission inspection. No independent general friction physics or scene timing.",
        },
    )
    assert (args.paired / "group0/returncode.txt").read_text().strip() == "0"
    assert (args.paired / "group1/returncode.txt").read_text().strip() == "0"
    warp, serial = [args.paired / "group0" / name for name in ["warp", "serial"]]
    results = [json.loads((path / "result.json").read_text()) for path in [warp, serial]]
    starts = [json.loads((path / "start.json").read_text()) for path in [warp, serial]]
    assert all(result["passed"] for result in results)
    assert starts[0]["identity"] == starts[1]["identity"]
    conformance, bindings = [], []
    with np.load(warp / "conformance.npz") as first, np.load(serial / "conformance.npz") as second:
        assert first.files == second.files
        for count in [16, 32]:
            line = float(np.max(np.abs(first[f"n{count}_line"] - second[f"n{count}_line"])))
            coefficients = float(np.max(np.abs(first[f"n{count}_coefficients"] - second[f"n{count}_coefficients"])))
            np.testing.assert_array_equal(first[f"n{count}_status"], second[f"n{count}_status"])
            assert line < limits["line_absolute"] and coefficients < limits["coefficient_absolute"]
            assert np.all(first[f"n{count}_status"] == 0)
            conformance.append(
                {
                    "intervals": count,
                    "line_maximum_difference": line,
                    "coefficient_maximum_difference": coefficients,
                    "maximum_iteration_difference": int(
                        np.max(np.abs(first[f"n{count}_iterations"] - second[f"n{count}_iterations"]))
                    ),
                }
            )
    timings = []
    for index, row in enumerate(results[0]["benchmark"]):
        count, batch = row["intervals"], row["batch"]
        baseline = results[1]["benchmark"][index]
        assert (count, batch) == (baseline["intervals"], baseline["batch"])
        checks = []
        for path, declared in zip([warp, serial], [row, baseline]):
            capture = path / f"benchmark-n{count}-b{batch}.npz"
            bindings.append(file_id(capture))
            with np.load(capture) as data:
                assert data["status"].shape == (64, batch)
                assert np.all(data["status"] == 0)
                assert np.isfinite(data["residual"]).all() and np.isfinite(data["bound"]).all()
                assert (
                    data["residual"].max() < limits["force_residual"] and data["bound"].max() <= limits["slope_bound"]
                )
                assert declared["admitted_updates"] == data["status"].size
                assert declared["maximum_audited_iterations"] == int(data["iterations"].max())
                checks.append(
                    {
                        "admitted_updates": int(data["status"].size),
                        "maximum_force_residual": float(data["residual"].max()),
                        "maximum_continuous_slope_bound": float(data["bound"].max()),
                    }
                )
            assert declared["median_seconds_per_update"] == float(np.median(declared["seconds_per_update"]))
        timings.append(
            {
                "intervals": count,
                "batch": batch,
                "warp_seconds": row["median_seconds_per_update"],
                "serial_seconds": baseline["median_seconds_per_update"],
                "reduction_fraction": 1 - row["median_seconds_per_update"] / baseline["median_seconds_per_update"],
                "three_repeat_ranges_separate": max(row["seconds_per_update"]) < min(baseline["seconds_per_update"]),
                "warp_serial_admission": checks,
            }
        )
    for path in [warp, serial]:
        bindings.extend(file_id(path / name) for name in ["conformance.npz", "result.json", "start.json"])
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "same_allocation_identity": starts[0]["identity"],
            "bindings": bindings,
            "conformance": conformance,
            "timings": timings,
            "limits": limits,
            "independent_general_vector_friction_physics_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
