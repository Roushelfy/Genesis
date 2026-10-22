"""Compare retained native source fields and matched scene timing captures."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--paired", type=Path, required=True)
    cli.add_argument("--controls", type=Path, required=True)
    cli.add_argument("--baseline-controls", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bindings, fields, timing = [], [], []
    for group in [0, 1]:
        current = args.controls / f"group{group}/rotated"
        baseline = args.baseline_controls / f"group{group}/check"
        for frame in range(8):
            paths = [baseline / f"frame{frame}.npz", current / f"frame{frame}.npz"]
            bindings.extend(file_id(path) for path in paths)
            with np.load(paths[0]) as expected, np.load(paths[1]) as actual:
                assert set(expected.files) == set(actual.files)
                for key in expected.files:
                    left, right = expected[key], actual[key]
                    if np.issubdtype(left.dtype, np.inexact):
                        np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
                        if key.endswith("_pa"):
                            rtol, atol = 1e-7, 1e-7
                        elif key.endswith("_m"):
                            rtol, atol = 0, 2e-16
                        elif key.endswith("_force_n") or key.endswith("_error_n"):
                            rtol, atol = 1e-12, 1e-14
                        elif key.endswith("_moment_nm") or key.endswith("_error_nm"):
                            rtol, atol = 1e-12, 1e-18
                        else:
                            rtol, atol = 1e-12, 1e-14
                        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol, equal_nan=True, err_msg=key)
                    else:
                        np.testing.assert_array_equal(left, right, err_msg=key)
                difference = actual["probe_tensor_pa"] - expected["probe_tensor_pa"]
                finite = np.isfinite(difference)
                fields.append(
                    {
                        "group": group,
                        "frame": frame,
                        "probe_tensor_relative": float(
                            np.linalg.norm(difference[finite])
                            / max(np.linalg.norm(expected["probe_tensor_pa"][finite]), 1)
                        ),
                        "all_source_and_field_arrays_checked": len(expected.files),
                    }
                )
    pair = args.paired / "group0/paired"
    results = [json.loads((pair / label / "result.json").read_text()) for label in ["reference", "current"]]
    assert all(row["passed"] for row in results)
    for left, right in zip(results[0]["rows"], results[1]["rows"]):
        batch, mode = left["batch"], left["mode"]
        assert (batch, mode) == (right["batch"], right["mode"])
        paths = [pair / label / f"{mode}-b{batch}-audit.npz" for label in ["reference", "current"]]
        bindings.extend(file_id(path) for path in paths)
        with np.load(paths[0]) as expected, np.load(paths[1]) as actual:
            np.testing.assert_allclose(expected["positions_m"], actual["positions_m"], rtol=0, atol=1e-12)
            if mode != "rigid":
                for key in ["vm_pa", "principal_pa"]:
                    np.testing.assert_allclose(expected[key], actual[key], rtol=1e-7, atol=1e-7)
                for key in ["input_status", "field_status", "loaded"]:
                    np.testing.assert_array_equal(expected[key], actual[key])
                assert (
                    right["admitted_audit_environment_steps"] == left["admitted_audit_environment_steps"] == 24 * batch
                )
        old, new = left["seconds_per_scene_step"], right["seconds_per_scene_step"]
        timing.append(
            {
                "batch": batch,
                "mode": mode,
                "baseline_seconds": old,
                "bounded_seconds": new,
                "baseline_median_seconds": left["median_seconds"],
                "bounded_median_seconds": right["median_seconds"],
                "observed_median_reduction": 1 - right["median_seconds"] / left["median_seconds"],
                "repeat_ranges_separate": max(new) < min(old) or max(old) < min(new),
                "admitted_environment_steps": right["admitted_audit_environment_steps"],
                "admitted_loaded_environment_steps": right["admitted_loaded_audit_steps"],
            }
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rotated_complete_source_field_checks": fields,
            "same_allocation_timings": timing,
            "bindings": bindings,
            "scope": "Implementation-only source pruning, preserving all saved provenance/geometry/load/status/full-field/probe arrays under predeclared numerical scales, plus matched actual rigid/max/full scene trajectories, peaks and acceptance. Repeat ranges are observations from three blocks, not confidence intervals. General friction and curved-material physics remain separate requirements.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
