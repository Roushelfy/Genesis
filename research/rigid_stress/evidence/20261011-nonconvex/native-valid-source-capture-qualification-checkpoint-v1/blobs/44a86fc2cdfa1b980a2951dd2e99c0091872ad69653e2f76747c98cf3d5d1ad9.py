"""Verify closed dynamic pairs, complete workload counters and assigned-device telemetry."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "family",
        choices=(
            "valid-source-capture",
            "coalesced-positions",
            "fp32-warm-jacobian",
            "active-raw-diagnostics",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    if args.family == "valid-source-capture":
        configurations, version, flag = ((8192, 734001), (16384, 991909)), 1, "valid_source_capture_only"
    elif args.family == "fp32-warm-jacobian":
        configurations, version, flag = ((16384, 623102),), 1, "fp32_warm_jacobian"
    elif args.family == "active-raw-diagnostics":
        configurations, version, flag = ((8192, 991909),), 1, "active_raw_diagnostics"
    else:
        configurations, version, flag = ((16384, 623102), (8192, 734001)), 1, "sample_contiguous_cached_positions"
    summaries = []
    rigid_keys = (
        "observed_steps",
        "lift_window_steps",
        "lift_grasp_pass_steps",
        "lift_grasp_fail_steps",
        "completed_lift_success_cycles",
        "completed_lift_failure_cycles",
        "weak_grip_steps",
        "weak_grip_descents",
        "release_steps",
        "release_descents",
        "two_finger_contact_steps",
    )
    for batch, seed in configurations:
        values, routes = {}, {}
        for mode in ("off", "on"):
            run = root / f"native-{args.family}-v{version}/b{batch}-varied-live-{mode}-{seed}"
            assert (run / "returncode.txt").read_text().strip() == "0"
            assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
            result = json.loads((run / "result.json").read_text())
            assert result[flag] == (mode == "on")
            assert result["envs"] == batch and result["seed"] == seed and result["scope"] == "live"
            assert result["dt"] == 0.01 and result["substeps"] == 1 and result["precision"] == "64"
            assert result["intrinsic_load_partition"] == "body_source_prior" and len(result["repeats"]) == 3
            for repeat, metrics in zip(result["repeats"], result["trajectory_metrics"]["repeats"]):
                assert (
                    repeat["steps"] == 1280
                    and repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
                )
                assert max(repeat["failed_steps_per_environment"]) == 0
                assert metrics["totals"]["observed_steps"] == batch * 1280
                assert metrics["totals"]["invalid_stress_steps"] == metrics["totals"]["lift_grasp_fail_steps"] == 0
                assert metrics["totals"]["completed_lift_failure_cycles"] == 0
            budgets = [
                float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
                for row in result["warmup_quality"]
            ]
            assert max(budgets) <= 1
            telemetry = list(
                csv.DictReader((run / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True)
            )
            assert len(telemetry) > 100 and all(row["uuid"] == result["gpu_uuid"] for row in telemetry)
            rate = batch * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
            routes[mode] = {
                "env_steps_per_second": rate,
                "batch_steps_per_second": rate / batch,
                "repeat_seconds": [row["seconds"] for row in result["repeats"]],
                "native_buffer_bytes": result["native_buffer_bytes"],
                "gpu_uuid": result["gpu_uuid"],
                "sampled_device_peak_MiB_including_setup": max(
                    int(row["memory.used [MiB]"].split()[0]) for row in telemetry
                ),
                "warmup_full_residual_budget_fraction_max": max(budgets),
                "resets": [row["resets"] for row in result["repeats"]],
                "trajectory_totals": [row["totals"] for row in result["trajectory_metrics"]["repeats"]],
            }
            values[mode] = result
        assert values["off"]["gpu_uuid"] == values["on"]["gpu_uuid"]
        position_difference = float(
            np.max(abs(np.array(values["on"]["egg_position_m"]) - np.array(values["off"]["egg_position_m"])))
        )
        assert position_difference <= 1e-12
        assert routes["off"]["resets"] == routes["on"]["resets"]
        for baseline, candidate in zip(
            values["off"]["trajectory_metrics"]["repeats"], values["on"]["trajectory_metrics"]["repeats"]
        ):
            for key in rigid_keys:
                assert baseline["per_environment"][key] == candidate["per_environment"][key]
        if args.family != "compact-cached-positions":
            base_run = root / f"native-{args.family}-v{version}/b{batch}-varied-live-off-{seed}"
            on_run = root / f"native-{args.family}-v{version}/b{batch}-varied-live-on-{seed}"
            with np.load(base_run / "conditions.npz") as original, np.load(on_run / "conditions.npz") as candidate:
                assert original.files == candidate.files
                for name in original.files:
                    assert np.array_equal(original[name], candidate[name])
        summaries.append(
            {
                "batch": batch,
                "seed": seed,
                "routes": routes,
                "live_speedup": routes["on"]["env_steps_per_second"] / routes["off"]["env_steps_per_second"],
                "native_bytes_saved": routes["off"]["native_buffer_bytes"] - routes["on"]["native_buffer_bytes"],
                "rigid_position_difference_m_max": position_difference,
                "identical_per_environment_rigid_contact_grasp_counters": True,
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "family": args.family,
                "pairs": summaries,
                "limitations": "200ms device memory samples include setup and are not exact transient peaks. These three-repeat pairs are separate decisions; small sub-percent ratios need repeat/drift evidence before selection. Numerical fixture qualifications are separate and required.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "family": args.family,
                "pairs": [
                    {
                        "batch": row["batch"],
                        "speedup": row["live_speedup"],
                        "native_bytes_saved": row["native_bytes_saved"],
                    }
                    for row in summaries
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
