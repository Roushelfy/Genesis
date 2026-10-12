"""Publish counted-sample position reuse with unchanged final numerical budgets."""

import ast
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / f"native-cached-fan-positions-v2/b2048-varied-live-{mode}-623102" for mode in ("off", "on")]
results, summaries = {}, {}
rigid_keys = (
    "observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
    "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps", "weak_grip_descents",
    "release_steps", "release_descents", "two_finger_contact_steps",
)
for mode, run in zip(("off", "on"), runs):
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    value = json.loads((run / "result.json").read_text())
    assert value["cached_fan_positions"] == (mode == "on")
    assert value["envs"] == 2048 and value["scope"] == "live" and value["seed"] == 623102
    assert value["dt"] == .01 and value["substeps"] == 1 and value["precision"] == "64"
    assert value["intrinsic_load_partition"] == "body_source_prior" and len(value["repeats"]) == 3
    for repeat in value["repeats"]:
        assert repeat["steps"] == 1280 and repeat["resets"] == 2323
        assert repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        assert max(repeat["failed_steps_per_environment"]) == 0
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in value["warmup_quality"]]
    assert max(budgets) <= 1
    metrics = value["trajectory_metrics"]
    assert len(metrics["repeats"]) == 3
    for row in metrics["repeats"]:
        assert row["totals"]["observed_steps"] == 2048 * 1280
        assert row["totals"]["invalid_stress_steps"] == row["totals"]["lift_grasp_fail_steps"] == 0
        assert row["totals"]["completed_lift_success_cycles"] == 3142
        assert row["totals"]["completed_lift_failure_cycles"] == 0
    telemetry = list(csv.DictReader((run / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True))
    assert len(telemetry) > 100 and all(row["uuid"] == value["gpu_uuid"] for row in telemetry)
    sampled_peak = max(int(row["memory.used [MiB]"].split()[0]) for row in telemetry)
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
    summaries[mode] = {
        "effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
        "repeat_seconds": [row["seconds"] for row in value["repeats"]],
        "repeat_env_steps_per_second": [2048 * 1280 / row["seconds"] for row in value["repeats"]],
        "native_buffer_bytes": value["native_buffer_bytes"], "gpu_uuid": value["gpu_uuid"],
        "setup_seconds": value["setup_seconds"], "sampled_assigned_device_peak_MiB_including_setup": sampled_peak,
        "warmup_full_residual_budget_fraction_max": max(budgets),
        "trajectory_totals": [row["totals"] for row in metrics["repeats"]],
    }
    results[mode] = value
assert len({row["gpu_uuid"] for row in results.values()}) == 1
assert np.max(abs(np.array(results["on"]["egg_position_m"]) - np.array(results["off"]["egg_position_m"]))) <= 1e-12
for baseline, candidate in zip(results["off"]["trajectory_metrics"]["repeats"], results["on"]["trajectory_metrics"]["repeats"]):
    for key in rigid_keys:
        assert baseline["per_environment"][key] == candidate["per_environment"][key]
fields, regression_summary, cached_differences = [], {}, []
for backend in ("cpu", "cuda"):
    run = root / f"native-cached-fan-positions-regression-v2/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "31" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    backend_fields = []
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            if "cached_position_max_difference_m" in values:
                assert backend == "cuda" and values["cached_position_max_difference_m"] <= 1e-15
                cached_differences.append(values["cached_position_max_difference_m"])
            backend_fields.append({"backend": backend, "case": case.attrib["name"], **values})
    assert len(backend_fields) == 16 and sum(len(row["native_peak_Pa"]) for row in backend_fields) == 48
    fields.extend(backend_fields)
    regression_summary[backend] = {"passed": 31, "seconds": float(suite.attrib["time"]), "independent_full_fields": 48}
    runs.append(run)
assert len(cached_differences) == 8
failed_runs = [root / f"native-cached-fan-positions-regression-v1/{backend}" for backend in ("cpu", "cuda")]
failed_runs.append(root / "native-cached-fan-positions-v1/b2048-varied-live-off-623102")
for run in failed_runs:
    assert (run / "returncode.txt").read_text().strip() == "1"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
runs.extend(failed_runs)
summary = {
    "live": summaries, "live_speedup": summaries["on"]["effective_env_steps_per_second"] / summaries["off"]["effective_env_steps_per_second"],
    "identical_rigid_positions_and_per_environment_contact_grasp_counters": True,
    "regression": regression_summary, "recorded_independent_full_fields": 96, "fields": fields,
    "cached_vs_direct_position_case_count": 8, "cached_vs_direct_position_difference_m_max": max(cached_differences),
    "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
    "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
    "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
    "failed_first_version": "Nested static-branch face-index scope; CPU/CUDA18 failed13 passed and baseline benchmark warmup compile failure, with source closures preserved.",
}
extras = [root / name for name in (
    "archive_native_cached_fan_positions_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_cached_fan_positions_v1.sh",
    "run_cached_fan_positions_pair_v2.sh", "run_cached_fan_positions_pair_v1.sh",
    "native-cached-fan-positions-pair-v2-allocation.log", "native-cached-fan-positions-tests-v2-cpu-allocation.log",
    "native-cached-fan-positions-tests-v2-cuda-allocation.log",
)]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-cached-fan-positions-checkpoint-v1"), runs, extras, summary,
        "Private basea6dc0fbc plus selected eight-sample-lane retry and qualified lossless producer-counted sample storage. This pair changes only current-step physical sample position caching; topology, charts, radius, friction, complete sample cloud, timestep, final wrench/gauge residual/stress budgets are fixed. CUDA native-entry fixtures compare every selected cached position with direct chart reconstruction before independent CPU FP64 fitting/nodal/full tensor/global peak checks. CPU retains full storage. Both routes include600 warmup and three1280-step varied live repeats, all environments/reset/observations/metrics/history/fallback.200ms assigned-device memory samples include setup and trajectory and cannot establish exact transient peak. This isolated cache comparison does not establish the combined compact/cache gain versus production50b1b61c. Wider matched scopes, separated seeds, remaining applicable candidates and a throughput plateau are pending. No physical convergence or RL training claim.")
