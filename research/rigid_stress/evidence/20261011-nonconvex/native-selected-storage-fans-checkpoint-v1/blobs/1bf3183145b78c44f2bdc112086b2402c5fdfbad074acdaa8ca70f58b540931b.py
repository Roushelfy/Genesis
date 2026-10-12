"""Qualify cleaned counted positions and cooperative fan sampling before publication."""

import ast
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs, results, routes = [], {}, {}
rigid_keys = (
    "observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
    "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
    "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps",
)
for mode in ("baseline", "candidate"):
    run = root / f"native-compact-cache-fan-selected-live-v1/b2048-{mode}-623102"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    result = json.loads((run / "result.json").read_text())
    assert result["envs"] == 2048 and result["seed"] == 623102 and result["scope"] == "live"
    assert result["dt"] == .01 and result["substeps"] == 1 and result["precision"] == "64"
    assert result["intrinsic_load_partition"] == "body_source_prior" and len(result["repeats"]) == 3
    for repeat, metrics in zip(result["repeats"], result["trajectory_metrics"]["repeats"]):
        assert repeat["steps"] == 1280 and repeat["resets"] == 2323
        assert repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        assert max(repeat["failed_steps_per_environment"]) == 0
        assert metrics["totals"]["observed_steps"] == 2048 * 1280
        assert metrics["totals"]["invalid_stress_steps"] == metrics["totals"]["lift_grasp_fail_steps"] == 0
        assert metrics["totals"]["completed_lift_failure_cycles"] == 0
        assert metrics["totals"]["completed_lift_success_cycles"] == 3142
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"]]
    assert max(budgets) <= 1
    telemetry = list(csv.DictReader((run / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True))
    assert len(telemetry) > 100 and all(row["uuid"] == result["gpu_uuid"] for row in telemetry)
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
    routes[mode] = {
        "effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
        "repeat_seconds": [row["seconds"] for row in result["repeats"]],
        "repeat_env_steps_per_second": [2048 * 1280 / row["seconds"] for row in result["repeats"]],
        "native_buffer_bytes": result["native_buffer_bytes"], "gpu_uuid": result["gpu_uuid"],
        "sampled_assigned_device_peak_MiB_including_setup": max(int(row["memory.used [MiB]"].split()[0]) for row in telemetry),
        "warmup_full_residual_budget_fraction_max": max(budgets),
        "resets": [row["resets"] for row in result["repeats"]],
        "trajectory_totals": [row["totals"] for row in result["trajectory_metrics"]["repeats"]],
    }
    results[mode] = result
    runs.append(run)
assert results["baseline"]["gpu_uuid"] == results["candidate"]["gpu_uuid"]
position_difference = float(np.max(abs(np.array(results["candidate"]["egg_position_m"]) - np.array(results["baseline"]["egg_position_m"]))))
assert position_difference <= 1e-12
for baseline, candidate in zip(results["baseline"]["trajectory_metrics"]["repeats"], results["candidate"]["trajectory_metrics"]["repeats"]):
    for key in rigid_keys:
        assert baseline["per_environment"][key] == candidate["per_environment"][key]
with np.load(runs[0] / "conditions.npz") as baseline, np.load(runs[1] / "conditions.npz") as candidate:
    assert baseline.files == candidate.files
    for key in baseline.files:
        assert np.array_equal(baseline[key], candidate[key])
fields, regression, cached_differences = [], {}, []
for backend in ("cpu", "cuda"):
    run = root / f"native-compact-cache-fan-selected-regression-v1/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "31" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    current_fields = []
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            properties = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(properties["force_errors"] + properties["moment_errors"]) <= 1e-8
            assert max(properties["complete_residual_budget_fraction"]) <= 1
            assert max(properties["peak_difference_Pa"]) <= 1e-3
            if "cached_position_max_difference_m" in properties:
                assert backend == "cuda" and properties["cached_position_max_difference_m"] <= 1e-15
                cached_differences.append(properties["cached_position_max_difference_m"])
            current_fields.append({"backend": backend, "case": case.attrib["name"], **properties})
    assert len(current_fields) == 16 and sum(len(row["native_peak_Pa"]) for row in current_fields) == 48
    fields.extend(current_fields)
    regression[backend] = {"passed": 31, "seconds": float(suite.attrib["time"]), "independent_full_fields": 48}
    runs.append(run)
    smoke = root / f"native-compact-cache-fan-selected-smoke-v1/{backend}"
    assert (smoke / "returncode.txt").read_text().strip() == "0"
    assert (smoke / "source.sha256").read_bytes() == (smoke / "source-after.sha256").read_bytes()
    runs.append(smoke)
assert len(cached_differences) == 8
summary = {
    "live": routes, "live_speedup": routes["candidate"]["effective_env_steps_per_second"] / routes["baseline"]["effective_env_steps_per_second"],
    "rigid_position_difference_m_max": position_difference,
    "identical_condition_banks_and_per_environment_rigid_contact_grasp_counters": True,
    "regression": regression, "recorded_independent_full_fields": 96, "fields": fields,
    "cached_vs_direct_position_case_count": 8, "cached_vs_direct_position_difference_m_max": max(cached_differences),
    "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
    "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
    "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
}
extras = [root / name for name in (
    "archive_native_selected_storage_fans_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_selected_storage_fans_v1.sh",
    "run_compact_cache_fan_selected_pair_v1.sh", "native-compact-cache-fan-selected-pair-v1-allocation.log",
    "native-compact-cache-fan-selected-tests-v1-cpu-allocation.log", "native-compact-cache-fan-selected-tests-v1-cuda-allocation.log",
)]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-selected-storage-fans-checkpoint-v1"), runs, extras, summary,
        "Clean production implementation combines exact producer-counted growth/current-step position reuse and CUDA cooperative refined fans; CPU retains serial full storage. Final wrench1e-8, complete gauge residualmax(1e-11N,1e-8||rhs||), stressrtol1e-4/atol1e-3Pa remain fixed.31 CPU/31 CUDA cases contain96 independent fields, native/manual fresh inputs and8 exact cached/direct checks. Same-GPU B2048/623102 pair compares prior qualified counted-position/serial-fan prototype with cleaned production combination,600 warmup and three1280-step varied live repeats with all environments/observations/contact updates/reset/metrics/history/fallback included. Conditions and per-environment rigid outcomes are identical.200ms assigned-device peaks include setup and trajectory without exact transient peak claim. Wider final-source four scopes/seeds, dynamic profile, bottleneck decisions and plateau remain pending. No physical convergence or RL training claim.")
