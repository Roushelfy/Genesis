"""Allocated-CPU unchanged-budget CUDA qualification and matched cold cooperative retry ablation."""
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / f"native-cooperative-retry-v1/b2048-varied-live-{mode}-623102" for mode in ("off", "on")]
results, summaries = {}, {}
rigid_keys = ("observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
              "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
              "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps")
for mode, run in zip(("off", "on"), runs):
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    value = json.loads((run / "result.json").read_text())
    assert value["cold_cooperative_retry"] == (mode == "on")
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
        assert row["totals"]["completed_lift_success_cycles"] == 3142 and row["totals"]["completed_lift_failure_cycles"] == 0
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
    summaries[mode] = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
                       "repeats": [{key: item for key, item in row.items() if key != "failed_steps_per_environment"} for row in value["repeats"]],
                       "trajectory_metrics": {"repeats": [{key: item for key, item in row.items() if key != "per_environment"} for row in metrics["repeats"]]},
                       "setup_seconds": value["setup_seconds"], "native_buffer_bytes": value["native_buffer_bytes"],
                       "gpu_uuid": value["gpu_uuid"], "warmup_full_residual_budget_fraction_max": max(budgets)}
    results[mode] = value
assert len({row["gpu_uuid"] for row in results.values()}) == 1
for mode, value in results.items():
    assert np.max(abs(np.array(value["egg_position_m"]) - np.array(results["off"]["egg_position_m"]))) <= 1e-12
    for baseline, candidate in zip(results["off"]["trajectory_metrics"]["repeats"], value["trajectory_metrics"]["repeats"]):
        for key in rigid_keys:
            assert baseline["per_environment"][key] == candidate["per_environment"][key]
reference_rate = 2048 * 1280 * 3 / sum(row["seconds"] for mode in ("off",) for row in results[mode]["repeats"])
regression = root / "native-cooperative-retry-regression-v1/cuda"
assert (regression / "returncode.txt").read_text().strip() == "0"
assert (regression / "source.sha256").read_bytes() == (regression / "source-after.sha256").read_bytes()
suite = ElementTree.parse(regression / "report.xml").getroot().find("testsuite")
assert suite.attrib["tests"] == "23" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
fields = []
for case in suite.findall("testcase"):
    if "wrench_cones_full_fields" in case.attrib["name"]:
        values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
        assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
        assert max(values["complete_residual_budget_fraction"]) <= 1
        assert max(values["peak_difference_Pa"]) <= 1e-3
        fields.append({"case": case.attrib["name"], **values})
assert len(fields) == 8 and sum(len(row["native_peak_Pa"]) for row in fields) == 24
runs.append(regression)
summary = {"live": summaries, "paired_original_rate": reference_rate,
           "live_speedup": summaries["on"]["effective_env_steps_per_second"] / reference_rate,
           "identical_rigid_positions_and_per_environment_grasp_contact_descent_counters": True,
           "cuda_passed": 23, "cuda_seconds": float(suite.attrib["time"]), "recorded_independent_full_fields": 24,
           "fields": fields, "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
extras = [root / name for name in ("archive_native_cooperative_retry_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_cooperative_retry_v1.sh", "run_cooperative_retry_live_pair_v1.sh", "native-cooperative-retry-live-v1-allocation.log")]
extras += [root / name for name in ("native-cooperative-retry-tests-v1-cuda-allocation.log", "baseline_cooperative_retry_fit_v1.py", "baseline_cooperative_retry_traction_v1.py")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-cooperative-retry-checkpoint-v1"), runs, extras, summary,
        "Private numerical base matches6a861556; immutable source closures and complete patch are captured. The candidate adds a cold128-evaluation warp retry only for unresolved numerical or sampled failures, keeps successful constraints unchanged, counts both fits, and retains original serial fallback and all final budgets. Source priors, Q10/s8 integration/radii/friction and physical objective are unchanged. CPU route is unchanged; CUDA23 including24 independent CPU FP64 full fields pass. Original/candidate run sequentially on one GPU, each600 warmup and three1280-step repeats with timed metrics/all environments/reset/history/fallback. Current-source combined qualification, dynamic tail attribution, more seeds, broader scopes and plateau remain pending. No physical convergence or RL training claim.")
