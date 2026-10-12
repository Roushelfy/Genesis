"""Allocated-CPU matched fastcache-only live check with original runs before and after."""
import json
from pathlib import Path

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / f"native-contact-fast-live-v1/b2048-varied-live-{mode}-623102" for mode in ("off", "fast", "off-after")]
results, summaries = {}, {}
rigid_keys = ("observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
              "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
              "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps")
for mode, run in zip(("off", "fast", "off-after"), runs):
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    value = json.loads((run / "result.json").read_text())
    assert value["native_contact_dispatch"] == ("fast" if mode == "fast" else "off")
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
reference_rate = 2048 * 1280 * 6 / sum(row["seconds"] for mode in ("off", "off-after") for row in results[mode]["repeats"])
summary = {"live": summaries, "paired_original_rate": reference_rate,
           "live_speedup": summaries["fast"]["effective_env_steps_per_second"] / reference_rate,
           "identical_rigid_positions_and_per_environment_grasp_contact_descent_counters": True}
extras = [root / name for name in ("archive_native_contact_fast_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_contact_fast_v1.sh", "run_native_contact_fast_live_all_v1.sh", "native-contact-fast-live-all-v1-allocation.log")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-contact-fastcache-checkpoint-v1"), runs, extras, summary,
        "Same original Quadrants numerical function bodies are rewrapped with fastcache=True without graph capture. Current source6a861556, low full hollow shell FP64/Q10/s8 and original final budgets. Three modes sequential on one GPU:original/fastcache/original, each600 warmup and three1280-step repeats with timed metrics/all environments/reset/history/fallback. Fastcache changes dispatch only. Profile comparison and complete-graph oracle evidence are published separately; this checkpoint checks live complete residuals and environment equality and is not an additional independent contact oracle. Single B2048 seed; broad scope/seed measurements, dynamic-tail optimization and plateau remain pending. No physical convergence or RL training claim.")
