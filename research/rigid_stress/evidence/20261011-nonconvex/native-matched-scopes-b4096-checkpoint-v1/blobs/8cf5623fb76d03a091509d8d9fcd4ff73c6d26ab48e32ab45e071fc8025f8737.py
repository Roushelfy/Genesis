"""Verify and publish the identical-condition four-scope B4096 checkpoint on allocated CPU."""
import hashlib
import json
from pathlib import Path

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
parent = root / "native-matched-scopes-v1/b4096-623103"
runs = [parent / scope for scope in ("rigid", "recovery", "live", "policy")]
bank_hash = hashlib.sha256((parent / "conditions.npz").read_bytes()).hexdigest()
results, summaries = {}, {}
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    assert hashlib.sha256((run / "conditions.npz").read_bytes()).hexdigest() == bank_hash
    value = json.loads((run / "result.json").read_text())
    assert value["scope"] == run.name and value["envs"] == 4096 and value["seed"] == 623103
    assert value["conditions_sha256"] == (None if run.name == "rigid" else bank_hash)
    assert value["dt"] == .01 and value["substeps"] == 1 and value["precision"] == "64"
    assert value["intrinsic_load_partition"] == "body_source_prior" and len(value["repeats"]) == 3
    for repeat in value["repeats"]:
        assert repeat["steps"] == 1280 and repeat["invalid_environment_steps"] == 0
        assert repeat["scatter_overflow_calls"] == 0
        assert repeat["failed_steps_per_environment"] is None if run.name == "rigid" else max(repeat["failed_steps_per_environment"]) == 0
        assert repeat["resets"] == (0 if run.name == "recovery" else 4643)
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in value["warmup_quality"] if "residual_N" in row]
    assert not budgets or max(budgets) <= 1
    rate = 4096 * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
    summary = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 4096,
               "repeats": [{key: item for key, item in row.items() if key != "failed_steps_per_environment"} for row in value["repeats"]],
               "native_buffer_bytes": value.get("native_buffer_bytes", 0), "device_free_bytes": value["device_free_bytes"],
               "gpu_uuid": value["gpu_uuid"], "setup_seconds": value["setup_seconds"],
               "build_timings_including_JIT": value.get("build_timings_including_JIT"),
               "warmup_full_residual_budget_fraction_max": max(budgets, default=0),
               "policy": value["policy"], "source_revision": value["source_revision"]}
    if run.name != "recovery":
        metrics = value["trajectory_metrics"]
        assert len(metrics["repeats"]) == 3
        for row in metrics["repeats"]:
            assert row["totals"]["observed_steps"] == 4096 * 1280
            assert row["totals"]["invalid_stress_steps"] == 0
            assert row["totals"]["completed_lift_failure_cycles"] == 0
            assert row["totals"]["lift_grasp_fail_steps"] == 0
        summary["trajectory_metrics"] = {"warmup": {key: item for key, item in metrics["warmup"].items() if key != "per_environment"},
                                         "repeats": [{key: item for key, item in row.items() if key != "per_environment"} for row in metrics["repeats"]]}
    results[run.name], summaries[run.name] = value, summary
assert len({row["gpu_uuid"] for row in results.values()}) == 1
position_error = float(np.max(np.abs(np.array(results["rigid"]["egg_position_m"]) - np.array(results["live"]["egg_position_m"]))))
assert position_error <= 1e-12
rigid_keys = ("observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
              "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
              "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps")
for rigid, live in zip(results["rigid"]["trajectory_metrics"]["repeats"], results["live"]["trajectory_metrics"]["repeats"]):
    for key in rigid_keys:
        assert rigid["per_environment"][key] == live["per_environment"][key]
snapshot = results["recovery"]["recovery_snapshot_check"]
assert snapshot["nodal_force_max_difference_N"] <= 1e-11 and snapshot["peak_max_difference_Pa"] <= 1e-3
assert max(snapshot[key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
summary = {"conditions_sha256": bank_hash, "rigid_live_position_error_m": position_error,
           "identical_rigid_grasp_contact_descent_counters": True, "scopes": summaries,
           "recovery_snapshot_check": snapshot}
extras = [parent / "conditions.npz", root / "native-matched-scopes-v1-b4096-623103-allocation.log"]
extras += [root / name for name in ("archive_native_matched_scopes_b4096_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_matched_scopes_b4096_v1.sh")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-matched-scopes-b4096-checkpoint-v1"), runs, extras, summary,
        "Current 6a861556 source, matched saved condition bank, low full hollow shell FP64/Q10/s8, original final budgets. Rigid/live/policy include timed metrics, contact/radius changes, observation, reset and maintenance. Recovery-only repeats a frozen actual snapshot and is explicitly not a changing-contact rollout. Policy is FP32 network inference rollout, not RL training. Four scopes run sequentially on one GPU. Single seed B4096 checkpoint; wider batches, more seeds, dynamic tail attribution and optimization plateau remain pending. No physical convergence claim.")
