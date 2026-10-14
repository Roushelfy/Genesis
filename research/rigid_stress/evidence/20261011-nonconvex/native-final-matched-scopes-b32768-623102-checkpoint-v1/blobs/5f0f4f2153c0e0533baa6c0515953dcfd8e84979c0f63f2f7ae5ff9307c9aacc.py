"""Verify complete final-source matched scopes, outcomes, banks and device telemetry."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from publish_native_bounded_checkpoint_v1 import publish_bounded

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("batch", type=int)
parser.add_argument("seed", type=int)
parser.add_argument("--revision", required=True)
args = parser.parse_args()
root = Path(__file__).parent
parent = root / f"native-final-matched-scopes-v1/b{args.batch}-{args.seed}"
runs = [parent / scope for scope in ("rigid", "recovery", "live", "policy")]
bank_hash = hashlib.sha256((parent / "conditions.npz").read_bytes()).hexdigest()
results, scopes = {}, {}
rigid_keys = (
    "observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
    "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
    "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps",
)
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    assert hashlib.sha256((run / "conditions.npz").read_bytes()).hexdigest() == bank_hash
    assert (run / "conditions-before.sha256").read_bytes() == (run / "conditions-after.sha256").read_bytes()
    value = json.loads((run / "result.json").read_text())
    assert value["envs"] == args.batch and value["seed"] == args.seed and value["scope"] == run.name
    assert value["conditions_sha256"] == bank_hash
    assert value["dt"] == .01 and value["substeps"] == 1 and value["precision"] == "64"
    assert value["intrinsic_load_partition"] == "body_source_prior" and len(value["repeats"]) == 3
    assert value["source_revision"].startswith((run / "base-revision.txt").read_text().strip())
    for path in (run / "source/genesis/engine/solvers/rigid/stress").glob("*.py"):
        tracked = f"genesis/engine/solvers/rigid/stress/{path.name}"
        assert path.read_bytes() == subprocess.check_output(["git", "show", f"{args.revision}:{tracked}"])
    for repeat in value["repeats"]:
        assert repeat["steps"] == 1280 and repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        if run.name == "rigid":
            assert repeat["failed_steps_per_environment"] is None
        else:
            assert len(repeat["failed_steps_per_environment"]) == args.batch and max(repeat["failed_steps_per_environment"]) == 0
        assert repeat["resets"] == 0 if run.name == "recovery" else repeat["resets"] > 0
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in value["warmup_quality"] if "residual_N" in row]
    assert not budgets or max(budgets) <= 1
    telemetry = list(csv.DictReader((run / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True))
    assert len(telemetry) > 100 and all(row["uuid"] == value["gpu_uuid"] for row in telemetry)
    rate = args.batch * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
    scope = {
        "effective_env_steps_per_second": rate, "batch_steps_per_second": rate / args.batch,
        "repeat_seconds": [row["seconds"] for row in value["repeats"]],
        "repeat_env_steps_per_second": [args.batch * 1280 / row["seconds"] for row in value["repeats"]],
        "resets": [row["resets"] for row in value["repeats"]],
        "native_buffer_bytes": value.get("native_buffer_bytes", 0), "device_free_bytes": value["device_free_bytes"],
        "sampled_assigned_device_peak_MiB_including_setup": max(int(row["memory.used [MiB]"].split()[0]) for row in telemetry),
        "gpu_uuid": value["gpu_uuid"], "setup_seconds": value["setup_seconds"],
        "warmup_full_residual_budget_fraction_max": max(budgets, default=0),
        "policy": value["policy"], "source_revision": value["source_revision"],
    }
    if run.name != "recovery":
        metrics = value["trajectory_metrics"]
        assert len(metrics["repeats"]) == 3
        for row in metrics["repeats"]:
            assert row["totals"]["observed_steps"] == args.batch * 1280
            assert row["totals"]["invalid_stress_steps"] == row["totals"]["completed_lift_failure_cycles"] == row["totals"]["lift_grasp_fail_steps"] == 0
            assert all(len(row["per_environment"][key]) == args.batch for key in rigid_keys)
        scope["trajectory_totals"] = [row["totals"] for row in metrics["repeats"]]
    results[run.name], scopes[run.name] = value, scope
assert len({row["gpu_uuid"] for row in results.values()}) == 1
position_error = float(np.max(abs(np.array(results["rigid"]["egg_position_m"]) - np.array(results["live"]["egg_position_m"]))))
assert position_error <= 1e-12
assert scopes["rigid"]["resets"] == scopes["live"]["resets"]
for rigid, live in zip(results["rigid"]["trajectory_metrics"]["repeats"], results["live"]["trajectory_metrics"]["repeats"]):
    for key in rigid_keys:
        assert rigid["per_environment"][key] == live["per_environment"][key]
snapshot = results["recovery"]["recovery_snapshot_check"]
assert snapshot["nodal_force_max_difference_N"] <= 1e-11 and snapshot["peak_max_difference_Pa"] <= 1e-3
assert max(snapshot[key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
summary = {"batch": args.batch, "seed": args.seed, "numerical_source_revision": args.revision, "conditions_sha256": bank_hash,
           "rigid_live_position_error_m": position_error, "identical_per_environment_rigid_grasp_contact_descent_counters": True,
           "scopes": scopes, "recovery_snapshot_check": snapshot}
extras = [parent / "conditions.npz", root / f"native-final-matched-scopes-v1-b{args.batch}-{args.seed}-allocation.log"]
extras.extend(root / name for name in ("archive_native_final_matched_scopes_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_final_matched_scopes_v1.sh"))
publish_bounded(root, Path(f"research/rigid_stress/evidence/20261011-nonconvex/native-final-matched-scopes-b{args.batch}-{args.seed}-checkpoint-v1"), runs, extras, summary,
                f"Production numerical source {args.revision}, byte-verified stress engine. Same saved condition bank/four sequential scopes on one assigned RTX PRO6000 Blackwell GPU, low full hollow shell FP64/Q10/s8/max output, original final budgets. Each scope600 varied warmup and three1280-step repeats. Rigid/live/policy time observations, contact/radius updates, reset, metrics/history/fallback and all environment work. Recovery-only repeats a frozen actual snapshot with matching pre-integration frame; it is not changing-contact rollout throughput. Policy is a FP32 network inference rollout with observation/action interoperability and rigid/stress included, not full RL training. All per-environment records and difficult cases remain retained; all invalid/overflow/failed lifts zero.200ms assigned-device memory samples include setup/trajectory, not exact transient peaks. One seed per width; plateau and candidate qualification are assessed separately. No physical convergence or goal-completion claim.")
