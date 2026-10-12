"""Verify a same-GPU changing-contact curve with every environment copied from one bank."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from publish_native_bounded_checkpoint_v1 import publish_bounded

root = Path(__file__).parent
runs = [root / f"native-selected-copied-live-v1/b{batch}-623102" for batch in (8192, 16384, 24576, 32768)]
bank = root / "native-selected-matched-scopes-v1/b2048-623102/conditions.npz"
bank_hash = hashlib.sha256(bank.read_bytes()).hexdigest()
keys = ("observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
        "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps", "weak_grip_descents",
        "release_steps", "release_descents", "two_finger_contact_steps")
base_positions, base_counters, source_hashes, gpu = None, None, None, None
summary = []
for run, batch in zip(runs, (8192, 16384, 24576, 32768)):
    assert (run / "returncode.txt").read_text().strip() == "0"
    closure = (run / "source.sha256").read_bytes()
    assert closure == (run / "source-after.sha256").read_bytes()
    assert (run / "conditions-before.sha256").read_bytes() == (run / "conditions-after.sha256").read_bytes()
    assert hashlib.sha256((run / "conditions.npz").read_bytes()).hexdigest() == bank_hash
    value = json.loads((run / "result.json").read_text())
    assert value["envs"] == batch and value["condition_count"] == 2048 and value["scope"] == "live"
    assert value["seed"] == 623102 and value["conditions_sha256"] == bank_hash
    assert value["source_revision"].startswith("92ec3e0c658f0436f67c81661e36305977487197")
    assert value["precision"] == "64" and value["dt"] == .01 and value["substeps"] == 1
    assert value["intrinsic_load_partition"] == "body_source_prior" and len(value["repeats"]) == 3
    positions = np.array(value["egg_position_m"])
    if base_positions is None:
        base_positions = positions[:2048].copy()
        base_counters = [{key: np.array(row["per_environment"][key][:2048]) for key in keys}
                         for row in value["trajectory_metrics"]["repeats"]]
        source_hashes, gpu = closure, value["gpu_uuid"]
    assert closure == source_hashes and value["gpu_uuid"] == gpu
    position_difference = float(np.max(abs(positions - np.tile(base_positions, (batch // 2048, 1)))))
    assert position_difference <= 1e-12
    for repeat, metrics, baseline in zip(value["repeats"], value["trajectory_metrics"]["repeats"], base_counters):
        assert repeat["steps"] == 1280 and repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        assert len(repeat["failed_steps_per_environment"]) == batch and max(repeat["failed_steps_per_environment"]) == 0
        assert metrics["totals"]["observed_steps"] == batch * 1280
        assert metrics["totals"]["invalid_stress_steps"] == metrics["totals"]["completed_lift_failure_cycles"] == 0
        assert metrics["totals"]["lift_grasp_fail_steps"] == 0
        for key in keys:
            assert np.array_equal(np.array(metrics["per_environment"][key]), np.tile(baseline[key], batch // 2048))
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
               for row in value["warmup_quality"]]
    assert max(budgets) <= 1
    telemetry = list(csv.DictReader((run / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True))
    assert len(telemetry) > 100 and all(row["uuid"] == gpu for row in telemetry)
    rate = batch * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
    summary.append({"batch": batch, "env_steps_per_second": rate, "batch_steps_per_second": rate / batch,
                    "repeat_seconds": [row["seconds"] for row in value["repeats"]],
                    "repeat_env_steps_per_second": [batch * 1280 / row["seconds"] for row in value["repeats"]],
                    "native_buffer_bytes": value["native_buffer_bytes"], "gpu_uuid": gpu,
                    "sampled_device_peak_MiB_including_setup": max(int(row["memory.used [MiB]"].split()[0]) for row in telemetry),
                    "warmup_full_residual_budget_fraction_max": max(budgets),
                    "copied_rigid_position_difference_m": position_difference,
                    "resets": [row["resets"] for row in value["repeats"]],
                    "trajectory_totals": [row["totals"] for row in value["trajectory_metrics"]["repeats"]]})
    del value
extras = [bank, root / "native-selected-copied-live-v1-allocation.log",
          root / "run_native_selected_copied_live_v1.sh", root / "archive_native_selected_copied_curve_v1.py",
          root / "run_archive_native_selected_copied_curve_v1.sh", root / "publish_native_checkpoint_v1.py"]
publish_bounded(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-selected-copied-curve-checkpoint-v1"),
                runs, extras, {"source_revision": "92ec3e0c658f0436f67c81661e36305977487197", "conditions_sha256": bank_hash,
                               "condition_count": 2048, "same_gpu_and_complete_rigid_counters": True, "widths": summary},
                "Same assigned RTX PRO6000 Blackwell GPU, production92ec3e0c, one saved2048-condition bank tiled exactly at8192/16384/24576/32768. FP64/Q10/s8/body_source_prior/max,dt.01/sub1,600 varied warmup and three1280-step changing live repeats. Includes observation/contact-radius updates/reset/metrics/history/fallback, every environment and outcome. Independent per-environment rigid/contact/grasp/descent counters are tiled identically across widths; fit iteration costs may differ. Zero invalid steps/overflows/failed lifts.200ms assigned GPU memory samples include setup, not exact transient peaks. No policy/full RL/physical convergence claim. This is a baseline curve; candidate optimizations still need final-source qualification.")
