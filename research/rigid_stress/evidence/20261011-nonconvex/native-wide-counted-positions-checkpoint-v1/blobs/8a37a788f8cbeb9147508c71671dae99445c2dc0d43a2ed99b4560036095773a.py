"""Verify and publish wide varied live probes with complete counted-position recovery."""

import csv
import json
from pathlib import Path

import numpy as np
from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs, summaries = [], []
for batch, seed in ((16384, 991909), (32768, 623102)):
    run = root / f"native-compact-cached-positions-v1/b{batch}-varied-live-on-{seed}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    result = json.loads((run / "result.json").read_text())
    assert result["envs"] == batch and result["seed"] == seed and result["scope"] == "live"
    assert result["compact_cached_fan_positions"] and result["precision"] == "64"
    assert result["dt"] == .01 and result["substeps"] == 1 and len(result["repeats"]) == 3
    for repeat, metrics in zip(result["repeats"], result["trajectory_metrics"]["repeats"]):
        assert repeat["steps"] == 1280
        assert repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        assert max(repeat["failed_steps_per_environment"]) == 0
        assert metrics["totals"]["observed_steps"] == batch * 1280
        assert metrics["totals"]["invalid_stress_steps"] == metrics["totals"]["lift_grasp_fail_steps"] == 0
        assert metrics["totals"]["completed_lift_failure_cycles"] == 0
    fractions = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"]]
    assert max(fractions) <= 1
    telemetry = list(csv.DictReader((run / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True))
    assert len(telemetry) > 100 and all(row["uuid"] == result["gpu_uuid"] for row in telemetry)
    rate = batch * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
    summaries.append({
        "batch": batch, "seed": seed, "gpu_uuid": result["gpu_uuid"],
        "effective_env_steps_per_second": rate, "batch_steps_per_second": rate / batch,
        "repeat_seconds": [row["seconds"] for row in result["repeats"]],
        "repeat_env_steps_per_second": [batch * 1280 / row["seconds"] for row in result["repeats"]],
        "native_buffer_bytes": result["native_buffer_bytes"], "device_free_bytes_at_report": result["device_free_bytes"],
        "sampled_assigned_device_peak_MiB_including_setup": max(int(row["memory.used [MiB]"].split()[0]) for row in telemetry),
        "warmup_full_residual_budget_fraction_max": max(fractions),
        "resets": [row["resets"] for row in result["repeats"]],
        "trajectory_totals": [row["totals"] for row in result["trajectory_metrics"]["repeats"]],
        "source_revision": result["source_revision"], "setup_seconds": result["setup_seconds"],
    })
    runs.append(run)
extras = [root / name for name in (
    "archive_native_wide_counted_positions_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_wide_counted_positions_v1.sh",
    "native-compact-cached-positions-live-v1-b16384-allocation.log", "native-compact-cached-positions-live-v1-b32768-allocation.log",
)]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-wide-counted-positions-checkpoint-v1"), runs, extras, {"live": summaries},
        "Low full hollow shell FP64/Q10/s8,max output,body_source_prior,mesh64/GJK patch-on,dt0.01/substep1. Private counted-position prototype combines qualified complete sampled storage/caching and production50b1b61c retry; it retains serial fan refinement and predates cleaned production parallel fans. Each batch has600 varied warmup plus three1280-step varied live repeats; observations, contact/radius updates, resets, metrics, history/fallback and all environments are included. All reported invalid/overflow/lift-failure counts are zero, full per-environment raw records remain archived, and native acceptance checks every step. Assigned-device200ms memory samples include setup/trajectory without an exact transient peak claim. B16384 uses separated seed991909; B32768 uses623102, and they run on different assigned GPUs. These are exploratory resident-batch rates, not a paired source gain, final-source scaling curve or measured optimum. Independent fixture qualification is the separately published31 CPU/31 CUDA/96-field cache checkpoint. Combined production source qualification, matched four scopes/seeds, later bottleneck decisions and plateau remain pending. No physical convergence or RL training claim.")
