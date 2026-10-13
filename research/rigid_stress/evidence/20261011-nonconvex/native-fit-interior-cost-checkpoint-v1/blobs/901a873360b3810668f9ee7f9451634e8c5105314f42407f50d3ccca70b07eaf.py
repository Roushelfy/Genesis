"""Audit complete applicability counters and preserve the no-mutation production source."""

import json
from pathlib import Path

import numpy as np
from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
run = root / "native-fit-interior-v1/b2048-623102"
assert (run / "returncode.txt").read_text().strip() == "0"
assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
result = json.loads((run / "result.json").read_text())
assert result["envs"] == 2048 and result["warmup"] == 600 and result["steps"] == 1280
with np.load(run / "result.npz") as saved:
    raw = saved["counters"]
assert raw.shape == (1280, 2048, 6) and raw.min() >= 0
assert np.array_equal(raw.sum(axis=(0, 1)), result["totals"])
assert raw[:, :, 0].max() == 1
interior = raw[:, :, 1] == 1
interior_samples = raw[:, :, 2][interior].sum()
interior_work = raw[:, :, 5].sum()
first = interior & (raw[:, :, 5] == raw[:, :, 2])
metrics = result["trajectory_metrics"]
assert metrics["totals"]["observed_steps"] == 2048 * 1280
assert metrics["totals"]["invalid_stress_steps"] == metrics["totals"]["lift_grasp_fail_steps"] == 0
assert metrics["totals"]["completed_lift_failure_cycles"] == 0
summary = {key: value for key, value in result.items() if key != "trajectory_metrics"}
summary["trajectory_metrics"] = {key: value for key, value in metrics.items() if key != "per_environment"}
summary["interior_positive_samples"] = int(interior_samples)
summary["interior_sample_weighted_mean_evaluations"] = float(interior_work / interior_samples)
summary["interior_first_evaluation_samples_fraction"] = float(raw[:, :, 2][first].sum() / interior_samples)
summary["interior_first_evaluation_constraints_fraction"] = float(first.sum() / interior.sum())
summary["decision_scope"] = "Applicability/cost evidence for an unconditional unconstrained Gram prepass. A Gram solve plus current cone/wrench verification needs complete source scans, while already-accepted warm cases need no search direction. These counters are work proxies; they do not establish a latency bound or reject all possible future selective shortcuts."
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-fit-interior-cost-checkpoint-v1"),
        [run], [root / name for name in ("archive_native_fit_interior_v1.py", "run_archive_native_fit_interior_v1.sh",
        "native-fit-interior-v1-allocation.log", "publish_native_checkpoint_v1.py")], summary,
        "Production source76537c2f, unchanged numerical code. Complete2048/623102 saved-bank trajectory with600warmup/1280steps, every environment and outcome included. Read-only Quadrants diagnostic classifies every positive sample's final dual proposal, with128eps scene-FP64 relative roundoff for the diagnostic only. No load clipping, contact rejection, altered stopping or acceptance. Keeps full per-environment/frame counters. Unconditional Gram prepass applicability/work evidence, not an implemented direct solver or performance bound; selective future schemes are not ruled out. No headline throughput/fullRL/physical convergence claim.")
