"""Preserve the isolated Gaussian cost decision, including the first signature failure."""

import json
from pathlib import Path

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / f"native-gaussian-precision-cost-v{version}" for version in (1, 2)]
for run, expected in zip(runs, ("1", "0")):
    assert (run / "returncode.txt").read_text().strip() == expected
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
value = json.loads((runs[1] / "result.json").read_text())
assert len(value["measurements"]) == 2
assert all(len(row["routes"]) == 3 for row in value["measurements"])
for row in value["measurements"]:
    assert [route["gaussian_fp32"] for route in row["routes"]] == [False, True, False]
    assert all(route["independent_profile_relative_error_max"] <= 1e-6 for route in row["routes"])
    assert row["captured_fenced_mean_step_fraction"] < .002
extras = [root / "archive_native_gaussian_cost_v1.py", root / "run_archive_native_gaussian_cost_v1.sh",
          root / "publish_native_checkpoint_v1.py"]
extras.extend(root.glob("native-gaussian-precision-cost-v*-allocation.log"))
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-gaussian-cost-checkpoint-v1"), runs, extras,
        {**value, "decision": "No production Gaussian FP32 change selected: isolated anticipated whole-step saving is below0.2%; geometry/distance and nonlinear fitting dominate. Final stress consistency for Gaussian FP32 remains unmeasured."},
        "Standalone native Quadrants Gaussian kernels only, RTX PRO6000 Blackwell node hardware CSV; assigned GPU UUID not retained separately. Mean/max sample counts come from production92ec3e0c B16384/seed623102 dynamic capture, whose lineage remains in native-selected-dynamic-checkpoint-v1. ContiguousFP64 inputs/output; FP32 only exp argument/result; five100-call repeats in off/on/off order, independent CPU FP64 profile comparison. The first run fails because ndarray cannot useqd.template; corrected v2 usesqd.types.ndarray. Both source closures and failure are preserved. This microkernel is cost evidence, not a formal whole-kernel speed bound, contact/stress acceptance, a rollout comparison or headline throughput. No physical/numerical production law is changed.")
