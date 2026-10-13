"""Publish validated synthetic counter costs and every failed probe revision."""
import json
from pathlib import Path

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / f"native-sampler-counter-cost-v{version}" for version in (1, 2, 3, 4)]
for version, run in enumerate(runs, 1):
    assert (run / "returncode.txt").read_text().strip() == ("0" if version == 4 else "1")
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
result = json.loads((runs[-1] / "result.json").read_text())
assert len(result["rows"]) == 6
rows = []
for row in result["rows"]:
    timings = [np.mean(route["seconds"]) * 1000 for route in row["timings"]]
    baseline = (timings[0] + timings[2]) / 2
    rows.append({"nominal_candidates": row["nominal_candidates"], "columns": row["columns"],
        "actual_candidates": row["actual_candidates"], "actual_accepted": row["actual_accepted"],
        "off_on_off_ms": timings, "saving_ms": float(baseline - timings[1])})
summary = {"rows": rows, "primitive_scope": result["scope"], "limitations": result["limitations"],
    "failed_probes": "v1 shuffle offset/index needs explicit u32 arguments. v2/v3 initial configurations pass, but recycled configuration buffers accumulate35passes in the fourth configuration instead of being cleared. v3 records32235=35*921 candidates. v4 retains all independent configuration buffers for the whole measurement and all six count/max/index-permutation validations pass. Failed logs/sources retained. This does not establish a general engine allocation bug.",
    "decision": "Savings2.48-5.39ms on the synthetic aggregate-count scales motivate a private complete geometry sampler implementation and real rollout pair. It is not selected from this primitive cost. Captured raw diagnostics can duplicate body-union work, so these figures are not a formal whole-sampler bound or production saving."}
extras = [root / "archive_native_sampler_counter_cost_v1.py", root / "run_archive_native_sampler_counter_cost_v1.sh"]
extras += list(root.glob("native-sampler-counter-cost-v*-allocation.log"))
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-sampler-counter-cost-checkpoint-v1"),
    runs, extras, summary, "Quadrants/CUDA synthetic counters/reservation/FP64 maximum/index writes, assigned RTX PRO6000 Blackwell hardware recorded, irregular20percent rejected samples, off/on/off with three ten-call repetitions. All columns and every accepted index verified. Geometry/profile/stress/rollout are outside this cost scope. No physical convergence, actual rollout gain, full RL training or goal completion claim.")
