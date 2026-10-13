"""Estimate first-warm-accept applicability from immutable complete trajectory counters."""

import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
with np.load(root / "native-fit-interior-v1/b2048-623102/result.npz") as saved:
    raw = saved["counters"]
loaded = raw[:, :, 0] == 1
first = loaded & (raw[:, :, 4] == raw[:, :, 2])
totals = raw.sum(axis=(0, 1))
fraction = raw[:, :, 2][first].sum() / totals[2]
result = {
    "first_accept_loaded_constraint_fraction": float(first.sum() / loaded.sum()),
    "first_accept_positive_sample_fraction": float(fraction),
    "first_accept_sample_evaluation_fraction": float(raw[:, :, 4][first].sum() / totals[4]),
    "later_accept_positive_sample_fraction": float(1 - fraction),
    "interior_extra_evaluations_work_fraction": float(
        (totals[5] - raw[:, :, 2][raw[:, :, 1] == 1].sum()) / totals[4]
    ),
    "limitations": "Work proxies, not wall-time bounds. A late-verification route must update current normalization, verify current wrench/cones and replay rejected groups; direct Gram routes must also form/solve the current geometry-dependent operator. Does not mutate production or establish a selected optimization.",
}
(root / "native-first-warm-accept-cost-v1.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
