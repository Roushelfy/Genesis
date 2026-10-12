"""Audit the first complete copied B1024 native source-prior inference rollout."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
run = root / "native-source-priors-policy-v1"
assert (run / "returncode.txt").read_text().strip() == "0"
assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
result = json.loads((run / "result.json").read_text())
assert result["envs"] == 1024 and result["scope"] == "policy"
assert result["intrinsic_load_partition"] == "body_source_prior"
assert result["load_model"].endswith("_body_source_prior")
assert result["policy"]["scope"] == "inference rollout"
assert result["dt"] == 0.01 and result["substeps"] == 1 and result["precision"] == "64"
assert result["conditions_sha256"] == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
assert len(result["repeats"]) == 3
for row in result["repeats"]:
    assert row["steps"] == 1280 and row["resets"] == 1184
    assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
    assert len(row["failed_steps_per_environment"]) == 1024 and max(row["failed_steps_per_environment"]) == 0
budgets = [np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))
           for row in result["warmup_quality"]]
assert max(budgets) <= 1
rate = 1024 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
summary = {"valid_env_steps_per_second": rate, "batch_steps_per_second": rate / 1024,
           "repeat_rates": [row["transitions_per_second"] for row in result["repeats"]],
           "resets_per_repetition": 1184, "invalid_environment_steps": 0,
           "gpu_uuid": result["gpu_uuid"], "native_buffer_bytes": result["native_buffer_bytes"],
           "device_free_bytes": result["device_free_bytes"],
           "torch_peak_allocated_bytes": result["torch_peak_allocated_bytes"],
           "actual_friction_ranges": [row["contact_friction_range"] for row in result["warmup_quality"]],
           "complete_residual_budget_fraction_max": float(max(budgets)), "policy": result["policy"],
           "result_sha256": hashlib.sha256((run / "result.json").read_bytes()).hexdigest(),
           "limitations": "One32-condition bank copied across all environments; source-prior law is an explicit model change. No matched old-model policy rate exists because it failed. Frontier not yet combined here. This proves a complete valid inference rollout on this bank; it does not establish wider seeds, grasp outcomes, maximum throughput, or a plateau. Per-raw contact error metadata denotes the aggregate constraint in this option and should not be read as independently conserved member wrenches."}
(run / "validated-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
