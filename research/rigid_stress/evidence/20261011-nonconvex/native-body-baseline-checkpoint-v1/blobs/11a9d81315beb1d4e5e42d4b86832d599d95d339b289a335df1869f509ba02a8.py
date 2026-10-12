"""Offline audit of the closed native body-prior/frontier copied B1024 scopes."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
summaries = {}
core_hashes = None
for scope in ("rigid", "recovery", "profile", "live", "policy"):
    run = root / f"native-body-baseline-v1/b1024-copied-{scope}-frontier-623101"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    current_hashes = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                      for line in (run / "source.sha256").read_text().splitlines()
                      if line.split(maxsplit=1)[1].startswith(("genesis/", "examples/"))}
    if core_hashes is None:
        core_hashes = current_hashes
    assert current_hashes == core_hashes
    result = json.loads((run / "result.json").read_text())
    assert result["envs"] == 1024 and result["scope"] == scope
    assert result["dt"] == 0.01 and result["substeps"] == 1 and result["precision"] == "64"
    assert result["intrinsic_load_partition"] == "body_source_prior"
    assert result["surface_fan_frontier"]
    assert result["conditions_sha256"] == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
    rows = result.get("repeats", [])
    for row in rows:
        assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
        if scope == "rigid":
            assert row["failed_steps_per_environment"] is None
        else:
            assert len(row["failed_steps_per_environment"]) == 1024
            assert max(row["failed_steps_per_environment"]) == 0
    budgets = []
    for quality in result["warmup_quality"]:
        if scope != "rigid":
            residual, rhs = np.array(quality["residual_N"]), np.array(quality["rhs_N"])
            assert np.all(np.isfinite(residual)) and np.all(np.isfinite(rhs))
            budgets.append(float(np.max(residual / np.maximum(1e-11, 1e-8 * rhs))))
    assert max(budgets, default=0) <= 1
    if rows:
        rate = 1024 * sum(row["steps"] for row in rows) / sum(row["seconds"] for row in rows)
    else:
        rate = None
    if scope != "profile":
        assert len(rows) == 3
    if scope in ("live", "policy"):
        assert all(row["steps"] == 1280 and row["resets"] == 1184 for row in rows)
    if scope == "policy":
        assert result["policy"]["scope"] == "inference rollout"
    summary = {
        "scope": scope, "valid_env_steps_per_second": rate,
        "batch_steps_per_second": None if rate is None else rate / 1024,
        "repetitions": [{key: row[key] for key in ("steps", "seconds", "transitions_per_second", "resets",
                                                   "invalid_environment_steps", "scatter_overflow_calls")} for row in rows],
        "complete_residual_budget_fraction_max": max(budgets, default=None),
        "actual_friction_ranges": [quality["contact_friction_range"] for quality in result["warmup_quality"]] if scope != "rigid" else None,
        "gpu_uuid": result["gpu_uuid"], "native_buffer_bytes": result["native_buffer_bytes"] if scope != "rigid" else 0,
        "device_free_bytes": result["device_free_bytes"], "policy": result.get("policy"),
        "stage_wall_ms": result.get("stage_wall_ms"),
        "recovery_snapshot_check": result.get("recovery_snapshot_check"),
        "source_revision_label": result["source_revision"],
        "result_sha256": hashlib.sha256((run / "result.json").read_bytes()).hexdigest(),
    }
    summaries[scope] = summary
    (run / "validated-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
out = {"scopes": summaries, "source_sha256": core_hashes,
       "limitations": "Copied32-condition bank. All fields/reset/fallback/observations included according to scope. Rigid-only minimum time extends actual steps and changes reset totals,so its rate is not a trajectory-matched rigid/live ratio. Recovery is a frozen snapshot. Wider B2048/B4096 independent conditions aborted and have no valid throughput. No diverse-seed acceptance,grasp-success criterion,full RL training throughput or optimization plateau is claimed."}
(root / "native-body-baseline-v1/validated-summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({scope: {key: summary[key] for key in ("valid_env_steps_per_second", "batch_steps_per_second",
                                                     "complete_residual_budget_fraction_max", "native_buffer_bytes")}
                  for scope, summary in summaries.items()}, indent=2), flush=True)
