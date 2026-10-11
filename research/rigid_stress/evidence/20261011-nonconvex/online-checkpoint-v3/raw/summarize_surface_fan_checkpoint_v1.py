"""Reduce immutable acceptance records without replacing their complete raw data."""

import json
from pathlib import Path

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
live = json.loads((root / "online-surface-fan-v3/results/summary.json").read_text())
records, captures = live["records"], live["oracle_captures"]
assert live["completed_steps"] == 640 and live["source_sha256"] == live["source_sha256_after"]
rigid = json.loads((root / "rigid-twin-v1/results/summary.json").read_text())
immutable = json.loads((root / "rigid-immutability-v2/summary.json").read_text())
assert rigid["completed_steps"] == 640
assert immutable["completed_steps"] == 640
assert all(row["before_sha256"] == row["after_sha256"] for row in immutable["records"])
result = {
    "live_completed_steps": live["completed_steps"],
    "environments": live["arguments"]["envs"],
    "seed": live["arguments"]["seed"],
    "normalized_force_error_max": max(row["max_force_error"] for row in records),
    "normalized_moment_error_max": max(row["max_moment_error"] for row in records),
    "complete_residual_max_N": max(row["full_residual_max_N"] for row in captures),
    "cpu_fp64_full_fields": len(captures) * live["arguments"]["envs"],
    "captures": len(captures),
    "fit_evaluations_max": max(row["max_evaluations"] for row in records),
    "positive_samples_max": max(row["max_samples"] for row in records),
    "contact_slot_changes": sum(row["contact_slot_changes"] for row in records),
    "actual_combined_mu": sorted({mu for row in records for mu in row["mu_values"]}),
    "reset_environments": records[-1]["reset_environments"],
    "rigid_twin_steps": rigid["completed_steps"],
    "rigid_twin_max_state_difference": max(max(row["maximum_state_error_per_env"]) for row in rigid["records"]),
    "recovery_rigid_byte_audit_calls": len(immutable["records"]),
    "recovery_rigid_byte_audit_fields": len(immutable["rigid_data_names"]),
    "recovery_rigid_byte_audit_reset_environments": immutable["reset_environments"],
    "scope": "Validation only; no throughput claim. The independent-scene rigid/stress trajectory comparison failed its 1e-12 bound at tick466 and is retained separately.",
}
(root / "surface-fan-checkpoint-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
