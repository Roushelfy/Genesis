"""Check complete numerical outputs before interpreting native workspace timings."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .audit_native_production_controls import MOTION


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = args.data_root / "native-module-replay-v2/group0/scr-full"
    folders = [args.data_root / f"native-scr-workspace-tuning-v1/group0/scr-block{b}" for b in (1, 4)]
    receipts, rows = [], []
    with np.load(baseline / "trajectory.npz") as saved:
        reference = {name: saved[name] for name in saved.files}
    receipts.extend([file_id(baseline / "trajectory.npz"), file_id(baseline / "result.json")])
    for folder in folders:
        result = json.loads((folder / "result.json").read_text())
        assert result["passed"] and result["replay"]["complete_history_check_passed"]
        assert result["steps"] == 1200 and result["envs"] == 8 and result["output_mode"] == "full"
        assert result["trajectory"] == file_id(folder / "trajectory.npz")
        receipts.extend([file_id(folder / "result.json"), result["trajectory"]])
        with np.load(folder / "trajectory.npz") as saved:
            actual = {name: saved[name] for name in saved.files}
        checks = {
            "all_steps_valid": bool(actual["valid"].all()),
            "errno_zero": bool((actual["errno"] == 0).all()),
            "all_contacts_retained": bool(np.array_equal(actual["contact_counts"], reference["contact_counts"])),
        }
        motion_errors = {}
        for name, start, end, target in MOTION:
            first, second = actual["motion"][..., start:end], reference["motion"][..., start:end]
            if name == "quaternion_up_to_sign":
                first = np.where((first * second).sum(axis=-1, keepdims=True) < 0, -first, first)
            error = float(np.max(np.abs(first - second)))
            motion_errors[name] = error
            checks["motion/" + name] = error <= target
        for name in ("peak_pa", "stress_pa", "mises_pa"):
            checks[name + "/finite"] = bool(np.isfinite(actual[name]).all())
            checks[name + "/matched_numerical_target"] = bool(
                np.allclose(actual[name], reference[name], rtol=1e-4, atol=1e-3)
            )
        tensor_delta = actual["stress_pa"] - reference["stress_pa"]
        relative = np.linalg.norm(tensor_delta) / np.linalg.norm(reference["stress_pa"])
        rates = [r["env_updates_per_second"] for r in result["replay"]["repeats"]]
        rows.append(
            {
                "variant": folder.name,
                "numerical_consistency_passed": all(checks.values()),
                "checks": checks,
                "motion_errors": motion_errors,
                "complete_snapshot_tensor_array_relative_difference": float(relative),
                "maximum_tensor_absolute_difference_pa": float(np.abs(tensor_delta).max()),
                "initial_contact_block_size": result["stress_configuration"]["contact_block_size"],
                "actual_workspace_contacts": result["intrinsic_workspace_contacts"],
                "median_module_env_updates_per_s": statistics.median(rates),
                "min_module_env_updates_per_s": min(rates),
                "max_module_env_updates_per_s": max(rates),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "audit_completed": True,
            "all_variants_consistent": all(r["numerical_consistency_passed"] for r in rows),
            "rows": rows,
            "receipts": receipts,
            "baseline_timing_same_gpu": False,
            "physical_error_inherited": False,
            "scope": "Full1200-step motion/peak and11 complete tensor-snapshot numerical checks against block16. Smaller initial workspaces grew to the actual packed source maximum; all observed contacts/validity remain checked. Timings from a different allocation do not isolate a speedup. Tensor-array difference is a numerical Frobenius check, not a volume-weighted physical error.",
        },
    )


if __name__ == "__main__":
    main()
