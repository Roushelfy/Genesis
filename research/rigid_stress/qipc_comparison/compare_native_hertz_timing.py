"""Compare admitted moving-input recovery audits from a paired native timing run."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--paired", type=Path, required=True)
    cli.add_argument("--scene", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, candidate = ((args.paired / "reference", args.paired / "current") if args.scene
                           else (args.paired / "baseline", args.paired / "parallel"))
    original = json.loads((baseline / "result.json").read_text())
    changed = json.loads((candidate / "result.json").read_text())
    assert original["passed"] and changed["passed"]
    rows, bindings = [], [file_id(baseline / "result.json"), file_id(candidate / "result.json")]
    for left, right in zip(original["rows"], changed["rows"], strict=True):
        batch, mode = left["batch"], left["mode"]
        assert (batch, mode) == (right["batch"], right["mode"])
        first, second = baseline / f"{mode}-b{batch}-audit.npz", candidate / f"{mode}-b{batch}-audit.npz"
        bindings.extend([file_id(first), file_id(second)])
        errors = {}
        with np.load(first) as before, np.load(second) as after:
            exact_keys = (["input_status", "field_status", "loaded"] if args.scene
                          else ["status", "loaded", "elastic_valid", "centers_m", "axes_m"])
            if mode == "rigid":
                exact_keys = []
            for key in exact_keys:
                np.testing.assert_array_equal(before[key], after[key])
            if args.scene:
                np.testing.assert_allclose(after["positions_m"], before["positions_m"], rtol=0, atol=1e-12)
                if mode != "rigid":
                    assert np.all(before["field_status"] <= 1) and np.all(before["input_status"] == 0)
            else:
                assert np.all(before["status"] <= 1) and before["elastic_valid"].all()
                assert max(before["wrench_error"].max(), after["wrench_error"].max()) < 1e-4
            stress_keys = ["vm_pa", "principal_pa" if args.scene else "tensile_pa"] if mode != "rigid" else []
            for key in stress_keys:
                assert np.isfinite(before[key]).all() and np.isfinite(after[key]).all()
                error = np.linalg.norm(before[key] - after[key]) / max(np.linalg.norm(before[key]), 1e-30)
                assert error < 1e-7, (batch, mode, key, error)
                np.testing.assert_allclose(after[key], before[key], rtol=1e-7, atol=1e-6)
                errors[key] = float(error)
        rows.append({"batch": batch, "mode": mode, "relative_peak_differences": errors,
                     "baseline_seconds": left["median_seconds"], "parallel_seconds": right["median_seconds"],
                     "speed_ratio": left["median_seconds"] / right["median_seconds"],
                     "admitted_audit_frames": left["admitted_audit_environment_steps" if args.scene
                                                   else "admitted_all_audit_frames"],
                     "loaded_audit_frames": left["admitted_loaded_audit_steps" if args.scene
                                                 else "admitted_loaded_audit_frames"]})
    assert len(rows) == (12 if args.scene else 8)
    write_json(args.output / "result.json", {
        "passed": True, "identity": runtime_identity(require_gpu_inventory=False), "rows": rows, "bindings": bindings,
        "scope": "Same-allocation baseline/parallel moving normal-module or declared sphere/flat friction-scene audits. Identical inputs/admission and every-frame sampled peaks; scene trajectories also match. Full-field physical controls are separate. Neither workload qualifies curved pressure or Franka throughput.",
        "complete_rigid_scene": args.scene,
        "whole_goal_complete": False,
    })


if __name__ == "__main__":
    main()
