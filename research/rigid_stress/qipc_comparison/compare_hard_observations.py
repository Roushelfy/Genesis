"""Compare complete timing audits using the existing public observation checks."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .time_hard_franka import compare


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--candidate-mode-subdir", help="Optional per-mode directory pattern, e.g. memory-{mode}/driver")
    cli.add_argument("--check-repeated-counts", action="store_true")
    cli.add_argument("--batch", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False)})
    rows = []
    for mode in ["rigid", "max", "full"]:
        candidate = args.candidate
        if args.candidate_mode_subdir:
            candidate = candidate / args.candidate_mode_subdir.format(mode=mode)
        paths = [root / f"{mode}-b{args.batch}-audit.npz" for root in [args.baseline, candidate]]
        with np.load(paths[0]) as original, np.load(paths[1]) as changed:
            assert set(original.files) == set(changed.files)
            compare(original, changed)
            counts = []
            if args.check_repeated_counts:
                for root, data in [(args.baseline, original), (candidate, changed)]:
                    result = json.loads((root / "result.json").read_text())
                    start = json.loads((root / "start.json").read_text())
                    assert result["passed"]
                    row = next(row for row in result["rows"] if row["batch"] == args.batch and row["mode"] == mode)
                    repeats = len(row["seconds_per_scene_step"])
                    assert repeats == start["repeats"]
                    assert len(data["position_m"]) == start["start_step"] + start["steps"]
                    assert row["audit_environment_steps"] == len(data["position_m"]) * args.batch
                    if mode != "rigid":
                        assert row["admitted_timed_environment_steps"] == start["steps"] * args.batch * repeats
                        for field, prefix in [("loaded", "loaded"), ("sharp_edge", "sharp")]:
                            window = int(data[field][start["start_step"] :].sum())
                            assert row[prefix + "_window_environment_steps"] == window
                            assert row[prefix + "_timed_environment_steps"] == repeats * window
                    counts.append({"result": file_id(root / "result.json"), "repeats": repeats})
            rows.append(
                {
                    "mode": mode,
                    "baseline": file_id(paths[0]),
                    "candidate": file_id(paths[1]),
                    "frames": len(original["position_m"]),
                    "batch": args.batch,
                    "repeated_count_audits": counts,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "scope": "Existing paired-timer public checks across engine sources: position absolute 1e-12m, stress relative 1e-7/absolute 1e-6Pa, exact validity/load/edge/status masks and infinite maxima. Separate raw full-field diagnostics retain their original criteria and outcomes.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
