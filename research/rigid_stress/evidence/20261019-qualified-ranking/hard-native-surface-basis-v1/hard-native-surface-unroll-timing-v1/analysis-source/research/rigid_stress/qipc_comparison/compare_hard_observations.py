"""Compare complete timing audits using the existing public observation checks."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .time_hard_franka import compare


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--batch", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False)})
    rows = []
    for mode in ["rigid", "max", "full"]:
        paths = [root / f"{mode}-b{args.batch}-audit.npz" for root in [args.baseline, args.candidate]]
        with np.load(paths[0]) as original, np.load(paths[1]) as changed:
            assert set(original.files) == set(changed.files)
            compare(original, changed)
            rows.append(
                {
                    "mode": mode,
                    "baseline": file_id(paths[0]),
                    "candidate": file_id(paths[1]),
                    "frames": len(original["position_m"]),
                    "batch": args.batch,
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
