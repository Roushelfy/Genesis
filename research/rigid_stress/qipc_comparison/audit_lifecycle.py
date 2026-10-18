"""Compare fresh-process and batch checkpoints in addition to recorded lifecycle continuations."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def compare(first, second):
    data = {}
    with np.load(first / "states.npz") as left, np.load(second / "states.npz") as right:
        for name in left.files:
            if not name.startswith("checkpoint_"):
                continue
            a, b = left[name][:1], right[name][:1]
            assert a.shape == b.shape
            finite = np.isfinite(a) & np.isfinite(b)
            delta = a[finite].astype(float) - b[finite].astype(float)
            norm = np.linalg.norm(a[finite].astype(float))
            data[name] = {
                "maximum_absolute_difference": np.abs(delta).max().item() if delta.size else None,
                "relative_l2_difference": np.linalg.norm(delta).item() / norm if norm else None,
                "first_nonfinite_values": int((~np.isfinite(a)).sum()),
                "second_nonfinite_values": int((~np.isfinite(b)).sum()),
            }
    return {"first": file_id(first / "states.npz"), "second": file_id(second / "states.npz"), "differences": data}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    histories, by_model = [], {}
    for group, tasks in enumerate(json.loads((args.campaign / "tasks.json").read_text())):
        for task in tasks:
            folder = args.campaign / f"group{group}" / task["name"]
            receipt = folder / "result.json"
            if not receipt.is_file():
                histories.append({"name": task["name"], "result_missing": True})
                continue
            value = json.loads(receipt.read_text())
            histories.append(
                {"name": task["name"], "receipt": file_id(receipt), "checks": value["checks"], "args": value["args"]}
            )
            if value["args"].get("wall_volume_input") is None:
                by_model[task["name"]] = folder
    comparisons = []
    for model in ("pressure", "intrinsic", "shell"):
        baseline = by_model[f"{model}-b1-fresh0"]
        for label in ("b1-fresh1", "b2-fresh0"):
            comparisons.append(
                {
                    "model": model,
                    "case": "fresh-process" if label == "b1-fresh1" else "B1-vs-first-environment-of-B2",
                    **compare(baseline, by_model[f"{model}-{label}"]),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "kind": "near-slip-batch-process-and-lifecycle-audit",
            "identity": runtime_identity(),
            "source_tasks": file_id(args.campaign / "tasks.json"),
            "histories": histories,
            "cross_process_and_batch_comparisons": comparisons,
            "reference_qualified": False,
            "scope": "Actual checkpoints at control tick427 and61-tick continuations. Every reported restore defect, expected native invalidation and nonfinite count is retained. Batch/fresh comparisons use the first identical case0 input, with Shell mass/thickness variants kept separate. No physical accuracy qualification follows from repeatability.",
        },
    )


if __name__ == "__main__":
    main()
