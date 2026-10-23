"""Compare every retained sharp-edge frame against the unchanged engine baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--diagnostic", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, candidate = [json.loads((root / "result.json").read_text()) for root in [args.baseline, args.candidate]]
    assert baseline["passed"] and candidate["passed"] and len(baseline["rows"]) == len(candidate["rows"])
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "baseline": file_id(args.baseline / "result.json"),
            "candidate": file_id(args.candidate / "result.json"),
            "relative_tolerance": 1e-9,
            "absolute_tolerance": 1e-12,
        },
    )
    rows = []
    for before, after in zip(baseline["rows"], candidate["rows"]):
        assert before["step"] == after["step"] and all(before["accepted"]) and all(after["accepted"])
        for key in set(before) - {"fields"}:
            assert before[key] == after[key], (before["step"], key)
        comparisons = []
        with np.load(before["fields"]["path"]) as original, np.load(after["fields"]["path"]) as changed:
            assert set(original.files) == set(changed.files)
            source_counts = [int(np.count_nonzero(data["law_status"] == 0)) for data in [original, changed]]
            source_slots = original["law_status"].size
            for key in original.files:
                left, right = original[key], changed[key]
                if left.dtype.kind in "biu":
                    conformant = np.array_equal(left, right)
                else:
                    conformant = np.allclose(right, left, rtol=1e-9, atol=1e-12, equal_nan=True)
                finite = np.isfinite(left) & np.isfinite(right)
                error = np.abs(right[finite] - left[finite]).max(initial=0) if left.dtype.kind == "f" else 0
                comparisons.append(
                    {
                        "field": key,
                        "maximum_absolute_difference": float(error),
                        "exact": bool(np.array_equal(left, right, equal_nan=True)),
                        "conformant": bool(conformant),
                        "relative_l2_difference": float(
                            np.linalg.norm(right[finite] - left[finite])
                            / max(np.linalg.norm(left[finite]), np.finfo(float).tiny)
                        )
                        if left.dtype.kind == "f"
                        else 0.0,
                    }
                )
        rows.append(
            {
                "step": before["step"],
                "baseline": before["fields"],
                "candidate": after["fields"],
                "fields": comparisons,
                "loaded_source_counts": source_counts,
                "source_slots": source_slots,
            }
        )
    conformant = all(field["conformant"] for row in rows for field in row["fields"])
    write_json(
        args.output / "result.json",
        {
            "passed": args.diagnostic or conformant,
            "diagnostic_only": args.diagnostic,
            "all_fields_conformant": conformant,
            "rows": rows,
            "all_fields_exact": all(field["exact"] for row in rows for field in row["fields"]),
            "scope": "Every exported array and every admission/status observation, including complete corner/probe tensors when present, native forces, geometry, friction histories and motion. Numerical conformance only; no independent physical accuracy claim.",
            "whole_goal_complete": False,
        },
    )
    assert args.diagnostic or conformant


if __name__ == "__main__":
    main()
