"""Compare retained physical/checkpoint arrays, naming every differing field."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with np.load(args.reference) as reference, np.load(args.candidate) as candidate:
        assert set(reference.files) == set(candidate.files)
        for name in reference.files:
            first, second = reference[name], candidate[name]
            if first.shape != second.shape or first.dtype != second.dtype:
                rows.append(
                    {
                        "name": name,
                        "shape": list(first.shape),
                        "candidate_shape": list(second.shape),
                        "dtype": str(first.dtype),
                        "candidate_dtype": str(second.dtype),
                        "equal_including_nan_positions": False,
                        "different_values": None,
                        "scope": "Storage layout differs; values are not compared across unequal shapes/dtypes.",
                    }
                )
                continue
            equal = (first == second) | (np.isnan(first) & np.isnan(second))
            finite = np.isfinite(first) & np.isfinite(second)
            difference = np.abs(second[finite].astype(float) - first[finite].astype(float))
            indices = np.argwhere(~equal)
            rows.append(
                {
                    "name": name,
                    "shape": list(first.shape),
                    "dtype": str(first.dtype),
                    "equal_including_nan_positions": bool(equal.all()),
                    "different_values": int((~equal).sum()),
                    "first_different_index": indices[0].tolist() if len(indices) else None,
                    "maximum_finite_absolute_difference": float(difference.max(initial=0)),
                    "nonfinite_pattern_equal": bool(np.array_equal(np.isfinite(first), np.isfinite(second))),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "comparison_completed": True,
            "all_arrays_equal": all(row["equal_including_nan_positions"] for row in rows),
            "array_count": len(rows),
            "different_arrays": [row for row in rows if not row["equal_including_nan_positions"]],
            "arrays": rows,
            "reference": file_id(args.reference),
            "candidate": file_id(args.candidate),
            "scope": "Direct complete-array comparison. NaNs are equal only at identical positions. Serialized arrays include meaningful parameters/state and retained derived buffers; differences in a field are evidence to inspect, not automatically a physics defect. No motion tolerance or production admission changes.",
        },
    )


if __name__ == "__main__":
    main()
