"""Compare complete native recovery outputs across boundary-traction implementations."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline-normal", "candidate-normal", "baseline-hard", "candidate-hard", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "relative_tolerance": 1e-7,
            "stress_absolute_tolerance_pa": 1e-6,
            "other_absolute_tolerance": 1e-12,
            "scope": "Complete supplied-normal and sharp/friction recovery outputs. Stress tolerances match public observation checks. Positions/displacements use1e-12m, forces1e-12N and dimensionless fields1e-12 absolute. Integer/mask fields compare exactly. Independent reference and lifecycle gates remain in the producing controls.",
        },
    )
    pairs = [
        (args.baseline_normal / f"{kind}-{mode}.npz", args.candidate_normal / f"{kind}-{mode}.npz")
        for kind in ["analytic", "estimated"]
        for mode in ["full", "max"]
    ]
    pairs.append((args.baseline_hard / "fields.npz", args.candidate_hard / "fields.npz"))
    rows = []
    for original, changed in pairs:
        fields = []
        with np.load(original) as baseline, np.load(changed) as candidate:
            assert set(baseline.files) == set(candidate.files)
            for key in baseline.files:
                left, right = baseline[key], candidate[key]
                if left.dtype.kind in "biu":
                    np.testing.assert_array_equal(right, left)
                else:
                    np.testing.assert_allclose(right, left, rtol=1e-7, atol=1e-6 if key.endswith("_pa") else 1e-12)
                finite = np.isfinite(left) & np.isfinite(right)
                difference = right[finite] != left[finite] if left.dtype.kind == "b" else right[finite] - left[finite]
                norm = np.linalg.norm(left[finite])
                fields.append(
                    {
                        "field": key,
                        "size": left.size,
                        "maximum_absolute_difference": float(np.max(np.abs(difference), initial=0)),
                        "relative_l2_difference": float(np.linalg.norm(difference) / norm) if norm > 0 else None,
                        "exact": bool(np.array_equal(right, left, equal_nan=True)),
                    }
                )
        rows.append({"baseline": file_id(original), "candidate": file_id(changed), "fields": fields})
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
