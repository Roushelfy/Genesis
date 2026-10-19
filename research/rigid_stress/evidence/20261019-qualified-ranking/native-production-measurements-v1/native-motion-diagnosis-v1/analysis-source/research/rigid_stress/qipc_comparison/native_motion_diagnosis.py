"""Locate construction, repeat and post-integration differences without relaxing motion targets."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .audit_native_production_controls import MOTION


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.data_root / "native-production-controls-v1/group1"
    controls = args.data_root / "native-motion-diagnostics-v1/group0"
    folders = {
        "original-rigid": base / "rigid-max",
        "original-dpr": base / "dpr-max",
        **{name: controls / name for name in ("rigid-repeat", "dpr-unrequested", "dpr-post-off")},
    }
    arrays, receipts = {}, []
    for name, folder in folders.items():
        result = json.loads((folder / "result.json").read_text())
        assert result["passed"] and result["steps"] == 1200 and result["envs"] == 8
        assert file_id(folder / "trajectory.npz") == result["trajectory"]
        receipts.extend([file_id(folder / "result.json"), result["trajectory"]])
        with np.load(folder / "trajectory.npz") as saved:
            arrays[name] = saved["motion"].copy()
    comparisons = []
    for actual, reference in (
        ("rigid-repeat", "original-rigid"),
        ("dpr-unrequested", "rigid-repeat"),
        ("dpr-post-off", "rigid-repeat"),
        ("dpr-post-off", "original-dpr"),
        ("dpr-unrequested", "original-dpr"),
    ):
        checks = []
        for metric, first, last, tolerance in MOTION:
            a, b = arrays[actual][..., first:last], arrays[reference][..., first:last]
            if metric == "quaternion_up_to_sign":
                a = np.where((a * b).sum(axis=-1, keepdims=True) < 0, -a, a)
            delta = np.abs(a - b)
            per_step = delta.max(axis=(1, 2))
            nonzero = np.flatnonzero(per_step > 0)
            exceeded = np.flatnonzero(per_step > tolerance)
            checks.append(
                {
                    "metric": metric,
                    "maximum_absolute_difference": float(delta.max()),
                    "unchanged_target": tolerance,
                    "passes_target": bool((delta <= tolerance).all()),
                    "first_nonzero_step": int(nonzero[0] + 1) if len(nonzero) else None,
                    "first_threshold_exceeding_step": int(exceeded[0] + 1) if len(exceeded) else None,
                }
            )
        comparisons.append(
            {
                "actual": actual,
                "reference": reference,
                "checks": checks,
                "motion_targets_passed": all(c["passes_target"] for c in checks),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "diagnosis_completed": True,
            "comparisons": comparisons,
            "receipts": receipts,
            "production_admitted": False,
            "scope": "Complete1200-step B8 differences among fresh rigid repeat, stress-built unrequested/manual, fused-post-off and original native paths. No threshold change or causal conclusion beyond the measured contrasts.",
        },
    )


if __name__ == "__main__":
    main()
