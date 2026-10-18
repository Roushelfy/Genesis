"""Summarize replay layers and independently verify the physical point-wrench packet contract."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    force_defects, moment_defects = [], []
    unit_control = None
    for path in sorted((args.packets / "frames").glob("step*.npz")):
        with np.load(path) as saved:
            position, force = saved["physical_contact_positions_material_m"], saved["contact_forces_material_n"]
            points, compressed = saved["compressed_positions_material_m"], saved["compressed_forces_material_n"]
            force_defects.append(np.linalg.norm(force.sum(axis=0) - compressed.sum(axis=0)).item())
            moment_defects.append(
                np.linalg.norm(np.cross(position, force).sum(axis=0) - np.cross(points, compressed).sum(axis=0)).item()
            )
            if np.isclose(saved["time_s"].item(), 0.2, rtol=0, atol=1e-12):
                barrier = saved["normal_barrier_forces_material_n"]
                active = np.linalg.norm(force, axis=1) > 1e-12
                assert active.sum() == 1
                magnitude = np.linalg.norm(barrier[active], axis=1).item()
                projected = np.einsum("vi,vi->", force[active], barrier[active] / magnitude).item()
                relative = abs(projected - magnitude) / magnitude
                assert relative < 0.01, "Normal-barrier units/timing do not agree with recorded plane reaction"
                unit_control = {
                    "time_s": 0.2,
                    "normal_barrier_n": magnitude,
                    "recorded_force_normal_projection_n": projected,
                    "relative_difference": relative,
                    "frame": file_id(path),
                }
    assert unit_control is not None
    histories = []
    for folder in args.replays:
        receipt = json.loads((folder / "result.json").read_text())
        rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        statistics = {}
        for name in ("common_tangent_error", "independent_p2_core_error"):
            selected = [row for row in rows if row[name] is not None]
            relative = [
                row[name]["relative_tensor_error"] for row in selected if row[name]["relative_tensor_error"] is not None
            ]
            statistics[name] = {
                "admitted_observations": len(selected),
                "nonzero_reference_observations": len(relative),
                "maximum_relative_error": max(relative) if relative else None,
                "median_relative_error": np.median(relative).item() if relative else None,
                "maximum_absolute_rms_pa": max(
                    (row[name]["absolute_tensor_rms_error_pa"] for row in selected), default=None
                ),
            }
        histories.append(
            {
                "receipt": receipt,
                "receipt_identity": file_id(folder / "result.json"),
                "statistics": statistics,
                "invalid_frames": [
                    {"time_s": row["time_s"], "errno": row["errno"], "contact_status": row["contact_status"]}
                    for row in rows
                    if not row["valid"]
                ],
                "maximum_complete_relative_residual": max(
                    (row["complete_force_relative_residual"] for row in rows if row["valid"]), default=None
                ),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "resolved-point-interface-and-inertia-replay-audit",
            "identity": runtime_identity(),
            "packets": file_id(args.packets / "result.json"),
            "maximum_physical_force_compression_defect_n": max(force_defects),
            "maximum_physical_moment_compression_defect_nm": max(moment_defects),
            "plane_normal_gradient_unit_control": unit_control,
            "replays": histories,
            "reference_qualified": False,
            "scope": "Each replay layer remains separate. Rejected fields and near-zero-reference denominators are explicit; scalar error norms are not added as a budget. Point-packet wrench checks use actual deformed physical application coordinates, not undeformed mesh vertices.",
        },
    )


if __name__ == "__main__":
    main()
