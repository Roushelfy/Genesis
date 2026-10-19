"""Audit complete native pad histories and the measured surface-source adapter."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def wrench(points, forces):
    return np.r_[forces.sum(axis=0), np.cross(points, forces).sum(axis=0) / 0.03]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--compare-motion", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases, baseline = [], None
    keys = ("compression_coordinate_m", "plate_speed_m_per_s", "upper_reaction_n", "lower_reaction_n")
    for run in args.runs:
        receipt = json.loads((run / "result.json").read_text())
        rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        motion = np.array([[row[k] for k in keys] for row in rows])
        if baseline is None:
            baseline = motion
        if args.compare_motion:
            np.testing.assert_allclose(motion, baseline, atol=1e-10, rtol=1e-9)
        fields = []
        for row in rows:
            if "stress" not in row:
                continue
            if not row["stress"].get("field"):
                continue
            packet = run / "frames" / f"stress-level{row['level']}.npz"
            geometry = run / "frames" / f"level{row['level']}.npz"
            with np.load(packet) as saved:
                original_force = saved["original_contact_forces_n"]
                original = wrench(saved["original_contact_points_m"], original_force)
                compressed = wrench(saved["source_points_m"], saved["source_forces_n"])
                recovered = wrench(saved["vertices_m"], saved["forces_n"])
                body = wrench(saved["vertices_m"], saved["rhs_n"] - saved["forces_n"])
            scale = max(np.linalg.norm(original_force, axis=1).sum(), 1e-12)
            compression_error = float(np.linalg.norm(compressed - original) / scale)
            with np.load(geometry) as saved:
                moment = np.cross(
                    saved["points_world_m"] - saved["applied_points_world_m"], saved["contact_force_n"]
                ).sum(axis=0)
            moment_error = float(np.linalg.norm(moment) / (0.03 * max(row["command_force_n"], 1e-3)))
            fields.append(
                {
                    "step": row["step"],
                    "field": file_id(packet),
                    "geometry": file_id(geometry),
                    "native_valid": row["stress"]["native_valid"],
                    "complete_source_compression_relative_wrench_error": compression_error,
                    "recovered_surface_relative_wrench_error": float(np.linalg.norm(recovered - original) / scale),
                    "body_correction_force_n": body[:3].tolist(),
                    "frozen_point_relative_moment_error": moment_error,
                    "adapter_passed": bool(
                        row["stress"]["native_valid"] and compression_error <= 1e-7 and moment_error <= 1e-7
                    ),
                }
            )
        cases.append(
            {
                "run": file_id(run / "result.json"),
                "journal": file_id(run / "steps.jsonl"),
                "steps": len(rows),
                "mechanical_history_passed": receipt["mechanical_history_passed"],
                "all_windows_passed": all(x["mechanical_window_passed"] for x in receipt["windows"]),
                "all_stress_observations_valid": all(row["stress"]["native_valid"] for row in rows if "stress" in row),
                "motion_maximum_absolute_differences": dict(
                    zip(keys, np.max(np.abs(motion - baseline), axis=0).tolist())
                )
                if args.compare_motion
                else None,
                "fields": fields,
            }
        )
    passed = all(
        x["mechanical_history_passed"]
        and x["all_windows_passed"]
        and x["all_stress_observations_valid"]
        and all(f["adapter_passed"] for f in x["fields"])
        for x in cases
    )
    write_json(
        args.output / "result.json",
        {"identity": runtime_identity(), "cases": cases, "passed": passed, "physical_accuracy_qualified": False},
    )
    assert passed, "Native pad history/source adapter failed; audit retained"


if __name__ == "__main__":
    main()
