"""Compare every-step peaks and endpoint fields of the two resolved transfers."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--native", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"peak_relative": 1e-8, "peak_absolute_pa": 1e-6, "complete_field_relative": 1e-6,
               "original_source_relative": 1e-9}
    histories, receipts, settings = [], [], []
    for directory in (args.baseline, args.native):
        start, result = (json.loads((directory / name).read_text()) for name in ("start.json", "result.json"))
        rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        assert result["completed"] and result["mechanical_history_passed"] and not result["history_failed_steps"]
        assert result["steps"] == len(rows) == 400
        assert all(w["mechanical_window_passed"] for w in result["windows"])
        assert all(row["stress"]["native_valid"] for row in rows)
        histories.append(rows)
        settings.append(start["args"])
        receipts.extend(file_id(directory / name) for name in ("start.json", "result.json", "steps.jsonl"))
    assert settings[0]["load_transfer"] == "resolved_outer_p2"
    assert settings[1]["load_transfer"] == "resolved_boundary_p2" and settings[1]["source_roundoff"]
    for key in ("mesh", "assets", "loads", "dt", "ramp", "hold", "friction", "gravity_x", "method",
                "source_reduction", "stress_stride"):
        assert settings[0][key] == settings[1][key], key
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "inputs": receipts, "targets": targets,
        "scope": "Pure transfer implementation comparison on each completed400-step physical history. The native path explicitly enables the previously studied bounded source-input reconciliation."})
    peaks = np.array([[row["stress"]["raw_vm_peak_pa"] for row in rows] for rows in histories])
    assert np.isfinite(peaks).all()
    np.testing.assert_allclose(peaks[1], peaks[0], rtol=targets["peak_relative"], atol=targets["peak_absolute_pa"])
    corrections = []
    for row in histories[1]:
        correction = row["stress"]["source_roundoff"]
        assert correction["force_change_bound_n"] <= correction["physical_force_budget_n"]
        if correction["corrected_sources"]:
            corrections.append({"step": row["step"], **correction})
    fields = []
    for level in (0, 1):
        paths = [directory / f"frames/stress-level{level}.npz" for directory in (args.baseline, args.native)]
        receipts.extend(file_id(path) for path in paths)
        with np.load(paths[0]) as first, np.load(paths[1]) as second:
            np.testing.assert_array_equal(first["vertices_m"], second["vertices_m"])
            np.testing.assert_array_equal(first["tetrahedra"], second["tetrahedra"])
            np.testing.assert_array_equal(first["original_contact_points_m"], second["original_contact_points_m"])
            errors = {}
            for key in ("stress_pa", "forces_n", "rhs_n", "original_contact_forces_n"):
                scale = np.linalg.norm(first[key])
                assert scale > 0
                error = float(np.linalg.norm(second[key] - first[key]) / scale)
                assert error <= targets["original_source_relative" if key == "original_contact_forces_n"
                                        else "complete_field_relative"], (level, key, error)
                errors[key] = error
        fields.append({"level": level, "errors": errors, "baseline": file_id(paths[0]), "native": file_id(paths[1])})
    path = args.output / "every-step-peaks.npz"
    np.savez_compressed(path, baseline_pa=peaks[0], native_pa=peaks[1])
    write_json(args.output / "result.json", {"passed": True, "observations": 400, "targets": targets,
        "maximum_peak_absolute_difference_pa": float(np.abs(peaks[1] - peaks[0]).max()),
        "maximum_peak_mixed_scaled_difference": float((np.abs(peaks[1] - peaks[0]) /
            (targets["peak_absolute_pa"] + targets["peak_relative"] * np.abs(peaks[0]))).max()),
        "corrections": corrections, "fields": fields, "peaks": file_id(path), "receipts": receipts,
        "whole_goal_complete": False, "automatic_pressure_prediction_admitted": False,
        "scope": "Actual complete histories and both endpoints agree within unchanged predeclared numerical checks. Correction magnitudes are retained per observation; original source arrays are retained at the endpoint keyframes. No new per-event correction stress bound is assigned to intermediate rows whose complete source arrays were not captured. Earlier exact P2 roundoff controls remain separate evidence. This is neither continuum accuracy nor production throughput."})


if __name__ == "__main__":
    main()
