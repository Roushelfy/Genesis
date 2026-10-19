"""Recover dimensional normal-pad errors from retained complete material columns."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = {}

    def record(path, expected=None):
        path = Path(path)
        if str(path) not in receipts:
            receipts[str(path)] = file_id(path)
        if expected is not None:
            assert receipts[str(path)] == expected, f"Changed input: {path}"
        return receipts[str(path)]

    def read(relative):
        path = args.data_root / relative
        record(path)
        return json.loads(path.read_text())

    normal = read("pad-normal-qualification-v2/group0/qualification/result.json")
    peaks = read("pad-common-peak-qualification-v2/group0/qualification/result.json")
    assert normal["normal_reference_change_targets_passed"] and all(normal["checks"].values())
    assert peaks["common_peak_change_targets_passed"] and all(peaks["checks"].values())
    for audit in (normal, peaks):
        for receipt in audit["receipts"]:
            record(receipt["path"], receipt)
    qualified_errors = {(r["load_n"], r["method"], r["metric"]): r for r in normal["errors"]}
    qualified_peaks = {(r["load_n"], r["method"], r["metric"]): r for r in peaks["peaks"]}
    rows, details, reference_norms = [], [], {}
    for level, load in enumerate((0.1, 0.3, 1.0)):
        for group, method in enumerate(("dpr", "scr", "shell")):
            folder = f"full-source-joint-normal-fields-v1/group{group}/{method}-{level}"
            result = read(f"{folder}/result.json")
            assert result["physical_volume_weighting"]
            assert result["surface_quadrature_points_per_face"] == 9
            assert result["depth_quadrature_points"] == 2
            assert len(result["cases"]) == 1
            case = result["cases"][0]
            assert case["load_n"] == load and len(case["adjacent_comparisons"]) == 1
            comparison = case["adjacent_comparisons"][0]
            for raw in case["raw"]:
                for key in ("mesh", "field"):
                    record(raw[key]["path"], raw[key])
            column_path = args.data_root / folder / f"columns-{load:g}-0.npz"
            record(column_path)

            def add(metric, absolute, reference, unit, relative, quadrature, qualification_key=None):
                assert np.isfinite([absolute, reference, relative]).all()
                assert absolute >= 0 and reference > 0
                np.testing.assert_allclose(absolute / reference, relative, rtol=1e-11, atol=1e-14)
                if qualification_key is not None:
                    np.testing.assert_allclose(
                        relative,
                        qualified_errors[load, method, qualification_key]["relative_error"],
                        rtol=1e-12,
                        atol=1e-14,
                    )
                # A shared reference norm must not depend on the compared method.
                key = (load, metric)
                reference_norms.setdefault(key, reference)
                np.testing.assert_allclose(reference, reference_norms[key], rtol=1e-9, atol=0)
                rows.append(
                    {
                        "load_n": load,
                        "method": method,
                        "metric": metric,
                        "absolute_error": float(absolute),
                        "unit": unit,
                        "reference_rms_or_peak": float(reference),
                        "relative_error": float(relative),
                        "surface_quadrature_points_per_face": quadrature,
                    }
                )

            with np.load(column_path) as columns:
                area = columns["area_m2"]
                assert area.ndim == 1 and np.isfinite(area).all() and np.all(area > 0)
                for metric, array, unit in (("membrane", "membrane_n_per_m", "N/m"), ("bending", "bending_n", "N")):
                    values = columns[array]
                    assert values.shape == (2, len(area), 2, 2) and np.isfinite(values).all()
                    difference = values[0] - values[1]
                    error = np.sqrt(np.einsum("qij,qij,q->", difference, difference, area) / area.sum())
                    reference = np.sqrt(np.einsum("qij,qij,q->", values[1], values[1], area) / area.sum())
                    add(metric, error, reference, unit, comparison[metric + "_relative_l2"], 9, metric + "_relative_l2")
                assert comparison["integration_measure_unit"] == "m3" and comparison["integration_measure"] > 0
                error2 = columns["error_square_pa2_m3"]
                reference2 = columns["reference_square_pa2_m3"]
                assert np.isfinite(error2).all() and np.isfinite(reference2).all()
                assert np.all(error2 >= 0) and np.all(reference2 >= 0)
                error = np.sqrt(error2.sum() / comparison["integration_measure"])
                reference = np.sqrt(reference2.sum() / comparison["integration_measure"])
                np.testing.assert_allclose(error, comparison["tangent_rms_error_pa"], rtol=1e-12, atol=0)
                add("tangent", error, reference, "Pa", comparison["tangent_relative_l2"], 9, "tangent_relative_l2")
            for surface in ("outer", "inner"):
                metric = surface + "_tangent"
                relative = comparison[metric + "_relative_l2"]
                assert relative > 0  # These retained normal cases have nonzero errors; no invented floor.
                error = comparison[metric + "_rms_error_pa"]
                add(metric, error, error / relative, "Pa", relative, 9, metric + "_relative_l2")
            for metric in ("mises", "tension"):
                actual, reference = [raw[f"raw_maximum_{metric}_pa"] for raw in case["raw"]]
                key = f"raw_{metric}_relative_change"
                add("raw_" + metric, abs(actual - reference), reference, "Pa", comparison[key], None, key)

            peak_path = (
                f"common-pad-peaks-v1/group1/{method}-0/result.json"
                if level == 0
                else f"common-pad-peak-loads-v1/group{group + 3}/{method}-{level}/result.json"
            )
            common = read(peak_path)
            assert common["surface_quadrature_points_per_face"] == 25 and common["common_peaks_evaluated"]
            peak_case = common["cases"][0]
            assert peak_case["load_n"] == load
            assert peak_case["raw"] == case["raw"], "Peak and RMS comparisons must use identical actual fields"
            measures = peak_case["adjacent_comparisons"][0]["common_spatial_measures"]
            for peak in measures["tangent_peaks"]:
                metric = peak["metric"]
                reference = peak["reference"]["stress_pa"]
                absolute = abs(peak["actual"]["stress_pa"] - reference)
                np.testing.assert_allclose(absolute, peak["absolute_difference_pa"], rtol=1e-12, atol=0)
                add("common_" + metric, absolute, reference, "Pa", peak["relative_difference"], 25)
                np.testing.assert_allclose(
                    peak["relative_difference"],
                    qualified_peaks[load, method, metric]["relative_error"],
                    rtol=1e-12,
                    atol=1e-14,
                )
            details.append(
                {
                    "load_n": load,
                    "method": method,
                    "raw": case["raw"],
                    "common_spatial_measures_q25": measures,
                    "columns": receipts[str(column_path)],
                }
            )
    assert len(rows) == 81 and len(details) == 9
    with (args.output / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": True,
            "rows": rows,
            "details": details,
            "receipts": list(receipts.values()),
            "whole_goal_complete": False,
            "scope": "Nine normal-pad comparisons. RMS uses the Frobenius tensor norm, full physical volume "
            "for tangent stress, and full mid-surface area for membrane, bending and boundary fibers. "
            "Membrane and bending use the original declared 500um material thickness. All retained columns "
            "are included without trimming. Reference surface RMS is derived from the already integrated "
            "absolute/relative pair, not independently reintegrated. Fields use q9/depth2 and common peaks "
            "use q25; raw peaks keep their original distinct samplings. Full3D-only components and peak "
            "locations remain in details; the saved raw position is the raw Mises location, not a tensile "
            "location. No near-zero floor is used: every reference normalization is positive. Existing "
            "conditional uncertainty/ranks remain unchanged; this is no new continuum qualification.",
        },
    )


if __name__ == "__main__":
    main()
