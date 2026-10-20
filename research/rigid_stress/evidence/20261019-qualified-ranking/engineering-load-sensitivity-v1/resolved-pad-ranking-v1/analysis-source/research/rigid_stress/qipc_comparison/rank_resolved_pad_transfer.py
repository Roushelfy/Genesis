"""Qualify resolved-boundary native fields against the admitted physical reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .stress_error_measures import (
    check_raw_receipt,
    conditional_interval,
    measure_case,
    pair_relation,
    raw_stress_peaks,
    uncertainty_product,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.data_root
    receipts = {}

    def verify(receipt):
        path = receipt["path"]
        if path not in receipts:
            receipts[path] = file_id(path)
        assert receipts[path] == receipt, f"Changed evidence: {path}"

    def read(path):
        verify(file_id(path))
        return json.loads(Path(path).read_text())

    qualification = read(root / "canonical-friction-resilient-qualification-v1/group0/qualification/result.json")
    baseline = read(root / "canonical-stvk-resilient-method-ranking-v1/group0/ranking/result.json")
    admission = read(root / "distributed-pad-history-admission-v1/group0/admission/result.json")
    assert qualification["passed"] and qualification["fine_reference_qualified"] and all(qualification["checks"].values())
    assert baseline["passed"] and baseline["errors_admitted"] and all(baseline["checks"].values())
    assert admission["passed"]
    for receipt in qualification["receipts"] + qualification["verified_inputs"] + baseline["receipts"]:
        verify(receipt)
    run = root / "distributed-pad-every-step-v1/group0/native"
    own = next(row for row in admission["cases"] if row["run"]["path"] == str(run / "result.json"))
    verify(own["run"])
    verify(own["journal"])
    assert own["steps"] == 400 and own["mechanical_history_passed"] and own["all_windows_passed"] and own["all_stress_observations_valid"]
    start = read(run / "start.json")
    end = read(run / "result.json")
    assert end["completed"] and end["mechanical_history_passed"] and not end["history_failed_steps"]
    assert start["args"]["load_transfer"] == "resolved_outer_p2" and start["args"]["source_reduction"] == "none"
    reference_start = read(Path(qualification["accepted_reference_frames"][0]).parent.parent / "start.json")
    for key in ("loads", "dt", "ramp", "hold", "damping", "clearance", "plate_offset_x",
                "shear_distance", "record_stride", "friction", "gravity_x", "gravity_start", "gravity_ramp"):
        assert start["args"][key] == reference_start["args"][key], key
    envelope = {(row["level"], row["metric"]): row for row in qualification["reference_envelopes"]}
    old_errors = {(row["level"], row["metric"]): row for row in baseline["rows"] if row["method"] == "dpr"}
    rows, details, inputs = [], [], []
    for level in (0, 1):
        endpoint = own["fields"][level]
        assert endpoint["step"] == 200 * (level + 1) and endpoint["native_valid"] and endpoint["adapter_passed"]
        assert max(endpoint["complete_source_compression_relative_wrench_error"],
                   endpoint["recovered_surface_relative_wrench_error"], endpoint["frozen_point_relative_moment_error"]) <= 1e-7
        verify(endpoint["field"])
        verify(endpoint["geometry"])
        old_frame = baseline["candidates"]["dpr"]["frames"][level]
        with np.load(endpoint["field"]["path"]) as new, np.load(old_frame["path"]) as old:
            input_changes = {}
            for key in ("original_contact_points_m", "original_contact_forces_n", "vertices_m"):
                scale = np.linalg.norm(old[key])
                change = np.linalg.norm(new[key] - old[key])
                assert scale > 0 and change / scale < 1e-9
                input_changes[key] = {"absolute_l2": float(change), "relative_l2": float(change / scale)}
            body_old, body_new = old["rhs_n"] - old["forces_n"], new["rhs_n"] - new["forces_n"]
            input_changes["mass_balanced_body_load_n"] = {"absolute_l2": float(np.linalg.norm(body_new - body_old))}
        inputs.append({"level": level, "paired_input_changes": input_changes})
        measurements, cases = {}, {}
        for order_index, order in enumerate((9, 25)):
            directory = root / "resolved-pad-qualified-reference-v1" / f"group{2 * level + order_index}" / f"resolved-level{level}-q{order}"
            result = read(directory / "result.json")
            assert result["physical_volume_weighting"] and result["common_peaks_evaluated"]
            assert result["surface_quadrature_points_per_face"] == order and result["depth_quadrature_points"] == 4
            case = result["cases"][0]
            assert len(result["cases"]) == 1 and len(case["raw"]) == 2 and case["load_n"] == 0.1
            assert case["raw"][0]["field"] == endpoint["field"] and case["raw"][0]["mesh"] == start["mesh"]
            assert case["raw"][1]["field"]["path"] == qualification["accepted_reference_frames"][level]
            assert case["raw"][1]["mesh"] == reference_start["mesh"]
            for raw in case["raw"]:
                verify(raw["field"])
                verify(raw["mesh"])
                if order == 25:
                    check_raw_receipt(raw, raw_stress_peaks(raw["mesh"]["path"], raw["field"]["path"]))
            column, surface = directory / "columns-0.1-0.npz", directory / "surface-fields-0.1-0.npz"
            verify(file_id(column))
            verify(file_id(surface))
            measurement = measure_case(case, column, surface)
            assert all(measurement["checks"].values()) and len(measurement["rows"]) == 9
            measurements[order] = {row["metric"]: row for row in measurement["rows"]}
            cases[order] = case
        for metric, high in measurements[25].items():
            low = measurements[9][metric]
            key = metric if metric.startswith("common_") else metric + ("_relative_change" if metric.startswith("raw_") else "_relative_l2")
            observed = envelope[level, key]
            reference_change = uncertainty_product(observed[k] for k in
                ("spatial_change", "time_change", "contact_change", "angular_change", "depth_change"))
            rms_sampling = abs(low["reference_rms_or_peak"] / high["reference_rms_or_peak"] - 1)
            method_sampling = abs(low["relative_error"] - high["relative_error"])
            if metric.startswith("common_"):
                peaks = [next(p for p in cases[q]["adjacent_comparisons"][0]["common_spatial_measures"]["tangent_peaks"]
                              if p["metric"] == metric.removeprefix("common_")) for q in (9, 25)]
                method_sampling = abs(peaks[0]["actual"]["stress_pa"] - peaks[1]["actual"]["stress_pa"]) / high["reference_rms_or_peak"]
                assert abs(peaks[0]["actual"]["stress_pa"] / peaks[1]["actual"]["stress_pa"] - 1) <= 0.01
            target = 0.01 if metric.startswith(("raw_", "common_")) else 0.005
            assert method_sampling <= target and rms_sampling <= target
            reference_change = (1 + reference_change) * (1 + rms_sampling) - 1
            interval = conditional_interval(high["relative_error"], reference_change, method_sampling)
            old = old_errors[level, metric]
            np.testing.assert_allclose(high["reference_rms_or_peak"], old["reference_rms_or_peak"], rtol=1e-9, atol=0)
            rows.append({"level": level, "method": "resolved_outer", **high,
                "conditional_lower": interval[0], "conditional_upper": interval[1],
                "conditional_reference_envelope": reference_change, "method_sampling_change": method_sampling,
                "fixed_radius_relative_error": old["relative_error"],
                "resolved_vs_fixed_conditional_relation": pair_relation(interval, (old["conditional_lower"], old["conditional_upper"]))})
        details.append({"level": level, "raw": cases[25]["raw"],
                        "common_spatial_measures": cases[25]["adjacent_comparisons"][0]["common_spatial_measures"]})
    with (args.output / "errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": True, "errors_admitted": True, "rows": rows, "details": details, "paired_inputs": inputs,
        "receipts": list(receipts.values()), "whole_goal_complete": False, "certified_continuum_error_bound": False,
        "scope": "Own continuous-history and endpoint admission for resolved-boundary native transfer, on the same physical pad problem and previously qualified fine reference. Complete retained arrays are independently reintegrated. Conditional intervals use observed reference and sampling envelopes, not certified continuum bounds. Input conformance does not equate differing transferred loads. Scalar error norms are not an attribution; the separate three-field signed tensor decomposition supplies that contrast."})


if __name__ == "__main__":
    main()
