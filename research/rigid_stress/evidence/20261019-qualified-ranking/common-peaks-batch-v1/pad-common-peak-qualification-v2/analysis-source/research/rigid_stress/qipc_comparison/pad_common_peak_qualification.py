"""Qualify common tangent-plane peak changes and retain separate full3D components."""

import argparse
import csv
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def read(relative):
        path = args.data_root / relative
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    normal = read("pad-normal-qualification-v2/group0/qualification/result.json")
    checks = {"normal_reference_and_histories": normal["normal_reference_change_targets_passed"]}
    controls = {}
    for group, name in enumerate(("quadrature", "wall125", "surface3", "joint")):
        result = read(f"common-pad-peak-reference-v1/group{group}/{name}/result.json")
        assert result["common_peaks_evaluated"] and result["physical_volume_weighting"]
        controls[name] = {
            case["load_n"]: {
                p["metric"]: p["relative_difference"]
                for p in case["adjacent_comparisons"][0]["common_spatial_measures"]["tangent_peaks"]
            }
            for case in result["cases"]
        }
    checks["spatial_peak_change_target"] = all(
        value <= 0.05
        for name in ("wall125", "surface3", "joint")
        for load in controls[name].values()
        for value in load.values()
    )
    checks["contact_quadrature_peak_target"] = all(
        value <= 0.01 for load in controls["quadrature"].values() for value in load.values()
    )
    rows, transverse, rankings = [], [], []
    sampling_passed = True
    for level, load in enumerate((0.1, 0.3, 1.0)):
        intervals = {name: {} for name in ("mises", "tension")}
        reference_peaks = {}
        for method_index, method in enumerate(("dpr", "scr", "shell")):
            samples = []
            for group_offset in (0, 1):
                path = (
                    f"common-pad-peaks-v1/group{group_offset}/{method}-0/result.json"
                    if level == 0
                    else f"common-pad-peak-loads-v1/group{method_index + 3 * group_offset}/{method}-{level}/result.json"
                )
                result = read(path)
                assert result["common_peaks_evaluated"] and result["cases"][0]["load_n"] == load
                samples.append(result["cases"][0]["adjacent_comparisons"][0]["common_spatial_measures"])
            for low, high in zip(samples[0]["tangent_peaks"], samples[1]["tangent_peaks"]):
                metric = high["metric"]
                actual, reference = high["actual"]["stress_pa"], high["reference"]["stress_pa"]
                reference_peaks.setdefault(metric, reference)
                assert abs(reference / reference_peaks[metric] - 1) < 1e-10, (
                    "Shared reference peak depends on compared method"
                )
                sampling_actual = abs(actual - low["actual"]["stress_pa"])
                sampling_reference = abs(reference - low["reference"]["stress_pa"])
                sampling_passed &= sampling_actual <= 0.01 * actual and sampling_reference <= 0.01 * reference
                spatial = max(controls[name][load][metric] for name in ("wall125", "surface3", "joint"))
                contact = controls["quadrature"][load][metric]
                delta = (1 + spatial) * (1 + contact) * (1 + sampling_reference / reference) - 1
                actual_delta = sampling_actual / reference
                error = high["relative_difference"]
                lower = max(error - delta - actual_delta, 0) / (1 + delta)
                upper = (error + delta + actual_delta) / (1 - delta)
                intervals[metric][method] = (lower, upper)
                rows.append(
                    {
                        "load_n": load,
                        "method": method,
                        "metric": metric,
                        "actual_peak_pa": actual,
                        "reference_peak_pa": reference,
                        "relative_error": error,
                        "observed_reference_resolution_envelope": delta,
                        "actual_sampling_change_over_reference": actual_delta,
                        "conditional_lower": lower,
                        "conditional_upper": upper,
                        "actual_peak_position_m": high["actual"]["common_reference_rest_position_m"],
                        "reference_peak_position_m": high["reference"]["common_reference_rest_position_m"],
                    }
                )
            transverse.append(
                {
                    "load_n": load,
                    "method": method,
                    "components": samples[1]["full3d_only_components"],
                    "transverse_frobenius_energy_fraction": samples[1]["transverse_frobenius_energy_fraction"],
                }
            )
        for metric, methods in intervals.items():
            pairs = []
            for a, b in (("dpr", "scr"), ("dpr", "shell"), ("scr", "shell")):
                relation = (
                    a + " lower error"
                    if methods[a][1] < methods[b][0]
                    else b + " lower error"
                    if methods[b][1] < methods[a][0]
                    else "unresolved"
                )
                pairs.append({"first": a, "second": b, "relation": relation})
            rankings.append({"load_n": load, "metric": metric, "pairs": pairs})
    checks["common_angular_sampling_target"] = bool(sampling_passed)
    with (args.output / "common-peaks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "checks": checks,
            "common_peak_change_targets_passed": all(checks.values()),
            "peaks": rows,
            "full3d_components": transverse,
            "conditional_rankings": rankings,
            "receipts": receipts,
            "certified_continuum_bound": False,
            "whole_goal_complete": False,
            "scope": "Normal physical-pad common tangent-plane Mises/tension peaks on the same125um/three-wall angular material grid,9/25 samples per face and complete one-sided through-wall partition traces. Numerical5% spatial and1% integration/sampling targets retain all extrema; no smoothing. Conditional ranges assume observed spatial/contact/angular changes bound unresolved reference and method sampling error. They are not statistical intervals or a continuum proof. Raw full3D maxima remain separate in the field receipts. Full3D normal/transverse shear components are reported without assigning Shell an unsupported component or error rank. Friction spatial qualification and matched repeated/batched primary costs remain required.",
        },
    )
    assert all(checks.values()), "Common peak qualification failed; preserve actual checks"


if __name__ == "__main__":
    main()
