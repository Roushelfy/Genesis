"""Inspect spatial/time variation and history versus proportional shear."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .hertz_tangential import elliptic_tangential_stress
from .study_hertz_nearfield import stress_measures


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def peaks(tensor):
    tensile, vm = stress_measures(tensor)
    return np.array([tensile.max(), vm.max()])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    roots = [args.campaign / f"group{i}/study" for i in range(5)]
    results = [json.loads((root / "result.json").read_text()) for root in roots]
    assert all(row["passed"] and row["numerical_reference_execution_passed"] for row in results)
    targets = {"tensor": 0.05, "peak": 0.10, "reference_change": 0.02}
    write_json(
        args.output / "start.json",
        {
            "targets": targets,
            "inputs": [file_id(root / "result.json") for root in roots],
            "reference_package_versions": {name: version(name) for name in ["numpy", "scipy", "cvxopt"]},
        },
    )
    variations, comparisons, bindings = [], [], []
    for coarse, fine, kind in [(0, 1, "nu0_spatial"), (2, 3, "nu03_spatial"), (4, 2, "nu03_time")]:
        for entry in results[fine]["rows"]:
            label = entry["label"]
            paths = [roots[i] / (label + ".npz") for i in [coarse, fine]]
            bindings.extend(file_id(path) for path in paths)
            with np.load(paths[0]) as left, np.load(paths[1]) as right:
                np.testing.assert_allclose(left["points"], right["points"], rtol=0, atol=1e-14)
                np.testing.assert_array_equal(left["normal_tensor"], right["normal_tensor"])
                a, b = left["reference_tangent"], right["reference_tangent"]
                normal = right["normal_tensor"]
                peak_change = np.abs(peaks(a + normal) - peaks(b + normal)) / np.maximum(
                    np.abs(peaks(b + normal)), 1e-30
                )
                row = {
                    "kind": kind,
                    "label": label,
                    "tangent_relative": relative(a, b),
                    "combined_relative": relative(a + normal, b + normal),
                    "tangent_change_over_normal_scale": relative(a - b, normal),
                    "tensile_peak_relative": float(peak_change[0]),
                    "vm_peak_relative": float(peak_change[1]),
                }
                row["strict_tangent_reference_change_target_met"] = (
                    row["tangent_relative"] < targets["reference_change"]
                )
                row["combined_field_peak_reference_change_target_met"] = (
                    max(row["combined_relative"], row["tensile_peak_relative"], row["vm_peak_relative"])
                    < targets["reference_change"]
                )
                variations.append(row)
    for group in [1, 3]:
        result = results[group]
        for entry in result["rows"]:
            label = entry["label"]
            normal = 1.0
            if label.startswith("waypoint-"):
                segment = int(label.split("-")[1]) - 1
                normal = next(
                    row["normal"]
                    for row in result["history"]
                    if row["segment"] == segment and row["part"] == result["subdivisions"]
                )
            radius = normal ** (1 / 3)
            peak = 3 * normal / (2 * np.pi * radius**2)
            path = roots[group] / (label + ".npz")
            bindings.append(file_id(path))
            with np.load(path) as saved:
                pressure = saved["normal_tensor"]
                reference = saved["reference_tangent"]
                ratio = saved["reference_cell_force"].sum(axis=0) / normal
                proportional = elliptic_tangential_stress(
                    saved["points"], np.array([radius, radius]), ratio, peak, result["poisson"]
                )
                np.savez_compressed(args.output / f"group{group}-{label}-proportional.npz", tensor=proportional)
                reference_peaks = peaks(pressure + reference)
                for model, tensor in [("radial_history16", saved["candidate_tangent"]), ("proportional", proportional)]:
                    peak_error = np.abs(peaks(tensor + pressure) - reference_peaks) / np.maximum(
                        np.abs(reference_peaks), 1e-30
                    )
                    tensor_error = relative(tensor + pressure, reference + pressure)
                    comparisons.append(
                        {
                            "group": group,
                            "poisson": result["poisson"],
                            "label": label,
                            "model": model,
                            "tangent_relative": relative(tensor, reference),
                            "combined_relative": tensor_error,
                            "tensile_peak_relative": float(peak_error[0]),
                            "vm_peak_relative": float(peak_error[1]),
                            "preexisting_combined_5p_peak_10p_targets_met_on_bank": bool(
                                tensor_error < targets["tensor"] and peak_error.max() < targets["peak"]
                            ),
                            "strict_tangent_5p_target_met": relative(tensor, reference) < targets["tensor"],
                        }
                    )
    exact_controls = [
        row["analytical_collinear_tensor_relative"]
        for row in results[1]["rows"]
        if row["analytical_collinear_tensor_relative"] is not None
    ]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "targets": targets,
            "variations": variations,
            "comparisons": comparisons,
            "nu0_analytical_monotonic_controls": exact_controls,
            "nu0_analytical_controls_within_2p": max(exact_controls) < targets["reference_change"],
            "all_combined_reference_changes_within_2p": all(
                row["combined_field_peak_reference_change_target_met"] for row in variations
            ),
            "all_strict_tangent_reference_changes_within_2p": all(
                row["strict_tangent_reference_change_target_met"] for row in variations
            ),
            "bindings": bindings,
            "scope": "Common positive-depth96-point banks under changing normal and two-component tangential forces, similar elastic half-spaces and supplied circular Hertz pressure. Analytical nu0 monotonic controls, spatial17/25 and time4/8 comparisons are explicit. The paired data keep separate combined engineering budgets and additional strict tangent targets. Observed reference changes are not rigorous continuum uncertainty bounds; no surface/global/full-volume or native-history qualification follows.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
