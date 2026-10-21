"""Inspect retained shell tensors, peak locations and independent-reference gates."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_hertz_nearfield import stress_measures


def peaks(field, corners, positions, probes):
    tensors = np.concatenate((field, corners.reshape(-1, 3, 3)))
    coordinates = np.concatenate((probes, positions.reshape(-1, 3)))
    tension, vm = stress_measures(tensors)
    return {
        key: {"value_pa": float(value.max()), "position_m": coordinates[value.argmax()].tolist()}
        for key, value in [("tensile", tension), ("von_mises", vm)]
    }


def compare_peaks(candidate, reference):
    return {
        key: {
            "relative_value_difference": abs(candidate[key]["value_pa"] / reference[key]["value_pa"] - 1),
            "position_distance_m": float(
                np.linalg.norm(np.array(candidate[key]["position_m"]) - reference[key]["position_m"])
            ),
        }
        for key in ["tensile", "von_mises"]
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.study / "result.json"
    result = json.loads(result_path.read_text())
    assert result["execution_completed"] and len(result["rows"]) == 3
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=False), "study_result": file_id(result_path)},
    )
    observations, inputs = {}, []
    for row in result["rows"]:
        mesh_path, fields_path = Path(row["mesh"]["path"]), Path(row["fields"]["path"])
        assert file_id(mesh_path) == row["mesh"] and file_id(fields_path) == row["fields"]
        inputs.extend([file_id(mesh_path), file_id(fields_path)])
        with np.load(mesh_path) as mesh, np.load(fields_path) as fields:
            positions = mesh["vertices"][mesh["tetrahedra"]]
            probes = fields["probe_points_m"]
            observations[row["name"]] = {}
            for method in ["direct", "corrected"]:
                current = peaks(
                    fields[f"{method}_probes_stress_pa"], fields[f"{method}_corner_stress_pa"], positions, probes
                )
                for key in ["tensile", "von_mises"]:
                    np.testing.assert_allclose(
                        current[key]["value_pa"], row["methods"][method][f"{key}_max_pa"], rtol=1e-13, atol=1e-9
                    )
                observations[row["name"]][method] = current
    comparisons = {
        "coarse_corrected_fine_direct": compare_peaks(
            observations["coarse"]["corrected"], observations["fine"]["direct"]
        ),
        "coarse_corrected_fine_corrected": compare_peaks(
            observations["coarse"]["corrected"], observations["fine"]["corrected"]
        ),
        "medium_direct_fine_direct": compare_peaks(observations["medium"]["direct"], observations["fine"]["direct"]),
        "medium_corrected_fine_corrected": compare_peaks(
            observations["medium"]["corrected"], observations["fine"]["corrected"]
        ),
    }
    targets = result["targets"]
    direct_changes = result["comparisons"]["medium_direct_fine_direct"]
    independent_reference_gate = all(value <= targets["reference_field_change"] for value in direct_changes.values())
    independent_reference_gate &= all(
        row["relative_value_difference"] <= targets["reference_peak_change"]
        for row in comparisons["medium_direct_fine_direct"].values()
    )
    candidate_errors = result["comparisons"]["coarse_corrected_fine_direct"]
    candidate_error_gate = all(value <= targets["field_relative"] for value in candidate_errors.values())
    candidate_error_gate &= all(
        row["relative_value_difference"] <= targets["peak_relative"]
        for row in comparisons["coarse_corrected_fine_direct"].values()
    )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "retained_field_peak_reductions_passed": True,
            "inputs": inputs,
            "peak_observations": observations,
            "peak_comparisons": comparisons,
            "field_comparisons": result["comparisons"],
            "targets": targets,
            "independent_direct_reference_change_gate_passed": independent_reference_gate,
            "coarse_corrected_vs_fine_direct_error_gate_passed": candidate_error_gate,
            "conditional_prescribed_pressure_stress_gate_passed": independent_reference_gate and candidate_error_gate,
            "corrected_internal_targets_met": result["corrected_internal_targets_met"],
            "pressure_support_prediction_qualified": False,
            "scope": "Read-only independent reductions of all retained element-corner and common near-field tensors, with global sampled tensile/von-Mises peak locations. Original field/peak/reference thresholds are unchanged. The direct-pressure reference change gate is required separately from corrected internal refinement. A conditional prescribed-pressure gate does not qualify predicted contact support/pressure, friction history, arbitrary geometry or a native scene estimator.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
