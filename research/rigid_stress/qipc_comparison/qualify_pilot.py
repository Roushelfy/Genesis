"""Measure actual solve-tolerance uncertainty from complete saved pilot fields and force histories."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.baseline / "rest.npz") as data:
        volumes = data["volumes_m3"]
    rows = []
    for path in sorted((args.baseline / "frames").glob("step*.npz")):
        candidate = args.candidate / "frames" / path.name
        with np.load(path) as reference, np.load(candidate) as actual:
            np.testing.assert_equal(actual["step"], reference["step"])
            expected, observed = reference["cauchy_pa"], actual["cauchy_pa"]
            reference_norm = np.sqrt(np.einsum("t,tij,tij->", volumes, expected, expected) / volumes.sum())
            error_norm = np.sqrt(
                np.einsum("t,tij,tij->", volumes, expected - observed, expected - observed) / volumes.sum()
            )
            rows.append(
                {
                    "step": reference["step"].item(),
                    "time_s": reference["time_s"].item(),
                    "reference_volume_weighted_tensor_rms_pa": reference_norm.item(),
                    "absolute_volume_weighted_tensor_rms_error_pa": error_norm.item(),
                    "relative_volume_weighted_tensor_error": (error_norm / reference_norm).item()
                    if reference_norm > 1
                    else None,
                    "relative_error_floor_pa": 1.0,
                    "maximum_position_difference_m": np.linalg.norm(
                        actual["positions_m"] - reference["positions_m"], axis=1
                    )
                    .max()
                    .item(),
                    "arm_coordinate_maximum_difference_rad": np.abs(actual["q"][:7] - reference["q"][:7]).max().item(),
                    "finger_coordinate_maximum_difference_m": np.abs(actual["q"][7:] - reference["q"][7:]).max().item(),
                }
            )
    reference_rows = [json.loads(row) for row in (args.baseline / "steps.jsonl").read_text().splitlines()]
    candidate_rows = [json.loads(row) for row in (args.candidate / "steps.jsonl").read_text().splitlines()]
    assert len(reference_rows) == len(candidate_rows)
    reference_force = np.array([row["contact_resultant_n"] for row in reference_rows])
    candidate_force = np.array([row["contact_resultant_n"] for row in candidate_rows])
    scale = np.linalg.norm(reference_force, axis=1)
    difference = np.linalg.norm(candidate_force - reference_force, axis=1)
    loaded = scale > 1e-8
    result = {
        "kind": "bounded-pilot-solve-tolerance-uncertainty",
        "identity": runtime_identity(),
        "baseline": file_id(args.baseline / "result.json"),
        "candidate": file_id(args.candidate / "result.json"),
        "comparison_scope": "same P1 domain, common observation times, world elastic Cauchy; a pilot is not a complete representative-grasp qualification",
        "fields": rows,
        "maximum_reaction_difference_n": difference.max().item(),
        "maximum_relative_reaction_difference": (difference[loaded] / scale[loaded]).max().item()
        if loaded.any()
        else None,
        "reaction_relative_floor_n": 1e-8,
        "baseline_maximum_local_force_residual_relative_contact": max(
            row["force_residual_relative_contact"] or 0 for row in reference_rows
        ),
        "candidate_maximum_local_force_residual_relative_contact": max(
            row["force_residual_relative_contact"] or 0 for row in candidate_rows
        ),
        "maximum_raw_vm_peak_relative_difference": max(
            abs(a["raw_vm_peak_pa"] - b["raw_vm_peak_pa"]) / max(abs(a["raw_vm_peak_pa"]), 1)
            for a, b in zip(reference_rows, candidate_rows)
        ),
        "complete_grasp_reference_qualified": False,
    }
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
