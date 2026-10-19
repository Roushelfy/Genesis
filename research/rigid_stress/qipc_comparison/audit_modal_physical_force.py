"""Aggregate the original detachment counterexample and canonical physical-force controls."""

import argparse
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks = [], {}

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    before = read(args.data_root / "modal-physical-force-before-v1/group0/controls/result.json")
    after = read(args.data_root / "modal-physical-force-after-v4/group0/detachment/result.json")
    core = read(args.data_root / "modal-physical-force-after-v4/group0/all-contact-controls/result.json")
    checks["before_detects_physical_mismatch"] = not before["passed"] and any(
        not row["checks"]["physical_friction_law"] for row in before["cases"]
    )
    checks["before_other_physical_controls_pass"] = all(
        all(value for name, value in row["checks"].items() if name != "physical_friction_law")
        for row in before["cases"]
    )
    checks["same12_detachment_cases_after"] = after["passed"] and len(before["cases"]) == len(after["cases"]) == 12
    checks["all24_original_contact_cases"] = core["passed"] and len(core["cases"]) == 24
    root = args.data_root / "solid-pad-stvk-physical-force-coarse-v1/group0"
    original = args.data_root / "solid-pad-stvk-response-reuse-control-v1/group0"
    reference, moment = read(root / "reference/result.json"), read(root / "moment/result.json")
    checks["complete400_step_history"] = (
        reference["completed"] and reference["mechanical_history_passed"] and reference["steps"] == 400
    )
    checks["independent_current_momentum"] = moment["passed"] and moment["history_admitted"]
    current_start, old_start = read(root / "reference/start.json"), read(original / "reference/start.json")
    checks["same_scientific_configuration"] = {
        key: value for key, value in current_start["args"].items() if key != "output"
    } == {key: value for key, value in old_start["args"].items() if key != "output"}
    journal = root / "reference/steps.jsonl"
    receipts.append(file_id(journal))
    steps = [json.loads(line) for line in journal.read_text().splitlines()]
    checks["full_velocity_force_used_every_step"] = len(steps) == 400 and all(
        row["contact_force_evaluation"] == "full_velocity" for row in steps
    )
    for name, target in (
        ("equation_residual_relative", 1e-7),
        ("normal_law_relative_error", 1e-7),
        ("friction_law_relative_error", 1e-7),
        ("point_velocity_prediction_error_m_per_s", 1e-9),
        ("current_moment_balance_relative_error", 1e-7),
    ):
        checks["all_steps/" + name] = all(row[name] <= target for row in steps)
    checks["complete_boundary_every_step"] = all(row["minimum_omitted_gap_m"] >= -1e-12 for row in steps)
    contrasts = []
    keys = (
        "tangent_relative_l2",
        "membrane_relative_l2",
        "bending_relative_l2",
        "outer_tangent_relative_l2",
        "inner_tangent_relative_l2",
        "raw_mises_relative_change",
        "raw_tension_relative_change",
    )
    for level in (0, 1):
        result = read(root / f"equivalence-level{level}/result.json")
        assert result["surface_quadrature_points_per_face"] == 9 and result["depth_quadrature_points"] == 4
        row = result["cases"][0]["adjacent_comparisons"][0]
        for name in keys:
            checks[f"level{level}/" + name] = row[name] <= 1e-8
        for raw in result["cases"][0]["raw"]:
            for name in ("mesh", "field"):
                assert file_id(raw[name]["path"]) == raw[name]
                receipts.append(raw[name])
        contrasts.append({"level": level, **{name: row[name] for name in keys}})
    passed = all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "receipts": receipts,
            "contrasts": contrasts,
            "maximum_before_friction_error": max(row["force_relative_error"] for row in before["cases"]),
            "maximum_after_friction_error": max(row["force_relative_error"] for row in after["cases"]),
            "maximum_trajectory_friction_error": max(row["friction_law_relative_error"] for row in steps),
            "physical_reference_qualified": False,
            "whole_goal_complete": False,
            "scope": "Original manufactured physical-force mismatch,12 corrected detachment cases,24 contact cases, unchanged400-step coarse physical/momentum admission and q9/depth4 old/new fields under the predeclared1e-8 numerical-control target. Fine spatial/time/contact uncertainty and complete friction rankings remain open. No force law, physical source, error normalization or tolerance is changed.",
        },
    )
    if not passed:
        raise SystemExit("Canonical physical-force control aggregation failed")


if __name__ == "__main__":
    main()
