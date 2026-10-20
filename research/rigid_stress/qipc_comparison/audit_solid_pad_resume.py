"""Audit complete resumed coarse dynamics against the actual uninterrupted reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .friction_reference_gates import ordered_fields_match, tangent_peaks
from .pad_normal_qualification import FIELDS, PEAKS
from .plate_solid_pad import step_checks
from .plates import force_history


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--comparison", type=Path, required=True)
    cli.add_argument("--contract", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, receipts, differences = {}, {}, {}
    error = None

    def identify(path):
        if str(path) not in receipts:
            receipts[str(path)] = file_id(path)
        return receipts[str(path)]

    def read(path):
        identify(path)
        return json.loads(Path(path).read_text())

    def verify(receipt):
        assert identify(receipt["path"]) == receipt

    def check(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    try:
        contract = read(args.contract)
        child, original = read(args.run / "result.json"), read(args.parent / "result.json")
        start, parent_start = read(args.run / "start.json"), read(args.parent / "start.json")
        provenance = start["resume_provenance"]
        verify(provenance["receipt"])
        proof = read(Path(provenance["receipt"]["path"]))
        check(
            "terminal_parent_source_is_exact_uninterrupted_control",
            proof["passed"]
            and proof["run"] == str(args.parent)
            and original["completed"]
            and original["mechanical_history_passed"],
        )
        for receipt in proof["receipts"]:
            verify(receipt)
        check(
            "same_physical_solver_arguments",
            {key: value for key, value in start["args"].items() if key not in ("output", "checkpoint_stride")}
            == {key: value for key, value in parent_start["args"].items() if key not in ("output", "checkpoint_stride")},
        )
        check(
            "complete400_step_mechanical_history",
            child["completed"]
            and child["steps"] == contract["control_parent_steps"] == original["steps"]
            and child["mechanical_history_passed"]
            and all(child["history_checks"].values()),
        )
        check(
            "honest_inherited_and_new_step_accounting",
            child["inherited_steps"] == proof["prefix_steps"] == contract["control_inherited_steps"]
            and child["newly_executed_steps"] == contract["control_new_steps"]
            and child["elapsed_seconds_scope"] == "newly executed segment, excluding inherited history and setup",
        )
        check(
            "both_full_equilibrium_windows",
            len(child["windows"]) == 2
            and all(
                window["samples"] >= 51 and window["mechanical_window_passed"] and all(window["checks"].values())
                for window in child["windows"]
            ),
        )
        path = args.run / "steps.jsonl"
        identify(path)
        data = path.read_bytes().splitlines(keepends=True)
        rows = [json.loads(line) for line in data]
        check("complete_sequential400_step_journal", [row["step"] for row in rows] == list(range(1, 401)))
        check(
            "inherited_prefix_byte_identity",
            hashlib.sha256(b"".join(data[: proof["prefix_steps"]])).hexdigest() == proof["prefix_sha256"],
        )
        expected = list(force_history(type("Parameters", (), start["args"])()))
        check(
            "global_load_time_and_level_continue",
            len(rows) == len(expected)
            and all(
                row["level"] == level
                and row["command_force_n"] == command
                and abs(row["time_s"] - (index + 1) * start["args"]["dt"]) <= 1e-12
                for index, (row, (level, _, command, _)) in enumerate(zip(rows, expected))
            ),
        )
        check(
            "all_original_equation_contact_moment_and_boundary_gates",
            all(all(step_checks(row).values()) and row["contact_force_evaluation"] == "full_velocity" for row in rows),
        )
        moment = read(args.run.parent / "moment/result.json")
        check(
            "own_independent_current_momentum_admitted",
            moment["passed"]
            and moment["history_admitted"]
            and moment["run"] == identify(args.run / "result.json")
            and len(moment["frames"]) == 2,
        )
        for frame in moment["frames"]:
            verify(frame["field"])
        check(
            "prefix_frame_copied_without_reencoding",
            identify(args.run / "frames/level0.npz")["sha256"] == identify(args.parent / "frames/level0.npz")["sha256"],
        )
        for receipt in (start["mesh"], parent_start["mesh"]):
            verify(receipt)
        check(
            "same_actual_geometry_and_mass",
            start["mesh"] == parent_start["mesh"] and start["mass_kg"] == parent_start["mass_kg"],
        )
        resumed_frame, original_frame = args.run / "frames/level1.npz", args.parent / "frames/level1.npz"
        identify(resumed_frame)
        identify(original_frame)
        with np.load(resumed_frame) as new, np.load(original_frame) as old:
            for key in ("vertices_m", "tetrahedra", "source_points_m", "projected_area_m2"):
                np.testing.assert_array_equal(new[key], old[key])
                check("same_" + key, True)
            for key, target in (
                ("displacement_m", "full_displacement_relative_target"),
                ("stress_pa", "full_stress_relative_target"),
                ("source_forces_n", "full_point_force_relative_target"),
            ):
                a, b = new[key], old[key]
                assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                scale = np.linalg.norm(b)
                assert scale > 0
                value = float(np.linalg.norm(a - b) / scale)
                differences[key + "_relative_l2"] = value
                check(key + "_complete_array_conformance", value <= contract[target])
            velocity = float(np.abs(new["velocity_m_per_s"] - old["velocity_m_per_s"]).max())
            differences["maximum_nodal_velocity_difference_m_per_s"] = velocity
            check(
                "full_velocity_absolute_conformance", velocity <= contract["maximum_nodal_velocity_difference_m_per_s"]
            )
        result = read(args.comparison)
        check(
            "common_metric_quadrature_and_volume",
            result["surface_quadrature_points_per_face"] == 9
            and result["depth_quadrature_points"] == 4
            and result["physical_volume_weighting"]
            and result["common_peaks_evaluated"],
        )
        check(
            "one_ordered_own_final_field_comparison",
            len(result["cases"]) == 1
            and ordered_fields_match(result["cases"][0], [identify(resumed_frame), identify(original_frame)]),
        )
        case = result["cases"][0]
        for raw in case["raw"]:
            verify(raw["mesh"])
            verify(raw["field"])
            check("comparison_mesh/" + raw["field"]["sha256"], raw["mesh"] == start["mesh"])
        compared = case["adjacent_comparisons"][0]
        for key in FIELDS + PEAKS:
            target = (
                contract["five_common_field_relative_targets"]
                if key in FIELDS
                else contract["raw_and_common_peak_relative_targets"]
            )
            differences[key] = compared[key]
            check("complete_common_or_raw/" + key, 0 <= compared[key] <= target)
        for key, row in tangent_peaks(compared).items():
            differences["common_" + key] = row["relative_difference"]
            check("common_peak/" + key, row["relative_difference"] <= contract["raw_and_common_peak_relative_targets"])
    except (OSError, KeyError, ValueError, AssertionError) as failure:
        error = str(failure)
        checks["complete_resumed_equivalence_contract"] = False
    passed = bool(checks) and all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "error": error,
            "differences": differences,
            "receipts": list(receipts.values()),
            "fine_reference_resumed": False,
            "physical_reference_spatially_qualified": False,
            "whole_goal_complete": False,
            "scope": "Actual400-step coarse uninterrupted reference versus its exact saved200-step prefix plus200 newly integrated steps, with original complete-history, equilibrium, contact, momentum and boundary gates. All body/plate/lagged-force state is restored; only algorithmic caches are rebuilt. Full saved arrays and q9/depth4 common fields/raw/common peaks use targets fixed before execution. This is a continuation control, not fine-reference qualification or whole-trajectory performance timing.",
        },
    )
    if not passed:
        raise SystemExit("Resumed coarse dynamics failed equivalence; no fine continuation admitted")


if __name__ == "__main__":
    main()
