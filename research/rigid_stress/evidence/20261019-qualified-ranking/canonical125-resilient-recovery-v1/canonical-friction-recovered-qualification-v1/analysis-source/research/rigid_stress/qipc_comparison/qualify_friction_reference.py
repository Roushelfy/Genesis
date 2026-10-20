"""Qualify the fixed nonlinear pad reference against its complete declared refinement matrix."""

import argparse
import csv
import json
import math
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .friction_reference_gates import (
    CONTROL_FIELD_TARGET,
    CONTROL_PEAK_TARGET,
    assess_pair,
    integration_changes,
    ordered_fields_match,
    tangent_peaks,
)
from .pad_normal_qualification import FIELDS, PEAKS
from .solid_pad_resume_admission import audit_resume_lineage

PARENTS = {
    "surface500": ("solid-pad-stvk-spatial500-v2/group0/reference", 400, 0.001, 10),
    "surface250": ("solid-pad-stvk-spatial250-cache-v1/group0/reference", 400, 0.001, 10),
    "wall250": ("solid-pad-stvk-wall250x3-cache-v1/group0/reference", 400, 0.001, 10),
    "surface125": ("solid-pad-stvk-canonical125-reference-v1/group0/reference", 400, 0.001, 10),
    "final125": ("solid-pad-stvk-canonical125-reference-v1/group1/reference", 400, 0.001, 10),
    "contact125": ("solid-pad-stvk-canonical125-time-contact-v1/group0/reference", 400, 0.001, 6),
    "halfdt125": ("solid-pad-stvk-canonical125-time-contact-v1/group1/reference", 800, 0.0005, 10),
}
SPATIAL = {
    "surface2": (0, "surface250", "surface125"),
    "wall125": (2, "surface125", "final125"),
    "surface3": (4, "wall250", "final125"),
    "joint": (6, "surface250", "final125"),
}
CONTROLS = {"contact-order": (0, "contact125"), "half-dt": (2, "halfdt125")}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--parent-admission", type=Path, required=True)
    cli.add_argument("--spatial-campaign", default="stvk-canonical125-fields-v2")
    cli.add_argument("--control-campaign", default="stvk-canonical125-time-contact-fields-v2")
    cli.add_argument("--parent-overrides", type=Path, help="Explicit relative paths for terminal-endpoint continuations")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, receipts, comparisons, envelopes, starts, verified = {}, [], {}, [], {}, {}
    error = None
    parent_specs, overrides, continuations = dict(PARENTS), {}, {}

    def verify(expected):
        path = expected["path"]
        if path not in verified:
            actual = file_id(path)
            assert actual == expected, f"Changed input: {path}"
            verified[path] = actual
        assert verified[path] == expected, f"Inconsistent input receipts: {path}"

    def read(path):
        receipts.append(file_id(path))
        return json.loads(Path(path).read_text())

    def require(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    try:
        if args.parent_overrides is not None:
            overrides = read(args.parent_overrides)
            require(
                "explicit_relative_continuation_parent_map",
                isinstance(overrides, dict) and bool(overrides) and set(overrides) <= set(PARENTS)
                and all(isinstance(value, str) and value and not Path(value).is_absolute()
                        and ".." not in Path(value).parts for value in overrides.values()),
            )
            for name, relative in overrides.items():
                parent_specs[name] = (relative, *PARENTS[name][1:])
        admission = read(args.parent_admission)
        require(
            "explicit_parent_admission_passed",
            admission["passed"] and all(row["passed"] and all(row["checks"].values()) for row in admission["runs"]),
        )
        admitted = {row["run"]: row for row in admission["runs"]}
        expected_runs = {str(args.data_root / spec[0]) for spec in parent_specs.values()}
        require("seven_distinct_parent_run_paths", len(expected_runs) == len(parent_specs))
        require(
            "all_seven_required_parent_histories",
            set(admitted) == expected_runs and len(admission["runs"]) == len(parent_specs),
        )
        for receipt in admission["receipts"]:
            verify(receipt)
        bound_fields = {row["path"]: row for row in admission["receipts"] if row["path"].endswith(".npz")}
        require("fourteen_independently_admitted_frames", len(bound_fields) == 14)
        for name, (relative, steps, dt, order) in parent_specs.items():
            run = args.data_root / relative
            require(name + "/expected_complete_step_count", admitted[str(run)]["expected_steps"] == steps)
            start = read(run / "start.json")
            parameters = start["args"]
            require(
                name + "/free_objective_physics",
                start["fixed_dynamic_dofs"] == 0
                and parameters["elasticity"] == "stvk"
                and parameters["contact_solver"] == "modal"
                and parameters["warm_contact"]
                and parameters["constitutive_correction"] == "direct",
            )
            require(
                name + "/declared_load_and_controls",
                parameters["loads"] == [0.1, 0.1]
                and parameters["dt"] == dt
                and parameters["quadrature_order"] == order
                and parameters["friction"] == 0.3
                and parameters["gravity_x"] == 1
                and parameters["record_stride"] == 1,
            )
            verify(start["mesh"])
            starts[name] = start
            if name in overrides:
                continuations[name] = audit_resume_lineage(
                    run, args.data_root / PARENTS[name][0], start, read, verify,
                    lambda key, passed, parent=name: require(parent + "/continuation/" + key, passed),
                )
        ignored = {
            "output",
            "mesh",
            "dt",
            "quadrature_order",
            "cache_constitutive_evaluations",
            "reuse_contact_velocity",
        }
        reference_parameters = {k: v for k, v in starts["final125"]["args"].items() if k not in ignored}
        require(
            "same_physical_problem_across_seven_parents",
            all(
                {k: v for k, v in start["args"].items() if k not in ignored} == reference_parameters
                for start in starts.values()
            ),
        )
        repair = read(args.data_root / "modal-physical-force-audit-v1/group0/audit/result.json")
        require(
            "canonical_force_repair_controls",
            repair["passed"] and all(repair["checks"].values()) and len(repair["checks"]) == 28,
        )
        for receipt in repair["receipts"]:
            verify(receipt)

        def comparison(path, first, second, level, order, depth):
            result = read(path)
            assert result["surface_quadrature_points_per_face"] == order and result["depth_quadrature_points"] == depth
            assert (
                result["physical_volume_weighting"]
                and result["common_peaks_evaluated"]
                and result["shell_indices"] == []
            )
            assert len(result["cases"]) == 1
            case = result["cases"][0]
            expected = [
                bound_fields[str(args.data_root / parent_specs[name][0] / "frames" / f"level{level}.npz")]
                for name in (first, second)
            ]
            require(str(path.relative_to(args.data_root)) + "/ordered_own_frames", ordered_fields_match(case, expected))
            assert case["load_n"] == 0.1 and len(case["adjacent_comparisons"]) == 1
            for raw, name in zip(case["raw"], (first, second)):
                assert raw["mesh"] == starts[name]["mesh"]
                verify(raw["mesh"])
                verify(raw["field"])
            value = case["adjacent_comparisons"][0]
            assert value["coarse_index"] == 0 and value["fine_index"] == 1
            assert (
                value["integration_measure_unit"] == "m3"
                and math.isfinite(value["integration_measure"])
                and value["integration_measure"] > 0
            )
            tangent_peaks(value)
            return value

        for level in range(2):
            for role, prefix, first, second in (
                (
                    "coarse-surface",
                    "stvk-spatial250-fields-v1/group{level}/surface500to250-level{level}",
                    "surface500",
                    "surface250",
                ),
                ("coarse-wall", "stvk-wall250-fields-v1/group{level}/wall250-level{level}", "surface250", "wall250"),
            ):
                for order in (9, 25):
                    path = args.data_root / (prefix.format(level=level) + f"-q{order}/result.json")
                    comparisons[role, level, order, 4] = comparison(path, first, second, level, order, 4)
            for role, (group, first, second) in SPATIAL.items():
                for order in (9, 25):
                    path = (
                        args.data_root
                        / args.spatial_campaign
                        / f"group{group + level}"
                        / f"{role}-level{level}-q{order}/result.json"
                    )
                    comparisons[role, level, order, 4] = comparison(path, first, second, level, order, 4)
                pair = assess_pair(comparisons[role, level, 9, 4], comparisons[role, level, 25, 4], spatial=True)
                checks.update({f"{role}/{level}/{key}": value for key, value in pair["checks"].items()})
            for role, (group, first) in CONTROLS.items():
                for order, depth in ((9, 2), (9, 4), (25, 4)):
                    path = (
                        args.data_root
                        / args.control_campaign
                        / f"group{group + level}"
                        / f"{role}-level{level}-q{order}-depth{depth}/result.json"
                    )
                    comparisons[role, level, order, depth] = comparison(path, first, "final125", level, order, depth)
                pair = assess_pair(comparisons[role, level, 9, 4], comparisons[role, level, 25, 4], spatial=False)
                checks.update({f"{role}/{level}/{key}": value for key, value in pair["checks"].items()})
                depth = integration_changes(comparisons[role, level, 9, 2], comparisons[role, level, 9, 4])
                checks[f"{role}/{level}/depth_field_contrasts"] = all(
                    v <= CONTROL_FIELD_TARGET for v in depth["fields"].values()
                )
                checks[f"{role}/{level}/depth_peak_values"] = all(
                    v <= CONTROL_PEAK_TARGET for v in depth["common_peaks"].values()
                )
            for role, old in (("surface2", "coarse-surface"), ("wall125", "coarse-wall")):
                checks[f"{role}/{level}/refinement_trend"] = all(
                    0 <= comparisons[role, level, 25, 4][key] < comparisons[old, level, 25, 4][key] for key in FIELDS
                )

            for metric in (*FIELDS, *PEAKS, "common_mises", "common_tension"):

                def value(role, order=25, depth=4):
                    row = comparisons[role, level, order, depth]
                    return (
                        tangent_peaks(row)[metric.removeprefix("common_")]["relative_difference"]
                        if metric.startswith("common_")
                        else row[metric]
                    )

                spatial = max(value(role) for role in SPATIAL)
                temporal, contact = value("half-dt"), value("contact-order")
                if metric.startswith("common_"):
                    peak = metric.removeprefix("common_")
                    angular = max(
                        integration_changes(comparisons[role, level, 9, 4], comparisons[role, level, 25, 4])[
                            "common_peaks"
                        ][peak + "/" + side]
                        for role in (*SPATIAL, *CONTROLS)
                        for side in ("actual", "reference")
                    )
                    depth = max(
                        integration_changes(comparisons[role, level, 9, 2], comparisons[role, level, 9, 4])[
                            "common_peaks"
                        ][peak + "/" + side]
                        for role in CONTROLS
                        for side in ("actual", "reference")
                    )
                else:
                    angular = max(abs(value(role, 9) - value(role)) for role in (*SPATIAL, *CONTROLS))
                    depth = max(abs(value(role, 9, 2) - value(role, 9, 4)) for role in CONTROLS)
                envelopes.append(
                    {
                        "level": level,
                        "metric": metric,
                        "spatial_change": spatial,
                        "time_change": temporal,
                        "contact_change": contact,
                        "angular_change": angular,
                        "depth_change": depth,
                        "observed_resolution_envelope": spatial + temporal + contact + angular + depth,
                    }
                )
        require("complete_declared_matrix", len(comparisons) == 36 and len(envelopes) == 18)
    except (OSError, KeyError, ValueError, AssertionError) as failure:
        checks["complete_input_contract"] = False
        error = str(failure)
    passed = bool(checks) and all(checks.values())
    if envelopes:
        with (args.output / "reference-envelopes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(envelopes[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(envelopes)
    rows = [
        {"role": role, "level": level, "surface_quadrature": order, "depth_quadrature": depth, "metrics": row}
        for (role, level, order, depth), row in comparisons.items()
    ]
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "fine_reference_qualified": passed,
            "checks": checks,
            "error": error,
            "receipts": receipts,
            "verified_inputs": list(verified.values()),
            "parent_overrides": overrides,
            "continued_parent_lineage": continuations,
            "comparisons": rows,
            "reference_envelopes": envelopes,
            "accepted_reference_frames": [
                str(args.data_root / parent_specs["final125"][0] / "frames" / f"level{level}.npz") for level in range(2)
            ]
            if passed
            else [],
            "certified_continuum_error_bound": False,
            "whole_goal_complete": False,
            "scope": "Declared0.1N normal/sustained-friction StVK compliant-pad reference only. All seven complete physical histories, independently bound endpoint frames, isolated surface/wall trends, joint changes, fine q6/q10 and1/0.5ms controls, q9/q25 and depth2/4 controls must pass unchanged2%/5% spatial and0.5%/1% control targets. Envelopes sum observed changes; they are conditional numerical sensitivity summaries, not statistical intervals or certified continuum bounds. No method errors/ranks, Shell mesh independence or failed every-step native stress are admitted here.",
        },
    )
    if not passed:
        raise SystemExit("Friction reference qualification failed; retained every actual check and input receipt")


if __name__ == "__main__":
    main()
