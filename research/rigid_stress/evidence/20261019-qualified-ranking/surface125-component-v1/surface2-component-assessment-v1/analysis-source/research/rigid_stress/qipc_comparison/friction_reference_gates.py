"""Predeclared complete-field/peak gates shared by friction qualification and its controls."""

import math

from .pad_normal_qualification import FIELDS, PEAKS

SPATIAL_FIELD_TARGET = 0.02
SPATIAL_PEAK_TARGET = 0.05
CONTROL_FIELD_TARGET = 0.005
CONTROL_PEAK_TARGET = 0.01


def tangent_peaks(comparison):
    for key in FIELDS + PEAKS:
        if not math.isfinite(comparison[key]) or comparison[key] < 0:
            raise ValueError(f"Invalid complete-field/peak change: {key}")
    rows = comparison["common_spatial_measures"]["tangent_peaks"]
    if len(rows) != 2 or {row["metric"] for row in rows} != {"mises", "tension"}:
        raise ValueError("Both complete common tangent peak definitions are required")
    for row in rows:
        if not math.isfinite(row["relative_difference"]) or row["relative_difference"] < 0:
            raise ValueError("Invalid common-peak change")
        for side in ("actual", "reference"):
            if not math.isfinite(row[side]["stress_pa"]) or row[side]["stress_pa"] <= 0:
                raise ValueError("Positive finite peaks are required for these nonzero-load cases")
            point = row[side]["common_reference_rest_position_m"]
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError("Common peak location must be a finite material point")
    return {row["metric"]: row for row in rows}


def ordered_fields_match(case, expected):
    """The actual/reference order fixes the denominator and endpoint identity."""
    return len(case["raw"]) == 2 and [row["field"] for row in case["raw"]] == list(expected)


def integration_changes(low, high):
    low_peaks, high_peaks = tangent_peaks(low), tangent_peaks(high)
    fields = {key: abs(low[key] - high[key]) for key in FIELDS}
    peaks = {
        metric + "/" + side: abs(low_peaks[metric][side]["stress_pa"] / high_peaks[metric][side]["stress_pa"] - 1)
        for metric in high_peaks
        for side in ("actual", "reference")
    }
    return {"fields": fields, "common_peaks": peaks}


def assess_pair(low, high, *, spatial):
    """Keep all fields and both raw/common peaks; no fitted target or omitted component."""
    field_target = SPATIAL_FIELD_TARGET if spatial else CONTROL_FIELD_TARGET
    peak_target = SPATIAL_PEAK_TARGET if spatial else CONTROL_PEAK_TARGET
    peaks = tangent_peaks(high)
    changes = integration_changes(low, high)
    values = [high[key] for key in FIELDS + PEAKS] + [row["relative_difference"] for row in peaks.values()]
    checks = {"finite_nonnegative_metrics": all(math.isfinite(value) and value >= 0 for value in values)}
    checks.update({key: 0 <= high[key] <= field_target for key in FIELDS})
    checks.update({key: 0 <= high[key] <= peak_target for key in PEAKS})
    checks.update({"common_" + key: 0 <= row["relative_difference"] <= peak_target for key, row in peaks.items()})
    checks["angular_field_contrasts"] = all(
        math.isfinite(v) and v <= CONTROL_FIELD_TARGET for v in changes["fields"].values()
    )
    checks["angular_peak_values"] = all(
        math.isfinite(v) and v <= CONTROL_PEAK_TARGET for v in changes["common_peaks"].values()
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "field_target": field_target,
        "peak_target": peak_target,
        "angular_changes": changes,
    }


def inspect_surface_pair():
    """Inspect the ready surface component without admitting the full reference."""
    import argparse
    import json
    from pathlib import Path

    from .artifacts import file_id, runtime_identity, write_json

    cli = argparse.ArgumentParser(description=inspect_surface_pair.__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--surface-pair-admission", type=Path, required=True)
    cli.add_argument("--spatial-campaign", default="stvk-canonical125-fields-v2")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, rows, checks = [], [], {}

    def read(path):
        receipts.append(file_id(path))
        return json.loads(Path(path).read_text())

    def require(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    error = None
    try:
        admission = read(args.surface_pair_admission)
        runs = [
            args.data_root / "solid-pad-stvk-spatial250-cache-v1/group0/reference",
            args.data_root / "solid-pad-stvk-canonical125-reference-v1/group0/reference",
        ]
        require("exact_complete_parent_pair", admission["passed"] and len(admission["runs"]) == 2
                and {r["run"] for r in admission["runs"]} == {str(r) for r in runs}
                and all(r["expected_steps"] == 400 and all(r["checks"].values()) for r in admission["runs"]))
        for ident in admission["receipts"]:
            require("receipt/" + ident["path"], file_id(ident["path"]) == ident)
            receipts.append(ident)
        starts = [read(run / "start.json") for run in runs]
        ignored = {"output", "mesh", "cache_constitutive_evaluations", "reuse_contact_velocity"}
        parameters = [{k: v for k, v in s["args"].items() if k not in ignored} for s in starts]
        require("same_physical_parameters", parameters[0] == parameters[1])
        require("original_case", all(
            s["fixed_dynamic_dofs"] == 0 and s["args"]["loads"] == [0.1, 0.1]
            and s["args"]["dt"] == 0.001 and s["args"]["quadrature_order"] == 10
            and s["args"]["friction"] == 0.3 and s["args"]["gravity_x"] == 1
            and s["args"]["elasticity"] == "stvk" and s["args"]["contact_solver"] == "modal"
            and s["args"]["warm_contact"] and s["args"]["constitutive_correction"] == "direct"
            for s in starts))
        for level in range(2):
            folder = args.data_root / args.spatial_campaign / f"group{level}"
            require(f"level{level}/terminal_field_task", (folder / "returncode.txt").read_text().strip() == "0")
            receipts.append(file_id(folder / "returncode.txt"))
            values = {}
            for order in (9, 25):
                path = folder / f"surface2-level{level}-q{order}/result.json"
                result = read(path)
                require(f"level{level}/q{order}/sampling", result["surface_quadrature_points_per_face"] == order
                        and result["depth_quadrature_points"] == 4 and result["physical_volume_weighting"]
                        and result["common_peaks_evaluated"] and result["shell_indices"] == [])
                require(f"level{level}/q{order}/one_complete_pair", len(result["cases"]) == 1)
                case = result["cases"][0]
                expected = [file_id(run / f"frames/level{level}.npz") for run in runs]
                require(f"level{level}/q{order}/ordered_own_frames", ordered_fields_match(case, expected))
                for raw, start, ident in zip(case["raw"], starts, expected):
                    require("mesh/" + ident["path"], raw["mesh"] == start["mesh"]
                            and file_id(raw["mesh"]["path"]) == raw["mesh"])
                    receipts.extend((ident, raw["mesh"]))
                require(f"level{level}/q{order}/actual_comparison", case["load_n"] == 0.1
                        and len(case["adjacent_comparisons"]) == 1)
                value = case["adjacent_comparisons"][0]
                require(f"level{level}/q{order}/volume_and_order", value["coarse_index"] == 0
                        and value["fine_index"] == 1 and value["integration_measure_unit"] == "m3"
                        and math.isfinite(value["integration_measure"]) and value["integration_measure"] > 0)
                tangent_peaks(value)
                values[order] = value
            assessment = assess_pair(values[9], values[25], spatial=True)
            checks.update({f"level{level}/surface_component/{k}": v for k, v in assessment["checks"].items()})
            rows.append({"level": level, "assessment": assessment, "q9": values[9], "q25": values[25]})
    except (OSError, KeyError, ValueError, AssertionError) as failure:
        checks["complete_component_inputs"] = False
        error = str(failure)
    passed = bool(checks) and all(checks.values()) and len(rows) == 2
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False),
        "passed": passed, "surface_pair_passed": passed,
        "checks": checks, "error": error, "rows": rows, "receipts": receipts,
        "fine_reference_qualified": False, "accepted_reference_frames": [],
        "whole_goal_complete": False,
        "scope": "Only the admitted250-to125um two-wall surface component at the two fixed normal/friction endpoints. Original2% field/5% raw-and-common peak and angular targets apply. Coarse refinement trend, wall/joint, time/contact/depth controls and the complete seven-parent qualification remain separate; this component produces no accepted fine-reference frames or method ranks.",
    })
    if not passed:
        raise SystemExit("Surface-refinement component failed; full reference qualification remains unproven")


if __name__ == "__main__":
    inspect_surface_pair()
