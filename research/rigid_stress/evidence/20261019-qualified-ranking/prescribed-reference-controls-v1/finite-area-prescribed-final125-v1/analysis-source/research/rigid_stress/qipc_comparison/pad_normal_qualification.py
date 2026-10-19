"""Evaluate predeclared normal-reference change gates and conditional error ranks."""

import argparse
import csv
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


FIELDS = (
    "tangent_relative_l2",
    "membrane_relative_l2",
    "bending_relative_l2",
    "outer_tangent_relative_l2",
    "inner_tangent_relative_l2",
)
PEAKS = ("raw_mises_relative_change", "raw_tension_relative_change")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def read(path):
        full = args.data_root / path
        receipts.append(file_id(full))
        return json.loads(full.read_text())

    def comparisons(path):
        result = read(path)
        assert result["physical_volume_weighting"]
        return {case["load_n"]: case["adjacent_comparisons"][0] for case in result["cases"]}

    reference = read("compliant-pad-joint-reference-v2/group0/reference/result.json")
    assert len(reference["cases"]) == 6
    assert {(c["load_n"], c["quadrature_order"]) for c in reference["cases"]} == {
        (load, order) for load in (0.1, 0.3, 1.0) for order in (6, 10)
    }
    checks = {"reference_equations": all(c["pad_equations_admitted"] for c in reference["cases"])}
    old_surface = comparisons("compliant-pad-spatial-comparison-v1/group0/surface500to250/result.json")
    next_surface = comparisons("compliant-pad-surface-final-comparison-v1/group0/surface250to125/result.json")
    old_wall = comparisons("compliant-pad-wall-comparison-v1/group0/wall2to3/result.json")
    wall = comparisons("compliant-pad-joint-reference-v2/group0/wall125/result.json")
    surface = comparisons("compliant-pad-joint-reference-v2/group0/surface3/result.json")
    joint9 = comparisons("compliant-pad-joint-diagonal-v1/group0/joint/result.json")
    joint25 = comparisons("compliant-pad-joint-diagonal-v1/group1/joint/result.json")
    quadrature = comparisons("compliant-pad-joint-reference-v2/group0/quadrature/result.json")
    checks["surface_trend"] = all(
        next_surface[load][key] < old_surface[load][key] for load in joint25 for key in FIELDS
    )
    checks["wall_trend"] = all(wall[load][key] < old_wall[load][key] for load in joint25 for key in FIELDS)
    checks["field_change_target"] = all(
        d[load][key] <= 0.02 for d in (wall, surface, joint25) for load in joint25 for key in FIELDS
    )
    checks["peak_change_target"] = all(
        d[load][key] <= 0.05 for d in (wall, surface, joint25) for load in joint25 for key in PEAKS
    )
    checks["contact_quadrature"] = all(
        quadrature[load][key] <= (0.005 if key in FIELDS else 0.01) for load in joint25 for key in FIELDS + PEAKS
    )
    checks["metric_quadrature"] = all(
        abs(joint25[load][key] - joint9[load][key]) <= 0.005 for load in joint25 for key in FIELDS
    )
    native_audit = read("full-source-pad-normal-fields-v1/group3/adapter/result.json")
    shell_audit = read("compliant-pad-fine-coarse-history-v1/group0/audit/result.json")
    checks["native_histories_and_sources"] = native_audit["passed"]
    checks["shell_history_and_windows"] = all(
        r["completed"] and all(w["mechanical_window_passed"] for w in r["equilibrium_windows"])
        for r in shell_audit["runs"]
    )
    shell_journal = args.data_root / "compliant-pad-fine-coarse-history-v1/group0/cycle/steps.jsonl"
    receipts.append(file_id(shell_journal))
    shell_steps = [json.loads(line) for line in shell_journal.read_text().splitlines()]
    checks["shell_all_steps"] = len(shell_steps) == 1000 and all(
        not any(row["solver_failure_flags"])
        and row["equation_residual_relative"] <= 1e-3
        and row["physical_pad"]["normal_law_and_transfer_passed"]
        for row in shell_steps
    )
    rows, rankings, changes = [], [], []
    for level, load in enumerate((0.1, 0.3, 1.0)):
        metrics = {}
        for group, method in enumerate(("dpr", "scr", "shell")):
            metrics[method] = comparisons(
                f"full-source-joint-normal-fields-v1/group{group}/{method}-{level}/result.json"
            )[load]
        for key in FIELDS + PEAKS:
            spatial = max(d[load][key] for d in (wall, surface, joint25))
            integration = abs(joint25[load][key] - joint9[load][key])
            delta = spatial + quadrature[load][key] + integration
            changes.append(
                {
                    "load_n": load,
                    "metric": key,
                    "spatial_change": spatial,
                    "contact_quadrature_change": quadrature[load][key],
                    "metric_quadrature_change": integration,
                    "observed_resolution_envelope": delta,
                }
            )
            intervals = {}
            for method in ("dpr", "scr", "shell"):
                value = metrics[method][key]
                lower, upper = max(value - delta, 0) / (1 + delta), (value + delta) / (1 - delta)
                intervals[method] = (lower, upper)
                rows.append(
                    {
                        "load_n": load,
                        "metric": key,
                        "method": method,
                        "relative_error": value,
                        "observed_resolution_envelope": delta,
                        "conditional_lower": lower,
                        "conditional_upper": upper,
                    }
                )
            pairs = []
            for first, second in (("dpr", "scr"), ("dpr", "shell"), ("scr", "shell")):
                relation = "unresolved"
                if intervals[first][1] < intervals[second][0]:
                    relation = first + " lower error"
                elif intervals[second][1] < intervals[first][0]:
                    relation = second + " lower error"
                pairs.append({"first": first, "second": second, "relation": relation})
            rankings.append({"load_n": load, "metric": key, "pairs": pairs})
    with (args.output / "errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "checks": checks,
            "shell_maximum_complete_history_equation_residual": max(
                row["equation_residual_relative"] for row in shell_steps
            ),
            "normal_reference_change_targets_passed": all(checks.values()),
            "receipts": receipts,
            "changes": changes,
            "errors": rows,
            "conditional_rankings": rankings,
            "certified_continuum_error_bound": False,
            "whole_goal_complete": False,
            "scope": "Normal physical-pad benchmark only: explicit2mm full-source native variants and repaired250um Shell against125um/three-wall, contact-order10 linear P2 reference. The envelope uses observed wall/surface/joint changes plus integration effects. Conditional ranges use triangle-inequality error propagation if that observed envelope bounds reference error; they are not a proved continuum bound or statistical confidence interval. Parameter-radius sensitivity remains separate. Sustained-friction reference, matched primary performance/batching and publication figures remain required.",
        },
    )
    assert all(checks.values()), "Normal-reference change qualification failed; actual checks retained"


if __name__ == "__main__":
    main()
