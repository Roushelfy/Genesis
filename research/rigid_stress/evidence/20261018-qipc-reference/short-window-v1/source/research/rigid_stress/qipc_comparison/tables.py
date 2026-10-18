"""Derive report tables from measured complete histories, with all qualification limits explicit."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def describe(observations):
    selected = [(time, value) for time, value in observations if value is not None]
    if not selected:
        return {"observations": 0}
    times, values = np.array(selected).T
    maximum = np.argmax(values).item()
    return {
        "observations": len(selected),
        "minimum": values.min().item(),
        "median": np.median(values).item(),
        "p95": np.quantile(values, 0.95).item(),
        "maximum": values[maximum].item(),
        "maximum_time_s": times[maximum].item(),
    }


def common_comparison(path):
    value = json.loads(path.read_text())
    rows = []
    for model in ("dpr", "scr", "shell"):
        for field, key in (
            ("tangent_field", "relative_tensor_error"),
            ("membrane_n_per_m", "relative_error"),
            ("bending_n", "relative_error"),
            ("inner_surface_tangent", "relative_tensor_error"),
            ("outer_surface_tangent", "relative_tensor_error"),
        ):
            samples = []
            for item in value["history"]:
                if not item["all_fields_finite"]:
                    continue
                entry = next(row for row in item["errors"] if row["method"] == model)
                samples.append((item["time_s"], entry[field][key]))
            phases = []
            for start, stop, phase in (
                (0, 0.9, "approach"),
                (0.9, 1.8, "squeeze"),
                (1.8, 3, "lift"),
                (3, 3.6, "hold"),
                (3.6, 4.2, "lateral"),
                (4.2, 4.56, "weak"),
                (4.56, 5.4, "hold-after-weak"),
                (5.4, 6.0001, "release-recontact"),
            ):
                phases.append({"phase": phase, **describe([(time, x) for time, x in samples if start <= time < stop])})
            rows.append(
                {"model": model, "metric": field, "whole_history": describe(samples), "phase_summaries": phases}
            )
    return {"source": file_id(path), "rows": rows, "reference_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    comparison = [common_comparison(path) for path in args.comparisons]
    pairs = json.loads(args.uncertainty.read_text())
    uncertainty = []
    for item in pairs["comparisons"]:
        uncertainty.append(
            {
                "candidate": item["candidate"],
                "config": item["candidate_config"],
                "observations": item["common_times"],
                "tangent_field": describe(
                    [(row["time_s"], row["common_tangent"]["relative_tensor_error"]) for row in item["history"]]
                ),
                "inner_surface": describe(
                    [(row["time_s"], row["inner_surface"]["relative_tensor_error"]) for row in item["history"]]
                ),
                "outer_surface": describe(
                    [(row["time_s"], row["outer_surface"]["relative_tensor_error"]) for row in item["history"]]
                ),
            }
        )
    constitutive = {}
    for key in ("hencky_at_identical_F_vs_recorded_energy", "corotated_linear_at_identical_F_vs_recorded_energy"):
        constitutive[key] = describe(
            [
                (row["time_s"], row[key]["relative_tensor_error"])
                for row in pairs["constitutive_at_identical_deformation"]
            ]
        )
    constitutive["principal_stretch_departure"] = describe(
        [
            (row["time_s"], row["maximum_principal_stretch_departure_from_one"])
            for row in pairs["constitutive_at_identical_deformation"]
        ]
    )
    cost = json.loads(args.cost.read_text())
    lifecycle = json.loads(args.lifecycle.read_text())
    replay = json.loads(args.replay.read_text())
    write_json(
        args.output / "result.json",
        {
            "kind": "new-comparison-derived-report-tables",
            "identity": runtime_identity(),
            "common_comparisons": comparison,
            "reference_uncertainty": uncertainty,
            "constitutive_same_F": constitutive,
            "cost": cost,
            "lifecycle": lifecycle,
            "replay": replay,
            "input_receipts": [file_id(path) for path in (args.cost, args.uncertainty, args.lifecycle, args.replay)],
            "reference_qualified": False,
            "scope": "Descriptive deterministic time summaries, not independent samples or confidence intervals. Near-zero-reference denominators remain undefined. Every reference uncertainty/interface/feedback layer is separate. No fastest-in-accuracy-band or three-way physical accuracy ranking is inferred.",
        },
    )
    lines = [
        "# Derived comparison tables",
        "",
        "All reference-dependent discrepancies below remain diagnostic. Median/P95 summarize time, not independent statistical samples.",
        "",
        "## Full-field and max-only cost",
        "",
        "| Configuration | Scope | env-step/s median [min,max] | Batch ms | Numerical valid rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in cost["configurations"]:
        if row.get("raw_repeats"):
            lines.append(
                f"| {row['configuration']} | {row['scope']} | {row['median_aggregate_env_steps_per_second']:.6g} [{row['minimum_aggregate_env_steps_per_second']:.6g},{row['maximum_aggregate_env_steps_per_second']:.6g}] | {row['median_batch_step_ms']:.6g} | {row['median_numerically_valid_rate']} |"
            )
    lines += [
        "",
        "## Whole-history common discrepancies",
        "",
        "| Comparison | Model | Metric | Defined observations | Median | P95 | Maximum | At time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(comparison):
        for row in item["rows"]:
            stats = row["whole_history"]
            if stats["observations"]:
                lines.append(
                    f"| {index} | {row['model']} | {row['metric']} | {stats['observations']} | {stats['median']:.6g} | {stats['p95']:.6g} | {stats['maximum']:.6g} | {stats['maximum_time_s']:.6g} |"
                )
    lines += [
        "",
        "## Paired reference uncertainty",
        "",
        "| Candidate | Observations | Common tangent median | P95 | Maximum | At time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in uncertainty:
        stats = item["tangent_field"]
        lines.append(
            f"| {Path(item['candidate']['path']).parent.name} | {stats['observations']} | {stats['median']:.6g} | {stats['p95']:.6g} | {stats['maximum']:.6g} | {stats['maximum_time_s']:.6g} |"
        )
    lines += [
        "",
        "Source identities and all phase/inner/outer/membrane/bending details are in result.json. Unmodified versus instrumented sources and failed process receipts remain in the source campaigns.",
        "",
    ]
    (args.output / "tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
