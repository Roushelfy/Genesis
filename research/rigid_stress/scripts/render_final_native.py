"""Render closed native measurements without inventing or dropping audit rows."""

import argparse
import hashlib
import json
from pathlib import Path


def number(value, digits=2):
    return "unrecorded" if value is None else f"{value:.{digits}f}"


def table(lines, names, rows):
    lines += ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    lines.append("")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.summary.read_bytes()
    data = json.loads(payload)
    assert data["complete"], "Only complete same-source evidence can become final tables."
    measurements = data["measurements"]
    assert len(measurements) == 40
    lines = [
        "# Native final measurement tables",
        "",
        f"Version: `{data['version']}`. This file is generated from the complete",
        f"summary `{args.summary.name}` (SHA256",
        f"`{hashlib.sha256(payload).hexdigest()}`). Renderer SHA256:",
        f"`{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`.",
        "",
        f"Measured source: `{measurements[0]['source_revision']}`.",
        "Per-case revisions and exact hashes remain in the summary. If a",
        "recovery-version override is recorded, it uses the benchmark's explicit",
        "standalone JIT/capture preparation fix; all engine, controller and",
        "option source hashes match the other scopes. Preparation cost is",
        "recorded separately. Original cold-path timings remain archived.",
        "Every rate uses 900 warmup steps and three repeats of at least 2400",
        "benchmark steps and ten seconds. Live/policy contacts change; recovery",
        "repeats its frozen actual snapshot. Both seeds are included. The",
        "aggregate is total valid transitions divided by total wall time; the",
        "six-repeat spread combines workload and temporal variation. All resets,",
        "contact parameters, controller and required observation/inference work",
        "belong to their stated scopes. Frozen recovery is a separate snapshot",
        "diagnostic. Policy measures inference and rollout, not RL training.",
        "",
        "## Four scopes and memory",
        "",
        "Sampled memory is whole-card peak for each recorded GPU UUID, including",
        "setup/JIT; it is separate from native buffer and Torch inventories.",
        "",
    ]
    table(
        lines,
        (
            "B",
            "Scope",
            "Valid env-step/s",
            "Batch step/s",
            "Six-repeat min–max",
            "Std env-step/s",
            "Peak MiB",
            "Invalid",
        ),
        (
            (
                row["envs"],
                row["scope"],
                number(row["valid_env_steps_per_second"]),
                number(row["batch_steps_per_second"]),
                "–".join(number(value) for value in row["repeat_min_max_valid_rate"]),
                number(row["repeat_std_valid_rate"]),
                number(row["sampled_whole_card_peak_MiB"], 0),
                row["invalid_environment_steps"],
            )
            for row in data["two_seed_aggregates"]
        ),
    )
    if data.get("recovery_version") is not None:
        lines += [
            "## Standalone recovery preparation",
            "",
            "This untimed call compiles/captures the graph actually measured,",
            "after restoring the pre-integration snapshot. Its time includes",
            "snapshot restoration, JIT, synchronization and the errno check;",
            "timed calls are unchanged.",
            "",
        ]
        table(
            lines,
            ("B", "Seed", "Preparation calls", "Preparation seconds including JIT"),
            (
                (
                    row["envs"],
                    row["seed"],
                    row["recovery_preparation_steps"],
                    number(row["recovery_preparation_seconds_including_JIT"], 6),
                )
                for row in measurements
                if row["scope"] == "recovery"
            ),
        )
    lines += [
        "## Matched rigid overhead",
        "",
        "Step-time increase and throughput decrease are different quantities.",
        "",
    ]
    table(
        lines,
        ("B", "Seed", "Scope", "Step time +%", "Throughput −%"),
        (
            (
                row["envs"],
                row["seed"],
                row["scope"],
                number(row["step_time_increase_percent"]),
                number(row["throughput_decrease_percent"]),
            )
            for row in data["same_seed_slowdown"]
        ),
    )
    lines += [
        "## Same-GPU full-field output cost",
        "",
        "Each pair uses the same physical GPU and source, with identical final budgets.",
        "",
    ]
    table(
        lines,
        (
            "B",
            "Seed",
            "Scope",
            "Max env-step/s",
            "Full env-step/s",
            "Full step time +%",
            "Additional native bytes",
            "Max peak MiB",
            "Full peak MiB",
        ),
        (
            (
                row["envs"],
                row["seed"],
                row["scope"],
                number(row["measurements"]["max"]["valid_env_steps_per_second"]),
                number(row["measurements"]["full"]["valid_env_steps_per_second"]),
                number(row["step_time_increase_percent"], 3),
                row["additional_native_bytes"],
                number(row["measurements"]["max"]["sampled_whole_card_peak_MiB"], 0),
                number(row["measurements"]["full"]["sampled_whole_card_peak_MiB"], 0),
            )
            for row in data["matched_output_cost"]
        ),
    )
    lines += [
        "## Every-environment episode outcomes",
        "",
        "These intrusive untimed audits cover 900 warmup and 2400 trajectory",
        "steps in every environment. Outcome totals below subtract warmup.",
        "Started, unfinished and restart-aborted episodes remain visible.",
        "Failure reasons overlap. Weak-grip height loss is a drop proxy, not",
        "a measured tangential slip distance or an additional episode-success",
        "criterion. Exact lift/hold/release definitions remain in every artifact.",
        "",
    ]
    table(
        lines,
        (
            "B",
            "Seed",
            "Scope",
            "Started",
            "Completed",
            "Success",
            "Failed",
            "Unfinished",
            "Aborted",
            "Weak >1cm drop",
            "Numeric steps failed",
        ),
        (
            (
                row["envs"],
                row["seed"],
                row["scope"],
                row["outcomes"]["started"],
                row["outcomes"]["completed"],
                row["outcomes"]["successful"],
                row["outcomes"]["completed"] - row["outcomes"]["successful"],
                row["unfinished_episodes_at_end"],
                row["outcomes"]["aborted_by_restart"],
                row["outcomes"]["weak_grip_height_drop_gt_1cm"],
                row["failed_environment_steps"],
            )
            for row in data["episode_audits"]
        ),
    )
    lines += [
        "## Numerical and work tails",
        "",
        "The following are worst values across environments, not individual",
        "environment GPU latencies. Audit work totals span warmup and trajectory.",
        "Complete balance includes all six gauge rows. Force and moment budgets",
        "remain 1e-8; the full residual budget remains max(1e-11 N, 1e-8 times",
        "the RHS norm). No slow or failed environment is removed.",
        "",
    ]

    def maximum(row, group, key):
        return row[group][key]["p50_p90_p99_max"][-1]

    table(
        lines,
        (
            "B",
            "Seed",
            "Scope",
            "Full residual / budget max",
            "Wrench force rel max",
            "Wrench moment rel max",
            "Newton eval max",
            "Retries max",
            "Corrections total",
            "Factor fallback total",
            "Packed columns max",
            "Face tasks max",
            "Zero RHS %",
        ),
        (
            (
                row["envs"],
                row["seed"],
                row["scope"],
                f"{maximum(row, 'errors', 'full_residual_budget_ratio_max'):.6g}",
                f"{maximum(row, 'errors', 'wrench_force_relative_max'):.6g}",
                f"{maximum(row, 'errors', 'wrench_moment_relative_max'):.6g}",
                number(maximum(row, "work", "pressure_newton_evaluations_max"), 0),
                number(maximum(row, "work", "local_integration_retries"), 0),
                number(row["work"]["inverse_corrections"]["sum"], 0),
                number(row["work"]["factor_fallbacks"]["sum"], 0),
                number(maximum(row, "additional_work", "packed_boundary_columns_max"), 0),
                number(maximum(row, "additional_work", "admitted_face_tasks_max"), 0),
                number(
                    100 * row["exact_zero_rhs_fraction"] if row.get("exact_zero_rhs_fraction") is not None else None, 4
                ),
            )
            for row in data["episode_audits"]
        ),
    )
    lines += [
        "## Frozen-snapshot split profile",
        "",
        "Separate synchronized stage passes are diagnostic; they are not added to",
        "derive live throughput. Pressure subdivision is another separate pass.",
        "",
    ]
    snapshots = [row for row in data["profiles"] if "stage_wall_ms" in row]
    table(
        lines,
        ("Snapshot", "Stage", "Wall ms"),
        (
            (Path(row["path"]).name, key, number(value, 6))
            for row in snapshots
            for key, value in row["stage_wall_ms"].items()
        ),
    )
    lines += [
        "## Changing-contact stream profiles",
        "",
        "Nested CUDA-stream intervals overlap and must not be summed. They include",
        "host preparation/dispatch idle gaps. Intrusive event instrumentation changes",
        "timing; ordinary benchmark rates above remain the throughput evidence.",
        "",
    ]
    table(
        lines,
        ("Profile", "Stage", "Calls", "CUDA interval mean ms", "CUDA p99 ms", "CUDA max ms", "Host submission mean ms"),
        (
            (
                Path(row["path"]).name,
                key,
                value["calls"],
                number(value["cuda_stream_interval_mean_ms"], 4),
                number(value["cuda_stream_interval_p50_p90_p99_max_ms"][2], 4),
                number(value["cuda_stream_interval_p50_p90_p99_max_ms"][3], 4),
                number(value["host_submission_mean_ms"], 4),
            )
            for row in data["profiles"]
            if "summary" in row
            for key, value in row["summary"].items()
        ),
    )
    lines += [
        "All per-environment p50/p90/p99/max work distributions, exact coefficients,",
        "reset counts, cache hit/miss/eviction counts, byte inventories, source hashes",
        "and original paths are preserved in the summary and its hashed artifacts.",
        "",
    ]
    args.output.write_text("\n".join(lines))
    print(
        f"Rendered {len(measurements)} measurements, {len(data['episode_audits'])} episode audits and {len(data['profiles'])} profiles.",
        flush=True,
    )


if __name__ == "__main__":
    main()
