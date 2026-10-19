"""Summarize explicitly instrumented live-pad step and observation costs."""

from .artifacts import write_json


def arguments(cli):
    cli.add_argument(
        "--intrusive-memory-profile",
        action="store_true",
        help="Mark this run's wall times as intrusive diagnostics, never primary timing",
    )
    cli.add_argument(
        "--primary-cost",
        action="store_true",
        help="Time complete levels including admission and I/O; disable phase timers",
    )
    cli.add_argument(
        "--stress-stride",
        type=int,
        default=0,
        help="0 observes endpoints only; positive cadence also observes endpoints",
    )
    cli.add_argument("--stress-output", choices=("full", "max"), default="full")
    cli.add_argument(
        "--cost-warm-levels",
        type=int,
        default=0,
        help="Positive value enables instrumented costs after these complete load levels",
    )


def write_cost(args, rows, completed, admitted=None, primary=None):
    if args.primary_cost:
        primary.write(rows, completed, admitted)
        return
    if not args.cost_warm_levels:
        return
    samples = []
    for level in sorted({row["level"] for row in rows}):
        selected = [row for row in rows if row["level"] == level]
        steps = len(selected)
        physics = sum(row["timing"]["physics_seconds"] for row in selected)
        observation = sum(row.get("stress", {}).get("observation_seconds_excluding_frame_IO", 0) for row in selected)
        observed = [row["stress"] for row in selected if "stress" in row]
        samples.append(
            {
                "level": level,
                "warmup": level < args.cost_warm_levels,
                "steps": steps,
                "physics_seconds": physics,
                "recovery_observation_seconds": observation,
                "segmented_pipeline_seconds": physics + observation,
                "segmented_latency_ms": 1000 * (physics + observation) / steps,
                "segmented_env_steps_per_second": steps / (physics + observation),
                "stress_observations": len(observed),
                "all_stress_observations_valid": all(
                    row.get("native_valid", row.get("valid", False)) for row in observed
                ),
                "field_host_bytes": sum(row["field_host_bytes"] for row in observed),
                "residual_host_bytes": sum(row.get("residual_host_bytes", 0) for row in observed),
                "raw_vm_peak_min_pa": min((row["raw_vm_peak_pa"] for row in observed), default=None),
                "raw_vm_peak_max_pa": max((row["raw_vm_peak_pa"] for row in observed), default=None),
            }
        )
    write_json(
        args.output / "cost.json",
        {
            "completed": completed,
            "mechanically_admitted": admitted,
            "method": getattr(args, "method", "shell"),
            "envs": 1,
            "output_mode": args.stress_output,
            "stress_stride": args.stress_stride,
            "samples": samples,
            "primary_timing": False,
            "scope": "Instrumented live trajectory, with existing mechanical admission after each step. Segmented wall times include actual contact/elastic mechanics, requested recovery and host output/acceptance. Mechanical diagnostics, frame/journal I/O and their extra transfers are outside these segments. This is a preflight phase-cost control, not the final uninstrumented primary wall-time benchmark or physical-accuracy qualification. Warmup levels are excluded from comparisons; use the same run's mechanical audit.",
        },
    )
