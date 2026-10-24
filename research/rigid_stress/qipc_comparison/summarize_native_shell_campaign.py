"""Summarize retained native shell measurements without importing scientific runtimes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def summarize_measurements(campaign):
    tasks = json.loads((campaign / "tasks.json").read_text())
    rows = []
    for group, entries in enumerate(tasks):
        for entry in entries:
            path = campaign / f"group{group}" / entry["name"]
            receipt = path.with_name(path.name + ".returncode.txt")
            row = {"name": entry["name"], "result": str(path), "returncode": None}
            if receipt.exists():
                row["returncode"] = int(receipt.read_text())
            if path.is_file() and row["returncode"] is not None:
                result = json.loads(path.read_text())
                args = result["arguments"]
                repeats = result.get("repeats", [])
                row.update(
                    scope=result["scope"],
                    envs=result["envs"],
                    backend=args["stress_backend"],
                    output_mode=args["output_mode"],
                    source_revision=result["source_revision"],
                    conditions_sha256=result["conditions_sha256"],
                    precision=result["precision"],
                    radius_model=result["radius_model"],
                    pressure_profile=result["pressure_profile"],
                    quadrature=result["quadrature"],
                    gpu=result["gpu"],
                    gpu_uuid=result["gpu_uuid"],
                    setup_seconds=result["setup_seconds"],
                    native_buffer_bytes=result.get("native_buffer_bytes"),
                    torch_peak_allocated_bytes=result["torch_peak_allocated_bytes"],
                    device_free_bytes=result["device_free_bytes"],
                    gpu_total_bytes=result["gpu_total_bytes"],
                    saved_reset_state_bytes=result["saved_reset_state_bytes"],
                    build_timings_including_JIT=result.get("build_timings_including_JIT"),
                    surface_quadrature_grid_seconds_including_JIT=result.get(
                        "surface_quadrature_grid_seconds_including_JIT"
                    ),
                    method_selected=result.get("method_selected"),
                    surface_inverse_selected=result.get("surface_inverse_selected"),
                    shell_recovery_graph_selected=result.get("shell_recovery_graph_selected"),
                    inverse_precision=result["inverse_precision"],
                    packed_surface_loads=result["packed_surface_loads"],
                    factor_scaling=args.get("factor_scaling", False),
                    factor_ordering=args.get("factor_ordering", "minimum_degree"),
                    shell_gauge=args.get("shell_gauge", "translation"),
                    effective_moment_reuse=result.get("effective_contact_moment_reuse"),
                    effective_packed_block_size=result.get("effective_packed_block_sizes"),
                    factor_blocks=result.get("factor_block_entries"),
                )
                unique_buffers = [b for b in result.get("native_buffers", []) if b.get("alias_of") is None]
                row["inverse_response_bytes"] = sum(
                    b["bytes"] for b in unique_buffers if ".inverse." in b["name"] or ".surface_inverse." in b["name"]
                )
                row["factor_bytes"] = sum(b["bytes"] for b in unique_buffers if ".factor." in b["name"])
                if repeats:
                    row.update(
                        median_batch_ms=median(r["seconds"] / r["steps"] * 1000 for r in repeats),
                        median_attempted_env_steps_s=median(r["attempted_transitions_per_second"] for r in repeats),
                        median_admitted_env_steps_s=median(r["transitions_per_second"] for r in repeats),
                        invalid_environment_steps=[r["invalid_environment_steps"] for r in repeats],
                        scatter_overflow_calls=[r["scatter_overflow_calls"] for r in repeats],
                        steps=[r["steps"] for r in repeats],
                        batch_ms=[r["seconds"] / r["steps"] * 1000 for r in repeats],
                    )
                if "stage_wall_ms" in result:
                    row["stage_wall_ms"] = result["stage_wall_ms"]
                    row["moment_stage_wall_ms"] = result["moment_stage_wall_ms"]
            rows.append(row)
    return rows


def summarize_trace(path):
    data = json.loads(path.read_text())
    events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    steps = sum(e.get("name") == "rigid_stress_trace_step" and e.get("cat") == "user_annotation" for e in events)
    assert steps > 0, path
    receipt = path.with_name(path.name.removesuffix(".trace.json"))
    metadata = json.loads(receipt.read_text()).get("trace_metadata") if receipt.is_file() else None
    if metadata is not None:
        assert steps == metadata["steps"], path
    by_category = Counter(e.get("cat", "") for e in events)
    copy_bytes = Counter()
    durations = defaultdict(list)
    for event in events:
        category, name = event.get("cat", ""), event["name"]
        if category == "gpu_memcpy":
            copy_bytes[name] += event["args"]["bytes"]
        if category in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"):
            durations[category, name].append(event["dur"])
    rows = [
        {
            "category": cat,
            "name": name,
            "calls": len(times),
            "calls_per_step": len(times) / steps,
            "inclusive_us_per_step": sum(times) / steps,
        }
        for (cat, name), times in durations.items()
    ]
    rows.sort(key=lambda r: -r["inclusive_us_per_step"])
    return {
        "trace": str(path),
        "steps": steps,
        "trace_metadata": metadata,
        "categories": dict(by_category),
        "copy_bytes_per_step": {name: count / steps for name, count in copy_bytes.items()},
        "events": rows,
        "note": "Intrusive trace after timing. Host/device events overlap; inclusive durations are not additive or headline throughput.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, nargs="*", default=[])
    cli.add_argument("--trace", type=Path, nargs="*", default=[])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "campaigns": {str(path): summarize_measurements(path) for path in args.campaign},
        "traces": [summarize_trace(path) for path in args.trace],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
