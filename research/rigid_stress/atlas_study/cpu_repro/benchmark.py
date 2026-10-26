"""Measure complete changing-load CPU recovery in a quiet single-thread window."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter, process_time

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from .fixtures import cases_manifest, performance_cases
from .loads import force_for_source
from .model import Mother
from .pipeline import Pipeline, PipelineResult, checksum, sampled_maximum
from .response import ExactTopResponse, MaximumBounds
from .selection import build_bank, field_weights, relative_field_error


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    research = package.parents[1]
    files = sorted(package.glob("*.py"))
    files.extend((research / "shell_cpu" / "core.py", research / "shell_cpu" / "geometry.py"))
    return {str(path.relative_to(research)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def environment() -> dict:
    cpu_models = []
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        cpu_models = sorted({line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                             if line.startswith("model name")})
    limits = {}
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            limits[name] = path.read_text().strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpu_models": cpu_models,
        "affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_limits": limits,
        "threadpools": threadpool_info(),
    }


def summarize(samples: list[dict]) -> dict:
    wall = np.array([row["wall_seconds"] for row in samples])
    cpu = np.array([row["process_seconds"] for row in samples])
    return {
        "wall_median_ms": 1000.0 * np.median(wall),
        "wall_min_ms": 1000.0 * np.min(wall),
        "wall_max_ms": 1000.0 * np.max(wall),
        "process_median_ms": 1000.0 * np.median(cpu),
        "samples": samples,
    }


def evaluate(pipeline: Pipeline, cases, batch: int, method: dict, order: int, profile: bool = False):
    if batch == 1:
        return tuple(pipeline.run((case.sources,), order=order, profile=profile, **method) for case in cases)
    source_cases = tuple(cases[i % len(cases)].sources for i in range(batch))
    return (pipeline.run(source_cases, order=order, profile=profile, **method),)


def result_checksum(results: tuple[PipelineResult, ...]) -> tuple[float, float]:
    values = [checksum(result.values) for result in results]
    return tuple(float(np.sum([value[i] for value in values])) for i in range(2))


def explicit_force_cases(mother, cases):
    return tuple(replace(case, sources=tuple(replace(source, force=force_for_source(mother, source).copy())
                                            for source in case.sources)) for case in cases)


def accuracy(pipeline: Pipeline, cases, methods: dict, order: int, reference_order: int) -> dict:
    source_cases = tuple(case.sources for case in cases)
    reference = pipeline.run(source_cases, output="local", order=reference_order)
    same_order = pipeline.run(source_cases, output="local", order=order)
    maximum_reference = sampled_maximum(reference.local_stress)
    maximum_same_order = sampled_maximum(same_order.local_stress)
    weights = field_weights(pipeline.mother)
    result = {}
    loaded = np.array([bool(case.sources) for case in cases])
    for name, method in methods.items():
        value = pipeline.run(source_cases, order=order, **method)
        if value.local_stress is None:
            peaks = value.values
            fields = None
            fields_same_order = None
        else:
            peaks = sampled_maximum(value.local_stress)
            fields = relative_field_error(value.local_stress, reference.local_stress, weights)
            fields_same_order = relative_field_error(value.local_stress, same_order.local_stress, weights)
        peak_error = np.abs(peaks - maximum_reference) / np.maximum(np.abs(maximum_reference), 1e-30)
        peak_error_same_order = np.abs(peaks - maximum_same_order) / np.maximum(np.abs(maximum_same_order), 1e-30)
        result[name] = {
            "reference_order": reference_order,
            "actual_order": order,
            "field_mean_loaded": None if fields is None else float(np.mean(fields[loaded])),
            "field_worst_loaded": None if fields is None else float(np.max(fields[loaded])),
            "peak_worst_loaded": float(np.max(peak_error[loaded])),
            "field_worst_vs_same_order_mother": None if fields_same_order is None else float(np.max(fields_same_order[loaded])),
            "peak_worst_vs_same_order_mother": float(np.max(peak_error_same_order[loaded])),
            "field_errors": fields,
            "peak_errors": peak_error,
            "rhs_groups": value.rhs_groups,
            "contributions": value.contributions,
            "release_output_maximum": float(np.max(np.abs(value.values[..., ~loaded]))),
        }
    return result


def run(args) -> dict:
    start = perf_counter()
    mother = Mother()
    bank = build_bank(mother, count=12)
    pipeline = Pipeline(mother, bank.spaces, bank.select)
    cases = explicit_force_cases(mother, performance_cases())
    manifest = cases_manifest(cases)
    manifest_json = json.dumps(manifest, default=json_default, sort_keys=True, separators=(",", ":"))
    records = {
        "kind": "complete_cpu_reconstruction_benchmark",
        "scope": "Current source integration, load preparation, solve/response, full output or sampled max. CPU module.",
        "output_full": "Both surfaces, every Gauss point, body-frame 3x3 tensors, including symmetric entries.",
        "batch_one_definition": "Arithmetic mean of 16 separate single-environment calls.",
        "cases_per_tile": len(cases),
        "loaded_cases_per_tile": sum(bool(case.sources) for case in cases),
        "sources_per_tile": sum(len(case.sources) for case in cases),
        "force_semantics": "Explicit body-frame force frozen from each immutable case before timing. Current cap geometry and loads rebuilt in every call.",
        "case_manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest(),
        "case_manifest": manifest,
        "mother_dofs": mother.n_dof,
        "geometry_convention": mother.geometry.position_convention,
        "local_stress_rows": mother.S.shape[0],
        "body_tensor_basis_shared_bytes": pipeline.tensor_basis.nbytes,
        "feature_dofs": [space.n_dof for space in bank.spaces],
        "unique_feature_factors": len({id(space.factor) for space in bank.spaces}),
        "feature_factor_payload_bytes": sum(space.factor.payload_bytes for space in bank.spaces),
        "mother_factor_payload_bytes": mother.factor.payload_bytes,
        "basic_setup_seconds": perf_counter() - start,
        "response_inventory": {},
        "orders": {},
    }
    methods = {
        "mother_full": {"strategy": "mother", "output": "full"},
        "continuous_lift_full": {"strategy": "continuous", "output": "full"},
        "continuous_projected_lift_full": {"strategy": "continuous", "output": "full", "rhs_mode": "projected"},
        "nearest_balanced_lift_full": {"strategy": "nearest", "output": "full"},
        "nearest_projected_lift_full": {"strategy": "nearest", "output": "full", "rhs_mode": "projected"},
        "narrow_lift_full": {"strategy": "narrow", "output": "full"},
        "narrow_projected_lift_full": {"strategy": "narrow", "output": "full", "rhs_mode": "projected"},
        "mother_max": {"strategy": "mother", "output": "max"},
        "nearest_projected_lift_max": {"strategy": "nearest", "output": "max", "rhs_mode": "projected"},
    }
    if args.independent_control:
        methods["continuous_independent_full"] = {
            "strategy": "continuous", "output": "full", "recovery_mode": "independent",
        }
    if args.include_response:
        response64 = ExactTopResponse.build(mother, chunk_columns=args.response_chunk)
        response32 = response64.quantized(np.float32)
        bounds = MaximumBounds.build(response64, block_elements=args.screen_block)
        records["response_inventory"] = {
            "shape": list(response64.values.shape), "float64_bytes": response64.nbytes,
            "float32_bytes": response32.nbytes, "float64_build_seconds": response64.build_seconds,
            "screen_block_elements": args.screen_block,
        }
        methods.update({
            "H64_full": {"strategy": "response", "output": "full", "response": response64},
            "H32_full": {"strategy": "response", "output": "full", "response": response32},
            "H64_scan_max": {"strategy": "response", "output": "max", "response": response64},
            "H32_scan_max": {"strategy": "response", "output": "max", "response": response32},
            "H64_screen_max": {"strategy": "screen", "output": "max", "response": bounds},
        })
    print("Setup complete", flush=True)
    records["source_hashes_before"] = source_hashes()
    records["environment_inside_thread_limit"] = environment()
    for order in args.orders:
        order_record = {"batches": {}, "accuracy": {}}
        if args.accuracy:
            order_record["accuracy"] = accuracy(pipeline, cases, methods, order, args.reference_order)
        if args.numeric_only:
            records["orders"][str(order)] = order_record
            continue
        for batch in args.batches:
            if batch != 1 and batch % len(cases):
                raise ValueError("Batched environments must tile the 16 frozen cases")
            timings = {name: [] for name in methods}
            names = tuple(methods)
            for name in names:
                results = evaluate(pipeline, cases, batch, methods[name], order)
                result_checksum(results)
            orders = []
            for repeat in range(args.repetitions):
                ordered = names if repeat % 2 == 0 else names[::-1]
                orders.append(list(ordered))
                for name in ordered:
                    results = None
                    wall_begin, cpu_begin = perf_counter(), process_time()
                    results = evaluate(pipeline, cases, batch, methods[name], order)
                    wall_elapsed = perf_counter() - wall_begin
                    cpu_elapsed = process_time() - cpu_begin
                    normalization = len(cases) if batch == 1 else 1
                    timings[name].append({
                        "wall_seconds": wall_elapsed / normalization,
                        "process_seconds": cpu_elapsed / normalization,
                        "checksum_outside_timing": result_checksum(results),
                        "rhs_groups": sum(result.rhs_groups for result in results),
                        "contributions": sum(result.contributions for result in results),
                    })
                print(f"q{order} B{batch} repeat {repeat + 1}/{args.repetitions}", flush=True)
            stages = {}
            for name in names:
                profiled = evaluate(pipeline, cases, batch, methods[name], order, profile=True)
                divisor = len(cases) if batch == 1 else 1
                keys = tuple(asdict(profiled[0].stages))
                stages[name] = {key: sum(asdict(result.stages)[key] for result in profiled) / divisor
                                for key in keys}
            order_record["batches"][str(batch)] = {
                "timing": {name: summarize(rows) for name, rows in timings.items()},
                "repeat_orders": orders,
                "single_profile_pass_seconds": stages,
                "cases_tiled": 1 if batch == 1 else batch // len(cases),
                "independent_cases": len(cases),
            }
            records["orders"][str(order)] = order_record
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, default=json_default, indent=2) + "\n")
    records["source_hashes_after"] = source_hashes()
    records["source_unchanged"] = records["source_hashes_before"] == records["source_hashes_after"]
    records["peak_process_rss_bytes"] = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    records["peak_process_rss_scope"] = "Process retains mother, feature factors, H64/H32 and screening inventory when enabled."
    records["environment_after"] = environment()
    args.output.write_text(json.dumps(records, default=json_default, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[8])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--include-response", action="store_true")
    parser.add_argument("--independent-control", action="store_true")
    parser.add_argument("--response-chunk", type=int, default=64)
    parser.add_argument("--screen-block", type=int, default=24)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--reference-order", type=int, default=24)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
