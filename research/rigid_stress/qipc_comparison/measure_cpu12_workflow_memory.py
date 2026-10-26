"""Measure sampled own-process GPU memory for one complete native workflow.

The sampler starts before runpy imports the frozen benchmark and GPU libraries.
All nested rates are intrusive diagnostics. Invalid/rejected counters are kept
as reported; this wrapper assesses execution/memory receipts, not spatial error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import platform
import resource
import runpy
import statistics
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

from .measure_cpu12_memory import ProcessSampler, _pid_mapping


def _file_id(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _standard(value):
    """Serialize existing driver metadata without GPU access or scientific imports."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _standard(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_standard(item) for item in value]
    return {"unserialized_driver_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _write(path, payload):
    path.write_text(json.dumps(_standard(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _option(arguments, name, default=None):
    value = default
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 == len(arguments):
                raise ValueError(f"Task option {name} requires a value.")
            value = arguments[index + 1]
        elif argument.startswith(name + "="):
            value = argument.split("=", 1)[1]
    return value


def _partial_result(error):
    """Retain result/finished rows from a failing driver without changing its state."""
    result, worker, counter_values = {}, None, {}
    frame_trace = error.__traceback__
    while frame_trace is not None:
        frame = frame_trace.tb_frame
        if Path(frame.f_code.co_filename).as_posix().endswith("examples/speed_benchmark/rigid_stress.py"):
            worker = frame.f_code.co_filename
            local = frame.f_locals
            if isinstance(local.get("result"), dict):
                result = dict(local["result"])
                if isinstance(local.get("rows"), list):
                    result["repeats"] = list(local["rows"])
            for key in (
                "steps",
                "elapsed",
                "invalid_environment_steps",
                "scatter_overflow_before",
                "failed_per_environment",
            ):
                if key in local:
                    value = local[key]
                    if key == "failed_per_environment" and value is not None:
                        try:
                            # The benchmark already copied this counter to CPU.
                            # The standard buffer protocol preserves its integers
                            # without importing NumPy or reading a GPU tensor.
                            value = memoryview(value).tolist()
                        except (TypeError, ValueError, NotImplementedError):
                            value = _standard(value)
                    counter_values[key] = _standard(value)
        frame_trace = frame_trace.tb_next
    return result, worker, counter_values


def _polling(memory):
    path = Path(memory["raw_log"]) if memory.get("raw_log") else None
    polls = []
    if path is not None and path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "poll":
                polls.append(row)
    starts = [row["started_seconds"] for row in polls if isinstance(row.get("started_seconds"), (int, float))]
    intervals = [right - left for left, right in pairwise(starts)]
    return {
        "poll_count": len(polls),
        "observed_interval_min_seconds": min(intervals, default=None),
        "observed_interval_median_seconds": statistics.median(intervals) if intervals else None,
        "observed_interval_max_seconds": max(intervals, default=None),
        "matched_gpu_uuids": sorted(memory.get("sampled_peak_bytes_by_gpu", {})),
        "sampling_outside_parent_GIL": True,
        "sampled_peak_is_lower_bound": True,
        "scope": "Separate standard-library process; requested 50 ms includes NVIDIA command latency and may miss brief peaks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Use Slurm rtx-mid with GENESIS_IMAGE_VER=1_26.")
    task_path, output = args.task.expanduser().resolve(), args.output.expanduser().resolve()
    task = json.loads(task_path.read_text())
    if not isinstance(task, dict) or task.get("module") != "examples.speed_benchmark.rigid_stress":
        parser.error("Task must invoke examples.speed_benchmark.rigid_stress.")
    arguments = task.get("args")
    if not isinstance(arguments, list) or any(not isinstance(value, str) for value in arguments):
        parser.error("Task args must be a list of CLI strings.")
    if any(value == "--output" or value.startswith("--output=") for value in arguments):
        parser.error("Remove the task's --output; the wrapper supplies benchmark.json as a FILE.")
    expected_steps = _option(arguments, "--steps")
    if expected_steps is None:
        parser.error("Task must explicitly bind --steps to its fixed workflow horizon (normally600).")
    expected_steps, expected_repetitions = int(expected_steps), int(_option(arguments, "--repetitions", "3"))
    if min(expected_steps, expected_repetitions) < 1:
        parser.error("Task steps and repetitions must be positive.")
    if _option(arguments, "--scope", "physics") in ("profile", "recovery"):
        parser.error("This wrapper measures complete workflows; use physics/policy/policy-rigid scope.")
    if output.is_relative_to(Path(__file__).resolve().parents[3]) or (output.exists() and any(output.iterdir())):
        parser.error("Use a fresh/empty project data directory outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(output / "cache/genesis"))
    benchmark_path, result_path = output / "benchmark.json", output / "result.json"
    driver_argv = [task["module"], *arguments, "--output", str(benchmark_path)]
    scientific_before = [name for name in ("numpy", "torch", "quadrants", "genesis", "scipy") if name in sys.modules]
    mapping = _pid_mapping()
    sampler = ProcessSampler(output, mapping)
    start = {
        "schema": "genesis.cpu12-workflow-memory.v1",
        "task": _file_id(task_path),
        "task_definition": task,
        "driver_argv": driver_argv,
        "driver_output_kind": "file",
        "expected_steps": expected_steps,
        "expected_repetitions": expected_repetitions,
        "primary_timing": False,
        "worker": _file_id(Path(__file__).resolve()),
        "sampler_worker": _file_id(Path(ProcessSampler.start.__code__.co_filename).resolve()),
        "scientific_modules_before_sampler": scientific_before,
        "pid_mapping": mapping,
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "genesis_image_version": os.environ.get("GENESIS_IMAGE_VER"),
    }
    _write(output / "start.json", start)
    _write(result_path, {**start, "complete": False, "passed": False})
    previous_argv = sys.argv
    driver_error, memory_error, namespace, recovered_worker = None, None, None, None
    partial, partial_counters, error_trace = {}, {}, None
    started = time.perf_counter()
    sampler.start()
    try:
        sys.argv = driver_argv
        namespace = runpy.run_module(task["module"], run_name="__main__", alter_sys=True)
    except BaseException as error:  # noqa: BLE001 - Persist receipts, then forward the original driver exception.
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            driver_error, error_trace = error, traceback.format_exc()
            partial, recovered_worker, partial_counters = _partial_result(error)
    finally:
        sys.argv = previous_argv
        try:
            memory = sampler.finish()
        except BaseException as error:  # noqa: BLE001 - Persist receipts, then forward the original sampler exception.
            memory_error = error
            memory = {
                "available": False,
                "raw_log": str(sampler.raw_path),
                "collector_stderr": str(sampler.error_path),
                "error": repr(error),
            }
    benchmark, parse_error, recovered = None, None, False
    if benchmark_path.is_file():
        try:
            benchmark = json.loads(benchmark_path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            parse_error = repr(error)
            (output / "benchmark.unparsed.json").write_bytes(benchmark_path.read_bytes())
    if not isinstance(benchmark, dict):
        recovered = True
        benchmark = _standard(partial)
        benchmark.update(
            wrapper_recovered_partial=True,
            driver_completed=False,
            driver_error=repr(driver_error) if driver_error else parse_error,
            partial_driver_counter_values=partial_counters,
            source_scope="Original exception-frame result/finished rows; incomplete execution is retained explicitly.",
        )
        benchmark.setdefault("repeats", [])
        _write(benchmark_path, benchmark)
    repeats = benchmark.get("repeats", [])
    repeats = repeats if isinstance(repeats, list) else []
    gpu_uuid = benchmark.get("gpu_uuid")
    collector_ready = memory.get("ready")
    matched_gpus = set(memory.get("sampled_peak_bytes_by_gpu", {}))
    checks = {
        "driver_completed": driver_error is None and parse_error is None and not recovered and bool(repeats),
        "complete_fixed_horizon": bool(repeats)
        and all(isinstance(row, dict) and row.get("steps") == expected_steps for row in repeats),
        "repetition_count_matches_task": len(repeats) == expected_repetitions,
        "complete_workflow_scope": benchmark.get("scope") is not None
        and benchmark.get("scope") not in ("profile", "recovery"),
        "own_pid_NVIDIA_available": memory_error is None
        and memory.get("available") is True
        and isinstance(memory.get("sampled_peak_device_bytes"), int)
        and memory["sampled_peak_device_bytes"] > 0,
        "same_gpu": bool(gpu_uuid) and matched_gpus == {gpu_uuid},
        "stdlib_collector_before_GPU_imports": not scientific_before
        and isinstance(collector_ready, dict)
        and collector_ready.get("pid") != os.getpid()
        and collector_ready.get("scientific_modules_loaded") == [],
        "task_input_unchanged": _file_id(task_path) == start["task"],
    }
    worker_path = namespace.get("__file__") if namespace is not None else recovered_worker
    genesis_module = sys.modules.get("genesis")
    prep_keys = (
        "setup_seconds",
        "build_timings_including_JIT",
        "surface_quadrature_grid_seconds_including_JIT",
        "recovery_preparation_seconds_including_JIT",
        "warmup_steps",
        "warmup_quality",
    )
    memory_keys = (
        "native_buffer_bytes",
        "native_logical_buffer_bytes",
        "native_allocation_count",
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "saved_reset_state_bytes",
    )
    invalid = [row.get("invalid_environment_steps") for row in repeats if isinstance(row, dict)]
    result = {
        **start,
        "complete": True,
        "passed": all(checks.values()),
        "checks": checks,
        "driver_error": repr(driver_error) if driver_error else None,
        "driver_traceback": error_trace,
        "memory_error": repr(memory_error) if memory_error else None,
        "benchmark_parse_error": parse_error,
        "benchmark": _file_id(benchmark_path),
        "benchmark_recovered_from_exception": recovered,
        "actual_worker": _file_id(Path(worker_path)) if worker_path and Path(worker_path).is_file() else None,
        "native_package": genesis_module.__file__ if genesis_module is not None else None,
        "source_revision": benchmark.get("source_revision"),
        "source_sha256": benchmark.get("source_sha256"),
        "benchmark_arguments": benchmark.get("arguments"),
        "memory": memory,
        "polling": _polling(memory),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "preparation": {key: benchmark[key] for key in prep_keys if key in benchmark},
        "native_memory": {key: benchmark[key] for key in memory_keys if key in benchmark},
        "native_buffer_inventory_available": isinstance(benchmark.get("native_buffers"), list),
        "native_buffer_inventory": "benchmark.json: native_buffers and stress_configurations",
        "repeat_receipts": repeats,
        "reported_invalid_environment_steps": invalid,
        "total_reported_invalid_environment_steps": sum(invalid)
        if invalid and all(isinstance(value, int) for value in invalid)
        else None,
        "partial_driver_counter_values": partial_counters,
        "invalid_counter_policy": "Retain invalid/rejected counters exactly; zero invalid work is not a memory-receipt gate.",
        "primary_timing": False,
        "spatial_qualification_established": False,
        "whole_goal_complete": False,
        "seconds_including_import_build_rollout_and_sampling": time.perf_counter() - started,
        "passed_semantics": "Driver completion, task-bound horizon/repetitions, own-PID same-GPU sampled memory and import boundary.",
        "scope": "Isolated complete workflow process, sampled before GPU imports through setup/JIT/warmup/rollout. All nested rates are intrusive diagnostics.",
        "peak_scope": "Sampled NVIDIA process peak is a lower bound at requested 50 ms plus command latency; the collector runs outside the parent GIL.",
    }
    _write(result_path, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": result["passed"],
                "checks": checks,
                "sampled_peak_device_bytes": memory.get("sampled_peak_device_bytes"),
                "primary_timing": False,
            }
        )
    )
    if driver_error is not None:
        raise driver_error
    if memory_error is not None:
        raise memory_error
    assert result["passed"], "Workflow memory receipt checks failed; benchmark and sampler records are retained."


if __name__ == "__main__":
    main()
