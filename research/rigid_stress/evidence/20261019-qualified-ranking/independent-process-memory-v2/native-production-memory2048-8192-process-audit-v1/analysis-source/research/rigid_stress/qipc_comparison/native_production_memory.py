"""Measure total process GPU memory in a separate complete native benchmark run."""

import argparse
import json
import os
import resource
import runpy
import sys
from pathlib import Path

from ..shell_comparison.memory import ProcessMemory
from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    task = json.loads(args.task.read_text())
    assert task["module"] == "examples.speed_benchmark.rigid_stress"
    assert "--output" not in task["args"]
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark = args.output / "benchmark.json"
    argv = [task["module"], *task["args"], "--output", str(benchmark)]
    write_json(args.output / "start.json", {"task": file_id(args.task), "driver_argv": argv})
    sampler = ProcessMemory(True)
    previous_argv = sys.argv
    driver_error, memory_error, memory = None, None, None
    sampler.start()
    try:
        sys.argv = argv
        runpy.run_module(task["module"], run_name="__main__")
    except BaseException as exc:
        driver_error = exc
    finally:
        sys.argv = previous_argv
        try:
            memory = sampler.finish()
        except Exception as exc:
            memory_error = exc
    result = json.loads(benchmark.read_text()) if benchmark.exists() else {}
    repeats = result.get("repeats", [])
    checks = {
        "driver_completed": driver_error is None and bool(repeats),
        "complete_fixed_horizon": bool(repeats) and all(row["steps"] == 1200 for row in repeats),
        "zero_runtime_invalid_counters": bool(repeats)
        and all(row["invalid_environment_steps"] == 0 for row in repeats),
        "memory_sampled": memory_error is None and bool(memory and memory["samples"]),
        "same_gpu": bool(memory and memory["samples"])
        and all(row["gpu_uuid"] == result.get("gpu_uuid") for row in memory["samples"]),
    }
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": all(checks.values()),
            "checks": checks,
            "primary_timing": False,
            "profile_pid": os.getpid(),
            "memory": memory,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "driver_error": repr(driver_error) if driver_error else None,
            "memory_error": repr(memory_error) if memory_error else None,
            "benchmark": file_id(benchmark) if benchmark.exists() else None,
            "scope": "Fresh process sampled before simulator import, through build/warmup/complete rollout. NVIDIA process memory includes non-Torch allocations. Requested0.1s intervals plus command latency can miss shorter peaks. All nested benchmark rates are intrusive diagnostics, never primary timing. Runtime counters do not replace separate complete motion/output conformance or physical accuracy.",
        },
    )
    if driver_error is not None:
        raise driver_error
    if memory_error is not None:
        raise memory_error
    assert all(checks.values()), "Native process-memory run failed its complete-runtime checks"


if __name__ == "__main__":
    main()
