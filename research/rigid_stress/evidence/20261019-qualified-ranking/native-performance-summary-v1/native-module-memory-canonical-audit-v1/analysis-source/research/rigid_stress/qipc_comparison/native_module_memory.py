"""Sample process memory during a separate complete native contact-replay run."""

import argparse
import json
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
    assert task["module"] == "research.rigid_stress.qipc_comparison.native_production_controls"
    assert "--replay" in task["args"] and "--output" not in task["args"]
    args.output.mkdir(parents=True, exist_ok=False)
    driver_output = args.output / "replay"
    argv = [task["module"], *task["args"], "--output", str(driver_output)]
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
    result_path = driver_output / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    replay = result.get("replay") or {}
    repeats = replay.get("repeats", [])
    samples = memory["samples"] if memory else []
    expected_gpu = result.get("gpu_uuid", "").removeprefix("GPU-")
    checks = {
        "driver_completed": driver_error is None and result.get("passed", False),
        "all_original_conformance_checks": bool(result.get("checks")) and all(result["checks"].values()),
        "complete_recording": result.get("envs") == 8 and result.get("steps") == 1200,
        "complete_replay_conformance": replay.get("complete_history_check_passed", False)
        and replay.get("partial_resets_replayed", False),
        "complete_fixed_horizon": bool(repeats) and all(row["updates"] == 1200 for row in repeats),
        "zero_runtime_invalid_counters": bool(repeats)
        and all(row["invalid_environment_updates"] == 0 for row in repeats),
        "all_recorded_buffers_retained": len(replay.get("buffers", [])) == 24,
        "memory_sampled": memory_error is None and bool(samples),
        "same_gpu": bool(expected_gpu and samples)
        and all(row["gpu_uuid"].removeprefix("GPU-") == expected_gpu for row in samples),
    }
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": all(checks.values()),
            "checks": checks,
            "primary_timing": False,
            "memory": memory,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "driver_error": repr(driver_error) if driver_error else None,
            "memory_error": repr(memory_error) if memory_error else None,
            "recording_and_replay": file_id(result_path) if result_path.exists() else None,
            "task": file_id(args.task),
            "scope": "Fresh process sampled before simulator import through build, warmup, recording, conformance and complete resident replay. Includes the1200-frame input history; not production-scene memory or isolated kernel workspace. Requested0.1s intervals plus command latency can miss peaks. Nested timings are intrusive diagnostics, never primary rates or a measured capacity limit.",
        },
    )
    if driver_error is not None:
        raise driver_error
    if memory_error is not None:
        raise memory_error
    assert all(checks.values()), "Native module-memory run failed its complete-runtime checks"


if __name__ == "__main__":
    main()
