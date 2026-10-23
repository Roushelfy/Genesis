"""Sample fresh-process sharp-contact scene memory separately from primary timing."""

import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import sys

from ..shell_comparison.memory import ProcessMemory
from .artifacts import file_id, runtime_identity, write_json
from .process_memory_receipts import audit_polling


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    task = json.loads(args.task.read_text())
    assert task["module"] == "research.rigid_stress.qipc_comparison.time_hard_franka"
    assert "--output" not in task["args"]
    args.output.mkdir(parents=True, exist_ok=False)
    driver = args.output / "driver"
    argv = [task["module"], *task["args"], "--output", str(driver)]
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
    result_path = driver / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    identity = runtime_identity()
    checks = {
        "driver_completed": driver_error is None and result.get("passed", False),
        "single_mode": len(result.get("rows", [])) == 1,
        "memory_sampled": memory_error is None and bool(memory and memory["samples"]),
    }
    coverage = None
    if checks["single_mode"]:
        row = result["rows"][0]
        checks["matched_case"] = row["batch"] == task["batch"] and row["mode"] == task["mode"]
        checks["complete_audit"] = row["audit_environment_steps"] == 130 * task["batch"]
        checks["one_repeat"] = len(row["seconds_per_scene_step"]) == 1
        if task["mode"] != "rigid":
            checks["all_timed_steps_admitted"] = row["admitted_timed_environment_steps"] == 30 * task["batch"]
    if checks["memory_sampled"]:
        uuids = {sample["gpu_uuid"] for sample in memory["samples"]}
        checks["one_visible_gpu"] = len(uuids) == 1 and all(
            uuid in identity["nvidia_smi_visible_inventory"] for uuid in uuids
        )
        if checks["one_visible_gpu"]:
            coverage = audit_polling(memory, os.getpid(), next(iter(uuids)))
    write_json(
        args.output / "result.json",
        {
            "identity": identity,
            "passed": all(checks.values()),
            "checks": checks,
            "primary_timing": False,
            "profile_pid": os.getpid(),
            "memory": memory,
            "polling_audit": coverage,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "driver_error": repr(driver_error) if driver_error else None,
            "memory_error": repr(memory_error) if memory_error else None,
            "driver_result": file_id(result_path) if result_path.exists() else None,
            "whole_goal_complete": False,
            "scope": "Fresh process from before simulator import through build,130-frame admission audit and one repeated100..129 window. Independent NVIDIA polling includes non-Torch allocations. Requested0.1s intervals plus query latency can miss short peaks; this is a sampled peak, not a capacity bound. Driver rates are intrusive diagnostics. Primary throughput and cross-mode/source output checks are separate.",
        },
    )
    if driver_error is not None:
        raise driver_error
    if memory_error is not None:
        raise memory_error
    assert all(checks.values()), "Sharp-contact memory run failed its driver or polling checks"


if __name__ == "__main__":
    main()
