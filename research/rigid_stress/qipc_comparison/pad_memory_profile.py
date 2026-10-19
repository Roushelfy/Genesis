"""Sample whole-process GPU memory in a separate, mechanically admitted rollout."""

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
    assert task["module"] in (
        "research.rigid_stress.qipc_comparison.plate_native_pad_batch",
        "research.rigid_stress.qipc_comparison.plate_shell_pad_batch",
    )
    assert "--output" not in task["args"]
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.output / "run"
    argv = [task["module"], *task["args"], "--intrusive-memory-profile", "--output", str(run)]
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(), "task": file_id(args.task), "driver_argv": argv},
    )
    sampler = ProcessMemory(True)
    sampler.start()
    previous_argv = sys.argv
    driver_error, memory_error, memory = None, None, None
    try:
        sys.argv = argv
        # Same process/PID, with sampling active before importing either simulator.
        runpy.run_module(task["module"], run_name="__main__")
    except BaseException as exc:
        driver_error = exc
    finally:
        sys.argv = previous_argv
        try:
            memory = sampler.finish()
        except Exception as exc:
            memory_error = exc
    receipts = [file_id(run / name) for name in ("start.json", "result.json", "cost.json") if (run / name).exists()]
    result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
    cost = json.loads((run / "cost.json").read_text()) if (run / "cost.json").exists() else {}
    checks = {
        "driver_completed": driver_error is None and result.get("completed", False),
        "own_history_admitted": result.get("mechanically_admitted", False),
        "memory_sampled": memory_error is None and bool(memory and memory["samples"]),
        "timing_explicitly_intrusive": cost.get("intrusive_memory_profile", False)
        and cost.get("primary_timing") is False,
    }
    write_json(
        args.output / "result.json",
        {
            "passed": all(checks.values()),
            "checks": checks,
            "primary_timing": False,
            "memory": memory,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "driver_error": repr(driver_error) if driver_error else None,
            "memory_error": repr(memory_error) if memory_error else None,
            "receipts": receipts,
            "scope": "Separate fresh-process build/warm/complete normal-load rollout. NVIDIA process memory includes non-PyTorch allocations; the 0.1s requested sampling interval plus command latency can miss shorter peaks. CPU max RSS is the Linux process high-water mark. All timing from this intrusive run is diagnostic. Its own complete mechanical history must pass; this receipt cannot admit another primary timing run. Run audit_pad_batch separately on run/ for same-mesh field and source conformance.",
        },
    )
    if driver_error is not None:
        raise driver_error
    if memory_error is not None:
        raise memory_error
    assert all(checks.values()), "Memory profile requires its own admitted complete trajectory"


if __name__ == "__main__":
    main()
