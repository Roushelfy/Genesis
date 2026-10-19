"""Compare independent memory polling with the captured thread under an actual GIL hold."""

import argparse
import ctypes
import os
from pathlib import Path
import runpy
from time import perf_counter, sleep

from ..shell_comparison.memory import ProcessMemory
from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--legacy-sampler", type=Path, required=True)
    cli.add_argument("--legacy-sha256", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    legacy_id = file_id(args.legacy_sampler)
    assert legacy_id["sha256"] == args.legacy_sha256
    legacy_type = runpy.run_path(str(args.legacy_sampler))["ProcessMemory"]
    checks = {}
    disabled = ProcessMemory(False)
    disabled.start()
    inactive = disabled.finish()
    checks["disabled_has_no_child_or_samples"] = disabled.process is None and inactive["samples"] == [] and not inactive["enabled"]
    current, old = ProcessMemory(True), legacy_type(True)
    current.start()
    old.start()
    # Import/initialization are deliberately inside the sampled window, as in
    # real profiles. The controlled GIL hold occurs after both see the GPU.
    import torch

    allocation = torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    deadline = perf_counter() + 10
    while len(old.samples) < 2 and perf_counter() < deadline:
        sleep(0.05)
    assert len(old.samples) >= 2, "Legacy sampler did not obtain baseline GPU observations"
    old_before = old.samples[-1]["seconds"]
    hold_started = perf_counter() - current.started
    ctypes.PyDLL(None).sleep(3)
    hold_finished = perf_counter() - current.started
    sleep(0.5)
    old_result = old.finish()
    current_result = current.finish()
    old_after = next(s["seconds"] for s in old_result["samples"] if s["seconds"] > old_before)
    hold_samples = [s for s in current_result["samples"] if hold_started + 0.1 < s["seconds"] < hold_finished - 0.1]
    hold_polls = [p for p in current_result["polls"] if hold_started <= p["started_seconds"] <= hold_finished]
    gaps = [b["seconds"] - a["seconds"] for a, b in zip(hold_samples[:-1], hold_samples[1:])]
    checks["actual_GIL_hold_at_least_2p9s"] = hold_finished - hold_started >= 2.9
    checks["legacy_thread_gap_at_least_2p5s"] = old_after - old_before >= 2.5
    checks["independent_samples_during_GIL_hold"] = len(hold_samples) >= 10
    checks["independent_query_receipts_during_GIL_hold"] = len(hold_polls) >= 10
    checks["independent_hold_sample_gap_below_1s"] = bool(gaps) and max(gaps) < 1.0
    checks["owned_separate_process"] = current_result["collector_pid"] != os.getpid() == current_result["target_pid"]
    checks["collector_imports_no_scientific_runtime"] = current_result["readiness"]["scientific_modules_loaded"] == []
    checks["collector_stopped_normally"] = current.process.poll() == current_result["collector_returncode"] == 0
    checks["same_parent_GPU_memory"] = {s["gpu_uuid"] for s in current_result["samples"]} == {
        s["gpu_uuid"] for s in old_result["samples"]
    } and all(s["process_mib"] > 0 for s in current_result["samples"])
    # A failed query must reject the profile, even if the collector was ready.
    failing_bin = args.output / "failing-bin"
    failing_bin.mkdir()
    executable = failing_bin / "nvidia-smi"
    executable.write_text("#!/bin/sh\nexit 17\n")
    executable.chmod(0o700)
    previous_path = os.environ["PATH"]
    error, rejected = None, ProcessMemory(True)
    try:
        os.environ["PATH"] = str(failing_bin) + os.pathsep + previous_path
        rejected.start()
        rejected.process.wait(timeout=10)
        rejected.finish()
    except RuntimeError as exc:
        error = str(exc)
    finally:
        os.environ["PATH"] = previous_path
    checks["failed_NVIDIA_query_rejected"] = error is not None and "17" in error
    checks["failed_collector_reaped"] = rejected.process.poll() is not None
    checks["process_environment_restored"] = os.environ["PATH"] == previous_path
    # Keep the allocation alive through the final query control.
    checks["GPU_allocation_remains_intact"] = allocation.numel() == 1024 * 1024 and int(allocation.sum()) == 0
    write_json(args.output / "result.json", {
        "identity": runtime_identity(), "passed": all(checks.values()), "checks": checks,
        "legacy_sampler": legacy_id, "legacy_memory": old_result, "independent_memory": current_result,
        "hold_started_seconds": hold_started, "hold_finished_seconds": hold_finished,
        "legacy_hold_gap_seconds": old_after - old_before,
        "independent_hold_sample_count": len(hold_samples),
        "independent_hold_maximum_sample_gap_seconds": max(gaps, default=None),
        "query_failure_error": error,
        "scope": "Same actual GPU parent, captured original thread and independent stdlib collector, with a3s ctypes.PyDLL C call retaining the parent GIL. This tests polling coverage and explicit failure/child cleanup, not a continuous memory-peak or capacity bound. PATH mutation is local to the negative control and restored.",
    })
    assert all(checks.values()), "Independent memory collector failed the actual GIL/failure controls"


if __name__ == "__main__":
    main()
