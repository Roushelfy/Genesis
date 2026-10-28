"""Run one fresh primary benchmark process and retain its original failure.

Receipt IO and optional failure export occur outside the benchmark's timers.
No observer, sampler, profiler or extra scientific work runs inside those timers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def _driver_context(error):
    frame = error.__traceback__
    while frame is not None:
        if Path(frame.tb_frame.f_code.co_filename).as_posix().endswith("examples/speed_benchmark/rigid_stress.py"):
            local = frame.tb_frame.f_locals
            if "workload" in local:
                return local
        frame = frame.tb_next
    return {}


def _device():
    torch, gs = sys.modules.get("torch"), sys.modules.get("genesis")
    if torch is None or gs is None:
        return None
    properties = torch.cuda.get_device_properties(gs.device)
    return {"uuid": "GPU-" + str(properties.uuid), "name": properties.name, "total_bytes": properties.total_memory}


def _failure_snapshot(error, output):
    """Export current raw inputs and one failed environment; never replay them."""
    local = _driver_context(error)
    workload = local.get("workload")
    if workload is None:
        return {"available": False, "reason": "Driver failed before workload construction completed."}
    import numpy as np

    from examples.speed_benchmark.rigid_stress_snapshot import contact_selection, read_raw
    from genesis.utils.misc import qd_to_numpy, qd_to_torch

    solver = workload.scene.rigid_solver
    selection = contact_selection(workload)
    records, arrays = read_raw(workload, selection)
    arrays.update(
        selection_environment=selection.environment,
        selection_rank=selection.rank,
        selection_slot=selection.slot,
    )
    entries = []
    for index, entry in enumerate(solver.stress_recovery.links):
        status = qd_to_numpy(entry.contacts.status, transpose=True, copy=True)
        valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)
        bad = np.argwhere(valid & (status != 0))
        state_valid = qd_to_numpy(entry.state.valid, transpose=True, copy=True).reshape(-1)
        invalid = np.flatnonzero(~state_valid)
        environment = int(bad[0, 0]) if len(bad) else int(invalid[0]) if len(invalid) else 0
        prefix = f"stress_{index}_"
        for member in dataclasses.fields(entry.contacts):
            if member.name in ("admission", "active_pairs", "active_count", "hertz_profile"):
                continue
            value = vars(entry.contacts)[member.name]
            arrays[prefix + "contacts_" + member.name] = (
                qd_to_torch(value, transpose=True, copy=False)[environment].detach().cpu().numpy().copy()
            )
        for name in ("force", "rhs", "displacement", "valid", "peak", "rhs_norm_squared", "residual_norm_squared"):
            value = vars(entry.state)[name]
            arrays[prefix + "state_" + name] = (
                qd_to_torch(value, transpose=True, copy=False)[environment].detach().cpu().numpy().copy()
            )
        arrays[prefix + "omega"] = (
            qd_to_torch(entry.omega, transpose=True, copy=False)[environment].detach().cpu().numpy().copy()
        )
        entries.append(
            {
                "link_index": entry.link.idx,
                "environment": environment,
                "selection_rule": "First valid source with nonzero status, else first invalid elastic environment, else0.",
                "nonzero_source_status_count": len(bad),
                "invalid_elastic_environment_count": len(invalid),
                "options": entry.options.model_dump(mode="json"),
            }
        )
    path = output / "exception-state.npz"
    np.savez_compressed(path, **arrays)
    return {
        "available": True,
        "tick": workload.tick,
        "payload": file_id(path),
        "raw_fields": records,
        "entries": entries,
        "scope": "Actual exception-time raw active collider/link/geom inputs and selected native source/elastic arrays. Warmup checks stop at the first error; end-of-timed-interval errors may precede these current buffers. No subsequent scene advance, refit, restoration or acceptance change.",
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
        or os.environ.get("QD_GRAPH") != "0"
    ):
        parser.error("Use Slurm rtx-mid / image1_26 with QD_GRAPH=0.")
    task_path, output = args.task.resolve(), args.output.resolve()
    task = json.loads(task_path.read_text())
    if task.get("module") != "examples.speed_benchmark.rigid_stress" or not isinstance(task.get("args"), list):
        parser.error("Task must bind the native rigid-stress benchmark and a CLI string list.")
    if any(not isinstance(value, str) for value in task["args"]) or any(
        value == "--output" or value.startswith("--output=") for value in task["args"]
    ):
        parser.error("Use CLI strings without --output; this wrapper supplies benchmark.json.")
    if output.is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error("Use an explicit project data output outside the source checkout.")
    output.mkdir(parents=True, exist_ok=False)
    scientific_before = [name for name in ("numpy", "torch", "quadrants", "genesis", "scipy") if name in sys.modules]
    start = {
        "schema": "genesis.automatic-dpr-primary-process.v1",
        "complete": False,
        "passed": False,
        "primary_benchmark_timing": True,
        "task": file_id(task_path),
        "task_definition": task,
        "worker": file_id(__file__),
        "runtime": runtime_identity(),
        "scientific_modules_before_driver": scientific_before,
        "scope": "One fresh native benchmark. Receipt IO and failure export are outside its own timers; no wrapper sampler/profiler or timed scientific observer is added.",
    }
    write_json(output / "start.json", start)
    write_json(output / "result.json", start)
    previous_argv = sys.argv
    driver_error, snapshot, capture_error, namespace = None, None, None, None
    benchmark_path = output / "benchmark.json"
    try:
        sys.argv = [task["module"], *task["args"], "--output", str(benchmark_path)]
        namespace = runpy.run_module(task["module"], run_name="__main__")
    except (Exception, SystemExit) as error:  # noqa: BLE001 - retain the original failure before exiting nonzero
        driver_error = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        try:
            snapshot = _failure_snapshot(error, output)
        except Exception:  # noqa: BLE001 - a failed export must not replace the original driver error
            capture_error = traceback.format_exc()
        traceback.print_exc()
    finally:
        sys.argv = previous_argv
    try:
        actual_device, device_error = _device(), None
    except Exception:  # noqa: BLE001 - preserve the original outcome when post-run device metadata is unavailable
        actual_device, device_error = None, traceback.format_exc()
    checks = {
        "driver_completed": driver_error is None and namespace is not None and benchmark_path.is_file(),
        "task_input_unchanged": file_id(task_path) == start["task"],
        "stdlib_before_driver": not scientific_before,
        "actual_cuda_device_available": actual_device is not None,
    }
    result = {
        **start,
        "complete": True,
        "passed": all(checks.values()),
        "checks": checks,
        "actual_device": actual_device,
        "device_error": device_error,
        "benchmark": file_id(benchmark_path) if benchmark_path.is_file() else None,
        "driver_error": driver_error,
        "failure_snapshot": snapshot,
        "failure_capture_error": capture_error,
        "qualification": "Wrapper execution receipt only. Per-route numerical validity, attempted/valid work, common module inputs and paired hardware must be audited from the original benchmark and failure arrays.",
    }
    write_json(output / "result.json", result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
