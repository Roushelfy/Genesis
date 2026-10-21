"""Intrusive CPU/CUDA launch trace of the moving hard-contact input path."""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
from time import perf_counter
from unittest.mock import patch

import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import DataKind, iter_data
from genesis.utils.misc import qd_to_torch

from .artifacts import file_id, runtime_identity, write_json
from .live_hertz_inputs import InputObserver


class TracedInputObserver(InputObserver):
    def __init__(self, solver):
        super().__init__(solver)
        self.is_traced = False

    def phase(self, name, function, *args):
        if self.is_traced:
            with torch.profiler.record_function("HertzInputs." + name):
                function(*args)
        else:
            function(*args)


def memory_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "device_free_bytes": free,
        "device_total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "host_maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--envs", type=int, required=True)
    cli.add_argument("--warmup", type=int, default=480)
    cli.add_argument("--steps", type=int, default=60)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "conditions": file_id(args.conditions),
            "envs": args.envs,
            "warmup": args.warmup,
            "steps": args.steps,
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    memory = {"initialized": memory_snapshot()}
    begun = perf_counter()
    workload = FrankaEgg(
        args.envs,
        level=1,
        stress=True,
        seed=623016,
        varied=True,
        conditions=args.conditions,
        output_mode="max",
        stress_update_mode="every_step",
        fused_rigid_post=False,
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    workload.stress = False
    solver = workload.scene.rigid_solver
    qd.sync()
    memory["scene_built"] = memory_snapshot()
    observer = TracedInputObserver(solver)
    observer.audit_enabled = False
    qd.sync()
    memory["inputs_built"] = memory_snapshot()
    setup_seconds = perf_counter() - begun

    def observe(recovery, i_substep):
        observer.update(recovery, i_substep)

    with patch.object(RigidStressRecovery, "recover", observe), torch.inference_mode():
        for _ in range(args.warmup):
            workload.step()
        qd.sync()
        solver.check_errno()
        memory["warmed"] = memory_snapshot()
        observer.is_traced = True
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=False,
            with_stack=False,
        ) as profiler:
            for _ in range(args.steps):
                with torch.profiler.record_function("HertzInputs.scene_step"):
                    workload.step()
            qd.sync()
        observer.is_traced = False
        solver.check_errno()
    memory["traced"] = memory_snapshot()
    trace_path = args.output / "trace.json"
    profiler.export_chrome_trace(str(trace_path))
    events = profiler.events()
    names = sorted({event.name for event in events if event.name.startswith("HertzInputs.")})
    scopes = [
        {
            "name": name,
            "calls": sum(event.name == name for event in events),
            "inclusive_host_us": sum(event.cpu_time_total for event in events if event.name == name),
            "exclusive_host_us": sum(event.self_cpu_time_total for event in events if event.name == name),
        }
        for name in names
    ]
    trace = json.loads(trace_path.read_text())
    device_events = [
        event for event in trace["traceEvents"] if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    device_names = Counter(event["name"] for event in device_events)
    device_rows = [
        {
            "name": name,
            "calls": count,
            "sum_duration_us": sum(event["dur"] for event in device_events if event["name"] == name),
        }
        for name, count in device_names.most_common()
    ]
    snapshot = observer.snapshot()
    geometry_bytes = {}
    for item in iter_data(observer.cache.info, "curvature", DataKind.CONSTANT):
        view = qd_to_torch(item.value, transpose=True)
        geometry_bytes[item.name] = view.numel() * view.element_size()
    shape_bytes = sum(
        size
        for name, size in geometry_bytes.items()
        if name in ("curvature.face_min", "curvature.face_max", "curvature.face_scale")
    )
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "envs": args.envs,
            "steps": args.steps,
            "warmup": args.warmup,
            "setup_seconds": setup_seconds,
            "curvature_build_seconds_including_first_compile": observer.curvature_build_seconds,
            "memory": memory,
            "new_shared_geometry_bytes": shape_bytes,
            "new_temporal_cache_bytes_including_diagnostics": 0,
            "shared_curvature_logical_bytes_by_field": geometry_bytes,
            "inclusive_and_exclusive_host_scopes": scopes,
            "device_events": device_rows,
            "trace_category_counts": dict(Counter(event.get("cat", "") for event in trace["traceEvents"])),
            "trace": file_id(trace_path),
            "final_valid_environments": snapshot["is_valid"].tolist(),
            "final_region_counts": (snapshot["source_count"] > 0).sum(axis=1).tolist(),
            "scope": "Intrusive CPU/CUDA trace with research audits disabled and one synchronization after the complete window. Host scopes are nested; device task durations may overlap and do not replace inclusive uninstrumented timing. Device free-memory snapshots include other users of the device; torch allocation counters exclude Quadrants allocations. Host maximum RSS includes setup and profiling. New geometry/cache byte counts exclude all existing state. Input-only; no stress update or physical-accuracy qualification.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
