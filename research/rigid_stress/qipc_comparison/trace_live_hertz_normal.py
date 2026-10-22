"""Observe original native calls and CUDA tasks on a separate normal scene."""

import argparse
from collections import Counter
import cProfile
import json
from pathlib import Path
import pstats
import resource

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.utils.array_class import DataKind
from genesis.utils.misc import qd_to_torch

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_normal import build_fixture, snapshot
from .time_live_hertz_normal import update


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--batches", type=int, nargs="+", default=[1, 1024])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(), "mesh": file_id(args.mesh)})
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    rows = []
    for batch in args.batches:
        fixture = build_fixture(args.mesh, batch=batch, dt=2e-4, substeps=2, output_mode="max", quadrature=None)
        assert fixture.entry is not None and fixture.entry.hertz is not None
        phase = 0.11 * np.arange(batch)
        for step in range(4):
            update(fixture, step, phase)
        qd.sync()
        fixture.scene.rigid_solver.check_errno()
        host = cProfile.Profile()
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=False,
            with_stack=False,
        ) as profiler:
            host.enable()
            try:
                for step in range(4):
                    with torch.profiler.record_function("LiveHertz.scene_step"):
                        update(fixture, step, phase)
                qd.sync()
            finally:
                host.disable()
        fixture.scene.rigid_solver.check_errno()
        final = snapshot(fixture.entry)
        assert final["elastic_valid"].all() and final["step_valid"].all()
        assert (final["input_status"] == 0).all() and (final["recovery_status"] <= 1).all()
        assert np.isfinite(final["vm_pa"]).all() and np.isfinite(final["principal_pa"]).all()
        np.savez_compressed(args.output / f"b{batch}-final.npz", **final)
        trace_path = args.output / f"b{batch}-trace.json"
        profiler.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
        device = [event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
        assert any(event.get("cat") == "kernel" for event in device)
        counts = Counter(event["name"] for event in device)
        device_rows = [
            {
                "name": name,
                "calls": count,
                "sum_duration_us": sum(event["dur"] for event in device if event["name"] == name),
            }
            for name, count in counts.most_common()
        ]
        host_rows = [
            {
                "file": filename,
                "line": line,
                "function": name,
                "primitive_calls": primitive,
                "calls": calls,
                "exclusive_host_seconds": exclusive,
                "inclusive_host_seconds": inclusive,
            }
            for (filename, line, name), (primitive, calls, exclusive, inclusive, _) in pstats.Stats(host).stats.items()
            if "/genesis/" in filename or filename.endswith("time_live_hertz_normal.py")
        ]
        logical_fields, seen = [], set()
        for item in fixture.scene.rigid_solver.stress_recovery.data:
            if item.kind != DataKind.CONFIG and id(item.value) not in seen:
                seen.add(id(item.value))
                view = qd_to_torch(item.value, transpose=True)
                logical_fields.append({"name": item.name, "bytes": view.numel() * view.element_size()})
        rows.append(
            {
                "batch": batch,
                "scene_steps": 4,
                "physical_substeps_per_scene_step": 2,
                "quadrature": fixture.entry.link.stress_options.quadrature,
                "default_quadrature_used": True,
                "trace": file_id(trace_path),
                "category_counts": dict(Counter(event.get("cat", "") for event in events)),
                "device_events": device_rows,
                "host_calls": sorted(host_rows, key=lambda row: row["inclusive_host_seconds"], reverse=True),
                "stress_logical_bytes_by_unique_field": logical_fields,
                "stress_logical_bytes": sum(row["bytes"] for row in logical_fields),
                "host_process_maximum_rss_kib_including_setup_and_profiler": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
        fixture.scene.destroy()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "source_producer_unmodified": True,
            "primary_timing": False,
            "scope": "Separate intrusive cProfile/Torch CPU/CUDA trace of the original actual native normal scene. Python inclusive calls overlap and contain profiler/dispatch work; device tasks can overlap and their durations are not a scene wall time. The memory inventory counts unique exposed stress fields, including immutable geometry and shared operators, and excludes compiler/allocator overhead. No producer is replaced. Required native errno acceptance is retained. Use the uninstrumented paired scene timer for throughput. This control does not qualify friction, curved material or Franka accuracy.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
