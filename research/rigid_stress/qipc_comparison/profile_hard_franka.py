"""Profile admitted moving sharp-edge Franka steps separately from primary timing."""

import argparse
import cProfile
from contextlib import redirect_stdout
from pathlib import Path
import pstats
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress import hertz_regions

from .artifacts import file_id, runtime_identity, write_json
from .planar_stage_probes import PlanarStageProbe
from .time_hard_franka import build_workload, compare, observe


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["stress-mesh", "collision-urdf", "conditions", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--batch", type=int, default=4)
    cli.add_argument("--planar-stages", action="store_true")
    args = cli.parse_args()
    assert args.batch > 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "inputs": [file_id(path) for path in [args.stress_mesh, args.collision_urdf, args.conditions]],
            "batch": args.batch,
            "planar_stage_probes": args.planar_stages,
            "scope": "Separate intrusive CPU/kernel profile of the moving max scene, steps100..129 after full0..129 admission. Larger batches tile the condition bank. Same native runtime checks and controller; no numerical producer is replaced. Use the profiler-off paired timer for throughput.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = build_workload(args, args.batch, "max")
    snapshots = []
    for step in range(130):
        workload.step()
        snapshots.append(observe(workload, "max"))
    np.savez_compressed(
        args.output / "audit.npz", **{key: np.stack([item[key] for item in snapshots]) for key in snapshots[0]}
    )
    workload.restart()
    probe = None
    if args.planar_stages:
        probe = PlanarStageProbe(workload.scene.rigid_solver.stress_recovery.links[0].contacts)
        hertz_regions.kernel_hertz_planar = probe
    for step in range(100):
        workload.step()
    qd.sync()
    if probe is not None:
        # The warmup can precede first contact. Keep its errors and count measured queries separately.
        probe.state.counts.fill(0)
    qd.profiler.clear_kernel_profiler_info()
    host = cProfile.Profile()
    begun = perf_counter()
    host.enable()
    try:
        for step in range(30):
            workload.step()
        qd.sync()
    finally:
        host.disable()
    elapsed = perf_counter() - begun
    with (args.output / "kernels.txt").open("w") as stream, redirect_stdout(stream):
        qd.profiler.print_kernel_profiler_info()
    workload.scene.rigid_solver.check_errno()
    compare(snapshots[-1], observe(workload, "max"))
    probe_result = None
    if probe is not None:
        probe_result = probe.report()
        hertz_regions.kernel_hertz_planar = probe.original
    host.dump_stats(str(args.output / "host.pstats"))
    rows = [
        {
            "file": filename,
            "line": line,
            "function": name,
            "primitive_calls": primitive,
            "calls": calls,
            "exclusive_seconds": exclusive,
            "inclusive_seconds": inclusive,
        }
        for (filename, line, name), (primitive, calls, exclusive, inclusive, _) in pstats.Stats(host).stats.items()
    ]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "primary_timing": False,
            "batch": args.batch,
            "steps": 30,
            "planar_stage_probes": probe_result,
            "instrumented_seconds_per_scene_step": elapsed / 30,
            "host_calls": sorted(rows, key=lambda row: row["inclusive_seconds"], reverse=True),
            "kernels": file_id(args.output / "kernels.txt"),
            "host_profile": file_id(args.output / "host.pstats"),
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
