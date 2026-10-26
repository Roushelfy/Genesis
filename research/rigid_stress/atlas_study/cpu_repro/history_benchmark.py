"""Interleave complete-trajectory timings after independent numerical qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter, process_time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .history import BatchHistory, RecoverySystem
from .history_scaled_trajectory import SPEED_DEFINITION, prepare_scaled_trajectory, scaled_current_force
from .history_study import Trajectory
from .model import Mother


def source_hashes() -> dict[str, str]:
    folder = Path(__file__).parent
    paths = list(folder.glob("*.py"))
    paths.append(folder.parents[1] / "shell_cpu" / "core.py")
    return {str(path.relative_to(folder.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def configurations(scope: str) -> tuple[tuple[str, float, bool, bool, bool], ...]:
    basic = (
        ("full", 0.01, True, False, False),
        ("copy_0.01", 0.01, False, False, True),
        ("rotate_0.01", 0.01, True, False, True),
        ("cached_rotate_0.01", 0.01, True, True, True),
    )
    if scope == "recovery":
        return basic + (
            ("copy_0.05", 0.05, False, False, True),
            ("rotate_0.05", 0.05, True, False, True),
            ("cached_rotate_0.05", 0.05, True, True, True),
        )
    return tuple(item for item in basic if item[0] != "copy_0.01")


def run_trace(
    engine: BatchHistory,
    mother: Mother,
    trajectory: Trajectory,
    order: int,
    complete: bool,
    speed: float = 1.0,
) -> dict:
    frames, batch = trajectory.active.shape
    accepted_count = failed_count = 0
    start_wall, start_cpu = perf_counter(), process_time()
    for frame in range(frames):
        if complete:
            forces, active, resets = scaled_current_force(mother, frame, batch, order, speed)
        else:
            forces = trajectory.forces[frame]
            active, resets = trajectory.active[frame], trajectory.resets[frame]
        output = engine.step(forces, active, reset=resets)
        accepted_count += np.count_nonzero(output.accepted_prediction)
        failed_count += len(output.failed_environments)
    elapsed_cpu, elapsed_wall = process_time() - start_cpu, perf_counter() - start_wall
    checksum = float(np.linalg.norm(output.stress) + np.linalg.norm(output.displacement))
    arrays = engine.q_buffers + engine.s_buffers + engine.g_buffers
    return {
        "wall_seconds": elapsed_wall,
        "cpu_seconds": elapsed_cpu,
        "wall_ms_per_frame": 1000.0 * elapsed_wall / frames,
        "cpu_ms_per_frame": 1000.0 * elapsed_cpu / frames,
        "last_checksum_outside_timing": checksum,
        "accepted_predictions": int(accepted_count),
        "failed_columns": int(failed_count),
        "mutable_state_bytes": sum(array.nbytes for array in arrays) + engine.depth.nbytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--scopes", nargs="+", choices=["recovery", "complete"], default=["recovery", "complete"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("history_timings.json"))
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be finite and positive")
    initial_hashes = source_hashes()
    rows = []
    with threadpool_limits(limits=1):
        mother = Mother()
        system = RecoverySystem.from_mother(mother)
        for batch in args.batches:
            trajectory = prepare_scaled_trajectory(mother, args.speed, batch, frames=args.frames, order=args.order)
            # Populate BLAS/sparse dispatch before any timed cold-history trace.
            system.stress @ system.solve(trajectory.forces[0][:, :1])
            for scope in args.scopes:
                methods = configurations(scope)
                samples = {method[0]: [] for method in methods}
                for repeat in range(args.repeats):
                    ordered = methods if repeat % 2 == 0 else tuple(reversed(methods))
                    for name, tolerance, rotation, cached, prediction in ordered:
                        engine = BatchHistory(
                            system, batch, tolerance=tolerance, rotation=rotation,
                            cached_product=cached, prediction=prediction,
                        )
                        sample = run_trace(engine, mother, trajectory, args.order,
                                           complete=scope == "complete", speed=args.speed)
                        sample.update(repeat=repeat)
                        samples[name].append(sample)
                        print(json.dumps({"batch": batch, "speed": args.speed, "scope": scope, "method": name,
                                          "repeat": repeat, "wall_ms": sample["wall_ms_per_frame"]}), flush=True)
                for name, values in samples.items():
                    wall = [value["wall_ms_per_frame"] for value in values]
                    cpu = [value["cpu_ms_per_frame"] for value in values]
                    rows.append({
                        "batch": batch, "frames": args.frames, "order": args.order, "speed": args.speed,
                        "scope": scope, "method": name, "force_sha256": trajectory.force_sha256,
                        "load_preparation_seconds": trajectory.construction_seconds,
                        "wall_median_ms": float(np.median(wall)),
                        "wall_min_ms": min(wall), "wall_max_ms": max(wall),
                        "cpu_median_ms": float(np.median(cpu)), "samples": values,
                    })
        pools = threadpool_info()
    final_hashes = source_hashes()
    if final_hashes != initial_hashes:
        raise RuntimeError("A captured source changed during the benchmark.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "reconstruction": True,
        "speed": args.speed, "speed_definition": SPEED_DEFINITION,
        "timing_scope": "CPU complete local-three-component shell field, not global-nine-component export",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "affinity_cpus": sorted(os.sched_getaffinity(0)),
                        "threadpools": pools},
        "source_hashes": initial_hashes,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
