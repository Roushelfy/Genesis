"""Measure exact pressure-map implementations on complete recorded contact snapshots.

These are mapping-only diagnostics. They do not report environment transitions/s or replace full-grasp benchmarks.
The spatial lookup changes candidate enumeration only; every quadrature point inside each footprint remains included.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .device_pressure import PadPressureGPU
from .replay import ContactReplayGPU
from .timing import DeviceMemorySampler, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--batches", type=int, nargs="+", default=(8, 32, 128))
    parser.add_argument("--widths", type=float, nargs="+", default=(0.003, 0.006))
    parser.add_argument("--stride", type=int, default=60)
    args = parser.parse_args()
    if args.stride < 1 or min(args.batches) < 1:
        raise ValueError("Positive sampling stride and batch counts are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes, rows = source_hashes(), []
    config = EggConfig(level=args.level, ordering="column-nd", factor_backend="none")
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(1, config, history=0)
        modes = cp.asarray(model.fem.r)
        center_of_mass = cp.asarray(model.fem.com)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            for environments in args.batches:
                source = ContactReplayGPU(args.replay, environments)
                frames = list(range(0, source.frames, args.stride))
                for width in args.widths:
                    for scatter in ("atomic", "warp"):
                        mapper = PadPressureGPU(
                            model.surface, anchor_to_surface=True, sampling="grid", grid_width_m=width, scatter=scatter
                        )
                        for tick in frames[:2]:
                            frame = source.frame(tick)
                            mapper.map(
                                frame.position_m,
                                frame.force_n,
                                frame.radius_m,
                                frame.normal,
                                frame.friction,
                                frame.valid,
                                source.source_epsilon,
                            )
                        stream.synchronize()
                        accepted = cp.ones(environments, dtype=bool)
                        force_error, moment_error = 0.0, 0.0
                        for repeat in range(3):
                            markers, outputs = [], []
                            start = perf_counter()
                            with DeviceMemorySampler() as memory:
                                for tick in frames:
                                    frame = source.frame(tick)
                                    first, last = cp.cuda.Event(), cp.cuda.Event()
                                    first.record()
                                    mapped = mapper.map(
                                        frame.position_m,
                                        frame.force_n,
                                        frame.radius_m,
                                        frame.normal,
                                        frame.friction,
                                        frame.valid,
                                        source.source_epsilon,
                                    )
                                    last.record()
                                    accepted &= mapped.is_accepted
                                    markers.append((first, last))
                                    # Conservation checks are outside the event span. Keep only the small wrench.
                                    wrench = modes.T @ mapped.nodal_force_n
                                    external_force = cp.where(frame.valid[..., None], frame.force_n, 0)
                                    expected_force = external_force.sum(axis=1)
                                    expected_moment = cp.cross(frame.position_m - center_of_mass, external_force).sum(
                                        axis=1
                                    )
                                    outputs.append((wrench, expected_force, expected_moment))
                                stream.synchronize()
                            elapsed = perf_counter() - start
                            assert cp.all(accepted).item(), "Rejected complete source contacts"
                            for wrench, expected_force, expected_moment in outputs:
                                force_error = max(force_error, float(cp.max(abs(wrench[:3].T - expected_force))))
                                moment_error = max(moment_error, float(cp.max(abs(wrench[3:].T - expected_moment))))
                            values = [cp.cuda.get_elapsed_time(first, last) for first, last in markers]
                            row = {
                                "repeat": repeat,
                                "environments": environments,
                                "scatter": scatter,
                                "grid_width_m": width,
                                "sampled_frames": len(frames),
                                "mapping_mean_ms": float(np.mean(values)),
                                "mapping_p95_ms": float(np.percentile(values, 95)),
                                "mapping_max_ms": float(np.max(values)),
                                "diagnostic_wall_seconds": elapsed,
                                "force_error_max_n": force_error,
                                "moment_error_max_nm": moment_error,
                                **memory.metadata(),
                            }
                            rows.append(row)
                            print(json.dumps(row), flush=True)
                        del mapper
                        cp.get_default_memory_pool().free_all_blocks()
        gpu = cp.cuda.runtime.getDeviceProperties(0)
        report = {
            "scope": "Exact pressure mapping only; stratified complete recorded contact snapshots",
            "environment_throughput_measured": False,
            "source_sha256_at_start": hashes,
            "mesh": asdict(config),
            "GPU": gpu["name"].decode(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "runs": rows,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        with args.output.with_suffix(".csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
