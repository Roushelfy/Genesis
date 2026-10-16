"""Measure warmed shared sparse recovery on assembled synthetic loads.

This scope includes gather, triangular solves, scatter, complete residual and full-domain peak. It excludes contact
mapping, Genesis dynamics, history, reset and policy inference, so the rate is recoveries/s, not env transitions/s.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .sparse_gpu import EggRecoveryGPU


@dataclass(frozen=True)
class Measurement:
    environments: int
    repeat: int
    batch_recoveries: int
    wall_seconds: float
    device_seconds: float
    recoveries_per_second: float
    batch_recoveries_per_second: float
    maximum_full_relative_residual: float
    maximum_peak_relative_error: float
    reserved_cupy_pool_bytes: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 10 or args.repeats < 3 or min(args.envs) < 1:
        raise ValueError("Use at least 10 seconds, 3 repeats and positive environment counts")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(config=EggConfig(level=args.level), history=0, direct=True)
        for environments in args.envs:
            start = perf_counter()
            gpu = EggRecoveryGPU(model.fem, environments)
            rng = np.random.default_rng(3619 + environments)
            rhs = model.fem.balance(rng.normal(size=(model.fem.ndof, environments)))
            expected = np.zeros_like(rhs)
            expected[model.fem.free] = model.fem.factor.solve(rhs[model.fem.free])
            truth = np.array([model.peak(expected[:, i_b])[0] for i_b in range(environments)])
            device_rhs = cp.asarray(rhs)
            for _ in range(5):
                result = gpu.recover(device_rhs)
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_array_equal(cp.asnumpy(result.is_accepted), True)
            error = abs(cp.asnumpy(result.peak_pa) - truth) / np.maximum(truth, 1)
            assert error.max() <= 1e-4
            print(
                json.dumps({"environments": environments, "build_and_warmup_seconds": perf_counter() - start}),
                flush=True,
            )
            for repeat in range(args.repeats):
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                batches = 0
                cp.cuda.get_current_stream().synchronize()
                start = perf_counter()
                begin.record()
                while perf_counter() - start < args.seconds:
                    for _ in range(16):
                        result = gpu.recover(device_rhs)
                    cp.cuda.get_current_stream().synchronize()
                    batches += 16
                end.record()
                end.synchronize()
                elapsed = perf_counter() - start
                device_seconds = cp.cuda.get_elapsed_time(begin, end) / 1000
                np.testing.assert_array_equal(cp.asnumpy(result.is_accepted), True)
                row = Measurement(
                    environments,
                    repeat,
                    batches,
                    elapsed,
                    device_seconds,
                    environments * batches / elapsed,
                    batches / elapsed,
                    float(cp.asnumpy(result.relative_residual).max()),
                    float(error.max()),
                    cp.get_default_memory_pool().total_bytes(),
                )
                rows.append(asdict(row))
                print(json.dumps(asdict(row)), flush=True)
                properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
                report = {
                    "scope": __doc__,
                    "gpu_name": properties["name"].decode(),
                    "cupy": cp.__version__,
                    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                    "driver": cp.cuda.runtime.driverGetVersion(),
                    "profile": "strict, FP64 direct, full residual <=1e-6, peak error <=1e-4",
                    "mesh": asdict(model.config),
                    "dofs": model.fem.ndof,
                    "tetrahedra": model.fem.ne,
                    "factor_nonzeros": model.fem.factor.L.nnz + model.fem.factor.U.nnz,
                    "physical_mesh_converged": False,
                    "live_scene_tested": False,
                    "raw_measurements": rows,
                }
                (args.output / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
                with (args.output / "recovery.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)


if __name__ == "__main__":
    main()
