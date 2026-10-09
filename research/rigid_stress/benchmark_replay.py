"""Full contact-replay stress scope, separate assembled-RHS ablations, and identical-RHS CPU validation."""

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .device_pressure import PadPressureGPU
from .peak_tensor import P2PeakTensorGPU
from .replay import ContactReplayGPU, ReplayRecoveryPipeline
from .sparse_gpu import EggRecoveryGPU
from .temporal_gpu import TemporalRecoveryGPU
from .timing import DeviceMemorySampler, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--factor-backend", choices=("spsm", "cudss"), default="cudss")
    parser.add_argument("--cpu-factor", choices=("superlu", "cholmod", "none"), default="none")
    parser.add_argument("--profile", choices=("strict", "throughput"), default="strict")
    parser.add_argument("--method", choices=("direct", "temporal"), default="direct")
    parser.add_argument("--history", type=int, choices=(0, 4, 8), default=4)
    parser.add_argument("--strategy", choices=("compact", "padded", "adaptive"), default="compact")
    parser.add_argument("--layout", choices=("history-major", "dof-major"), default="history-major")
    parser.add_argument("--precision", choices=("32", "64"), default="64")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--refinements", type=int, default=4)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--rebuild-gram", action="store_true")
    parser.add_argument("--peak", choices=("fused", "tensor"), default="fused")
    parser.add_argument("--sampling", choices=("scan", "grid"), default="grid")
    parser.add_argument("--scope", choices=("stress", "assembled-rhs"), default="stress")
    parser.add_argument("--abrupt", action="store_true", help="Permute complete recorded source frames each transition")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify-frames", type=int, default=0)
    parser.add_argument(
        "--save-rhs", type=Path, help="Explicit data-side destination for complete FP64 verification RHS"
    )
    args = parser.parse_args()
    if args.seconds < 10 or args.repeats < 3:
        raise ValueError("Warmed measurements require at least ten seconds and three repeats")
    if args.verify_frames and args.cpu_factor == "none":
        raise ValueError("Same-RHS CPU verification requires an explicit FP64 CPU factor")
    if args.precision != "64" and args.method == "direct":
        raise ValueError("FP32 requires explicit refinement and FP64 fallback")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    rows, validation = [], []
    config = EggConfig(level=args.level, ordering="column-nd", factor_backend=args.cpu_factor)
    with threadpool_limits(limits=1):
        start = perf_counter()
        model = EggRecoveryCPU(args.envs, config, history=0, direct=True, anchor_to_surface=True)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            rtol = 1e-6 if args.profile == "strict" else 1e-3
            if args.method == "direct":
                recovery = EggRecoveryGPU(model.fem, args.envs, rtol=rtol, factor_backend=args.factor_backend)
            else:
                recovery = TemporalRecoveryGPU(
                    model.fem,
                    args.envs,
                    args.history,
                    rtol,
                    args.strategy,
                    args.precision,
                    args.scaling,
                    args.refinements,
                    args.chunk,
                    cached_gram=not args.rebuild_gram,
                    factor_backend=args.factor_backend,
                    layout=args.layout,
                )
            if args.peak == "tensor":
                recovery.peak = P2PeakTensorGPU(model.fem.glambda, model.fem.elements, config.young_pa / 2 / 1.3)
            source = ContactReplayGPU(args.replay, args.envs)
            pipeline = ReplayRecoveryPipeline(
                source,
                PadPressureGPU(model.surface, anchor_to_surface=True, sampling=args.sampling),
                recovery,
            )
            setup_s = perf_counter() - start
            reference = (
                recovery
                if args.method == "direct"
                else EggRecoveryGPU(
                    model.fem,
                    args.envs,
                    factor_backend=args.factor_backend,
                    shared_factor=recovery.factor,
                )
            )
            warmup_accepted = cp.ones(args.envs, dtype=bool)
            peak_errors, zero_errors = [], []
            snapshots = []
            start = perf_counter()
            for tick in range(source.frames):
                out = pipeline.step(tick)
                exact = out if args.method == "direct" else reference.recover(pipeline.last_rhs)
                warmup_accepted &= out.is_accepted & exact.is_accepted & pipeline.mapping_accepted
                difference = abs(out.peak_pa - exact.peak_pa)
                peak_errors.append(cp.where(exact.peak_pa > 1, difference / cp.maximum(exact.peak_pa, 1), 0))
                zero_errors.append(cp.where(exact.peak_pa <= 1, difference, 0))
                if args.scope == "assembled-rhs" and tick in (0, 100, 200, 300, 450, 550):
                    snapshots.append(pipeline.last_rhs.copy())
            stream.synchronize()
            assert cp.all(warmup_accepted).item()
            peak_errors, zero_errors = cp.stack(peak_errors), cp.stack(zero_errors)
            assert float(peak_errors.max()) <= (1e-4 if args.profile == "strict" else 1e-2)
            assert float(zero_errors.max()) <= 1e-3
            gpu_validation = {
                "oracle": "Same complete device RHS, shared FP64 direct factor, independently CPU-tested backend",
                "frames": source.frames,
                "peak_relative_error_max": float(peak_errors.max()),
                "peak_relative_error_p95": float(cp.percentile(peak_errors, 95)),
                "peak_relative_error_mean": float(peak_errors.mean()),
                "near_zero_absolute_peak_error_max_pa": float(zero_errors.max()),
            }
            warmup_s = perf_counter() - start
            if args.save_rhs is not None:
                args.save_rhs.mkdir(parents=True, exist_ok=True)
            recovery.reset()
            for tick in range(args.verify_frames):
                out = pipeline.step(tick)
                rhs = pipeline.last_rhs.get()
                expected = model.recover(rhs)
                actual = out.peak_pa.get()
                difference = abs(actual - expected.peak_pa)
                relative = difference / np.maximum(expected.peak_pa, 1)
                zero = expected.peak_pa <= 1
                assert cp.all(out.is_accepted & pipeline.mapping_accepted).item()
                assert relative[~zero].max(initial=0) <= (1e-4 if args.profile == "strict" else 1e-2)
                assert difference[zero].max(initial=0) <= 1e-3
                if args.save_rhs is not None:
                    np.savez_compressed(args.save_rhs / f"frame-{tick:06d}.npz", rhs_n=rhs)
                validation.append(
                    {
                        "frame": tick,
                        "rhs_sha256": hashlib.sha256(rhs.tobytes(order="F")).hexdigest(),
                        "reference_peak_pa": expected.peak_pa.tolist(),
                        "optimized_peak_pa": actual.tolist(),
                        "peak_relative_error": relative.tolist(),
                        "near_zero_absolute_error_pa": difference[zero].tolist(),
                        "full_relative_residual": out.relative_residual.get().tolist(),
                        "full_absolute_residual_n": out.absolute_residual_n.get().tolist(),
                    }
                )
                if tick % 50 == 0:
                    print(json.dumps({"verified_frame": tick}), flush=True)
            for repeat in range(args.repeats):
                recovery.reset()
                pipeline.reset_count = 0
                accepted = cp.ones(args.envs, dtype=bool)
                peak, residual = cp.zeros(args.envs), cp.zeros(args.envs)
                meaningful_residual, absolute_residual = cp.zeros(args.envs), cp.zeros(args.envs)
                failure, fallback, refinement = (cp.zeros(args.envs, dtype=np.int64) for _ in range(3))
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                stream.synchronize()
                begin.record()
                start, steps = perf_counter(), 0
                with DeviceMemorySampler() as memory:
                    while steps < source.frames or perf_counter() - start < args.seconds:
                        tick = (steps * 163 + 71) % source.frames if args.abrupt else steps
                        if args.scope == "stress":
                            out = pipeline.step(tick)
                            accepted &= pipeline.mapping_accepted
                        else:
                            # An explicitly synthetic changing combination for the isolated recovery ablation.
                            phase = steps * 0.03
                            rhs = snapshots[0] + np.sin(phase) * snapshots[2] + np.cos(phase * 0.7) * snapshots[4]
                            if args.abrupt:
                                rhs += snapshots[(steps * 17) % len(snapshots)]
                            if steps % source.frames == 0:
                                recovery.reset()
                            out = recovery.recover(rhs, source.dt_s)
                        accepted &= out.is_accepted
                        cp.maximum(peak, out.peak_pa, out=peak)
                        cp.maximum(residual, out.relative_residual, out=residual)
                        cp.maximum(
                            meaningful_residual,
                            cp.where(out.rhs_norm_n >= recovery.atol_n / recovery.rtol, out.relative_residual, 0),
                            out=meaningful_residual,
                        )
                        cp.maximum(absolute_residual, out.absolute_residual_n, out=absolute_residual)
                        failure += recovery.statistics.failed
                        fallback += recovery.statistics.used_fp64_fallback
                        refinement += recovery.statistics.refinement_count
                        steps += 1
                    end.record()
                    stream.synchronize()
                elapsed = perf_counter() - start
                assert cp.all(accepted).item()
                row = {
                    "repeat": repeat,
                    "scope": args.scope,
                    "environments": args.envs,
                    "steps": steps,
                    "recovery_transitions": steps * args.envs,
                    "wall_seconds": elapsed,
                    "cuda_span_seconds": cp.cuda.get_elapsed_time(begin, end) / 1000,
                    "recovery_transitions_s": steps * args.envs / elapsed,
                    "batch_steps_s": steps / elapsed,
                    "reset_environments": pipeline.reset_count,
                    "accepted": True,
                    "peak_max_pa": float(peak.max()),
                    "full_relative_residual_max": float(residual.max()),
                    "full_relative_residual_max_above_absolute_floor": float(meaningful_residual.max()),
                    "full_absolute_residual_max_n": float(absolute_residual.max()),
                    "correction_environment_events": int(failure.sum()),
                    "fp64_fallback_events": int(fallback.sum()),
                    "refinement_events": int(refinement.sum()),
                    **memory.metadata(),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        gpu = cp.cuda.runtime.getDeviceProperties(0)
        report = {
            "GPU_tested": True,
            "accepted": True,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "scope": args.scope,
            "mesh": asdict(config),
            "dofs": model.fem.ndof,
            "tetrahedra": model.fem.ne,
            "dt_s": source.dt_s,
            "source_environments": source.source_environments,
            "source_frames": source.frames,
            "source_replicated_for_larger_batches": args.envs > source.source_environments,
            "GPU": gpu["name"].decode(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "CUDA_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "setup_seconds": setup_s,
            "warmup_seconds": warmup_s,
            "warmup_frames": source.frames,
            "source_sha256_at_start": hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "CPU_solve_in_timed_loop": False,
            "GPU_direct_validation": gpu_validation,
            "assembled_scope_input": "Changing sinusoidal combinations of six complete live RHS snapshots",
            "validation": validation,
            "runs": rows,
            "mean_recovery_transitions_s": sum(row["recovery_transitions"] for row in rows)
            / sum(row["wall_seconds"] for row in rows),
            "repeat_rate_std": float(np.std([row["recovery_transitions_s"] for row in rows])),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
