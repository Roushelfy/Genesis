"""Full contact-replay stress scope, separate assembled-RHS ablations, and identical-RHS CPU validation."""

import argparse
import csv
import hashlib
import json
import subprocess
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import ContactBatch, EggConfig, EggRecoveryCPU
from .cudss import SharedCuDSSFactor
from .device_pressure import PadPressureGPU
from .graph_gpu import CapturedDirectRecoveryGPU
from .peak_tensor import P2PeakTensorGPU
from .replay import ContactReplayGPU, ReplayRecoveryPipeline
from .sparse_gpu import EggRecoveryGPU
from .temporal_gpu import TemporalRecoveryGPU
from .timing import DeviceMemorySampler, factor_metadata, host_metadata, source_hashes


def main(
    argv: Sequence[str] | None = None,
    shared_model: EggRecoveryCPU | None = None,
    shared_factor: SharedCuDSSFactor | None = None,
    shared_stream: cp.cuda.Stream | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--factor-backend", choices=("spsm", "cudss"), default="cudss")
    parser.add_argument("--native-order", choices=("auto", "natural"), default="auto")
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
    parser.add_argument("--scatter", choices=("atomic", "warp"), default="atomic")
    parser.add_argument("--inertia", choices=("sparse", "quadratic"), default="sparse")
    parser.add_argument("--body-products", choices=("cublas", "fused"), default="cublas")
    parser.add_argument("--sparse-layout", choices=("F", "C"), default="F")
    parser.add_argument("--recovery-graph", action="store_true")
    parser.add_argument("--scope", choices=("stress", "assembled-rhs"), default="stress")
    parser.add_argument("--graph", action="store_true", help="Capture the direct assembled-RHS recovery scope")
    parser.add_argument("--async-allocator", action="store_true", help="Use the public cuDSS stream-ordered allocator")
    parser.add_argument(
        "--stages", action="store_true", help="Separately measure warm stage spans across a full replay"
    )
    parser.add_argument("--abrupt", action="store_true", help="Permute complete recorded source frames each transition")
    parser.add_argument("--reset-every-frame", action="store_true", help="Include cold-history all-failed controls")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify-frames", type=int, default=0)
    parser.add_argument("--verify-stride", type=int, default=1)
    parser.add_argument(
        "--verify-mapping", action="store_true", help="Also compare complete CPU-assembled contact/body RHS"
    )
    parser.add_argument(
        "--validation-only", action="store_true", help="Run CPU same-RHS checks without throughput repetitions"
    )
    parser.add_argument(
        "--save-rhs", type=Path, help="Explicit data-side destination for complete FP64 verification RHS"
    )
    args = parser.parse_args(argv)
    if args.seconds < 10 or args.repeats < 3:
        raise ValueError("Warmed measurements require at least ten seconds and three repeats")
    if args.verify_frames and args.cpu_factor == "none":
        raise ValueError("Same-RHS CPU verification requires an explicit FP64 CPU factor")
    if args.verify_stride < 1 or (args.validation_only and args.verify_frames < 1):
        raise ValueError("Validation requires a positive source stride and at least one requested frame")
    if args.precision != "64" and args.method == "direct":
        raise ValueError("FP32 requires explicit refinement and FP64 fallback")
    if args.graph and (args.scope != "assembled-rhs" or args.method != "direct"):
        raise ValueError("This graph experiment covers direct recovery of an assembled RHS")
    if args.recovery_graph and (
        args.method != "direct"
        or args.factor_backend != "cudss"
        or args.peak != "fused"
        or args.graph
        or shared_factor is not None
    ):
        raise ValueError("Pipeline recovery graphs require their owned native FP64 factors and fused full peak")
    if args.recovery_graph and args.native_order != "auto":
        raise ValueError("Captured recovery uses the validated automatic native ordering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    rows, validation = [], []
    config = EggConfig(level=args.level, ordering="column-nd", factor_backend=args.cpu_factor)
    with threadpool_limits(limits=1):
        start = perf_counter()
        model = (
            EggRecoveryCPU(args.envs, config, history=0, direct=True, anchor_to_surface=True)
            if shared_model is None
            else shared_model
        )
        if model.config != config or (args.verify_frames and model.environments != args.envs):
            raise ValueError("Reused operators must match the declared mesh/material and CPU verification batch")
        stream = cp.cuda.Stream(non_blocking=True) if shared_stream is None else shared_stream
        with stream:
            rtol = 1e-6 if args.profile == "strict" else 1e-3
            if args.recovery_graph:
                recovery = CapturedDirectRecoveryGPU(
                    model.fem, args.envs, rtol, args.inertia, args.body_products, args.sparse_layout
                )
            elif args.method == "direct":
                recovery = EggRecoveryGPU(
                    model.fem,
                    args.envs,
                    rtol=rtol,
                    factor_backend=args.factor_backend,
                    natural_order=args.native_order == "natural",
                    async_allocations=args.graph or args.async_allocator,
                    shared_factor=shared_factor,
                    inertia=args.inertia,
                    body_products=args.body_products,
                    sparse_layout=args.sparse_layout,
                )
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
                    natural_order=args.native_order == "natural",
                    shared_factor=shared_factor,
                    inertia=args.inertia,
                    body_products=args.body_products,
                    sparse_layout=args.sparse_layout,
                )
            if args.peak == "tensor":
                recovery.peak = P2PeakTensorGPU(model.fem.glambda, model.fem.elements, config.young_pa / 2 / 1.3)
            source = ContactReplayGPU(args.replay, args.envs)
            if source.frames < 600:
                raise ValueError("Complete-grasp benchmarks require at least 600 recorded physical samples")
            pipeline = ReplayRecoveryPipeline(
                source,
                PadPressureGPU(model.surface, anchor_to_surface=True, sampling=args.sampling, scatter=args.scatter),
                recovery,
            )
            setup_s = perf_counter() - start
            factor_info = factor_metadata(recovery)
            print(json.dumps({"setup_seconds": setup_s, "device_factors": factor_info}), flush=True)
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
            peak_errors, zero_errors, positive_reference = [], [], []
            snapshots = []
            start = perf_counter()
            warmup_ticks = (
                (0, 100, 200, 300, 450, 550)
                if args.scope == "assembled-rhs" or args.validation_only
                else range(source.frames)
            )
            for tick in warmup_ticks:
                out = pipeline.step(tick)
                exact = (
                    EggRecoveryGPU.recover(recovery, pipeline.last_rhs)
                    if args.recovery_graph
                    else out if reference is recovery else reference.recover(pipeline.last_rhs)
                )
                warmup_accepted &= out.is_accepted & exact.is_accepted & pipeline.mapping_accepted
                difference = abs(out.peak_pa - exact.peak_pa)
                peak_errors.append(cp.where(exact.peak_pa > 1, difference / cp.maximum(exact.peak_pa, 1), 0))
                zero_errors.append(cp.where(exact.peak_pa <= 1, difference, 0))
                positive_reference.append(exact.peak_pa > 1)
                if args.recovery_graph:
                    # Do not retain an untimed full-domain eager displacement during the next mapped load.
                    del exact
                if args.scope == "assembled-rhs":
                    snapshots.append(pipeline.last_rhs.copy())
                if tick % 200 == 0:
                    print(json.dumps({"warmup_frame": tick}), flush=True)
            stream.synchronize()
            assert cp.all(warmup_accepted).item()
            peak_errors, zero_errors = cp.stack(peak_errors), cp.stack(zero_errors)
            positive_reference = cp.stack(positive_reference)
            nonzero_errors = peak_errors[positive_reference]
            assert float(peak_errors.max()) <= (1e-4 if args.profile == "strict" else 1e-2)
            assert float(zero_errors.max()) <= 1e-3
            gpu_validation = {
                "oracle": "Same complete device RHS, shared FP64 direct factor, independently CPU-tested backend",
                "frames": len(warmup_ticks),
                "eager_reference_shares_immutable_operators": args.recovery_graph,
                "peak_relative_error_max": float(peak_errors.max()),
                "peak_relative_error_p95": float(cp.percentile(nonzero_errors, 95)) if nonzero_errors.size else 0.0,
                "peak_relative_error_mean": float(nonzero_errors.mean()) if nonzero_errors.size else 0.0,
                "relative_error_scope": "Reference global peak >1 Pa; near-zero cases reported separately",
                "nonzero_environment_frames": int(nonzero_errors.size),
                "near_zero_environment_frames": int(positive_reference.size - nonzero_errors.size),
                "near_zero_absolute_peak_error_max_pa": float(zero_errors.max()),
            }
            warmup_s = perf_counter() - start
            graph, graphed, rhs_buffer = None, None, None
            capture_pool = None
            graph_info = {
                "requested": args.graph or args.recovery_graph,
                "supported": True if args.recovery_graph else None,
                "scope": "Full recovery of each changing complete RHS"
                if args.recovery_graph
                else "Assembled-RHS experiment",
            }
            if args.graph:
                rhs_buffer = cp.zeros_like(pipeline.last_rhs, order="F")
                rhs_buffer[:] = snapshots[2]
                for _ in range(4):
                    recovery.recover(rhs_buffer)
                stream.synchronize()
                # Retain capture temporaries in an isolated pool so eager validation cannot reuse their addresses.
                capture_pool = cp.cuda.MemoryPool()
                try:
                    with cp.cuda.using_allocator(capture_pool.malloc):
                        stream.begin_capture()
                        graphed = recovery.recover(rhs_buffer)
                        graph = stream.end_capture()
                except (RuntimeError, cp.cuda.runtime.CUDARuntimeError):
                    graph_info.update(supported=False, reason=traceback.format_exc())
                    if stream.is_capturing():
                        try:
                            stream.end_capture()
                        except cp.cuda.runtime.CUDARuntimeError:
                            pass
                    args.output.write_text(
                        json.dumps(
                            {
                                "GPU_tested": True,
                                "timing_measured": False,
                                "graph": graph_info,
                                "device_factors": factor_info,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    print(json.dumps(graph_info), flush=True)
                    return
                graph_error = 0.0
                for frame in range(10):
                    rhs = snapshots[frame % len(snapshots)] * (1 + frame * 0.13)
                    rhs_buffer[:] = rhs
                    graph.launch(stream)
                    exact = recovery.recover(rhs)
                    difference = abs(graphed.peak_pa - exact.peak_pa) / cp.maximum(exact.peak_pa, 1)
                    graph_error = max(graph_error, float(difference.max()))
                    print(
                        json.dumps(
                            {
                                "graph_validation_frame": frame,
                                "relative_residual": graphed.relative_residual.get().tolist(),
                                "absolute_residual": graphed.absolute_residual_n.get().tolist(),
                                "exact_relative_residual": exact.relative_residual.get().tolist(),
                            }
                        ),
                        flush=True,
                    )
                    assert cp.all(graphed.is_accepted & exact.is_accepted).item()
                assert graph_error <= 1e-4
                graph_info.update(supported=True, peak_relative_error_max=graph_error)
            if args.save_rhs is not None:
                args.save_rhs.mkdir(parents=True, exist_ok=True)
            recovery.reset()
            for verification_index in range(args.verify_frames):
                tick = verification_index * args.verify_stride
                out = pipeline.step(tick)
                rhs = pipeline.last_rhs.get()
                mapping_validation = None
                if args.verify_mapping:
                    frame = source.frame(tick)
                    contacts = ContactBatch(
                        frame.position_m.get(),
                        frame.force_n.get(),
                        frame.radius_m.get(),
                        frame.valid.get(),
                        frame.friction.get(),
                        frame.normal.get(),
                        source.source_epsilon,
                    )
                    mapped_cpu = model.map_contacts(contacts)
                    cpu_external = mapped_cpu.nodal_force_n + model.fem.mr[:, :3] @ frame.gravity_m_s2.get().T
                    cpu_rhs = model.compatible_rhs(cpu_external, frame.omega_rad_s.get())
                    discrepancy = np.linalg.norm(rhs - cpu_rhs, axis=0)
                    np.testing.assert_allclose(rhs, cpu_rhs, rtol=1e-7, atol=1e-12)
                    mapping_validation = {
                        "complete_rhs_difference_max_n": float(np.max(abs(rhs - cpu_rhs))),
                        "complete_rhs_difference_norm_n": discrepancy.tolist(),
                        "complete_rhs_reference_norm_n": np.linalg.norm(cpu_rhs, axis=0).tolist(),
                        "cpu_moment_error_max_nm": float(
                            np.max(abs(mapped_cpu.resultant_moment_nm - mapped_cpu.input_moment_nm))
                        ),
                    }
                expected = model.recover(rhs)
                actual = out.peak_pa.get()
                difference = abs(actual - expected.peak_pa)
                relative = difference / np.maximum(expected.peak_pa, 1)
                zero = expected.peak_pa <= 1
                assert cp.all(out.is_accepted & pipeline.mapping_accepted).item()
                assert relative[~zero].max(initial=0) <= (1e-4 if args.profile == "strict" else 1e-2)
                assert difference[zero].max(initial=0) <= 1e-3
                if args.save_rhs is not None:
                    np.savez_compressed(
                        args.save_rhs / f"frame-{tick:06d}.npz",
                        rhs_n=rhs,
                        reference_peak_pa=expected.peak_pa,
                        optimized_peak_pa=actual,
                    )
                validation.append(
                    {
                        "frame": tick,
                        "CPU_mapping_validation": mapping_validation,
                        "rhs_sha256": hashlib.sha256(rhs.tobytes(order="F")).hexdigest(),
                        "reference_peak_pa": expected.peak_pa.tolist(),
                        "optimized_peak_pa": actual.tolist(),
                        "peak_relative_error": relative.tolist(),
                        "near_zero_absolute_error_pa": difference[zero].tolist(),
                        "full_relative_residual": out.relative_residual.get().tolist(),
                        "full_absolute_residual_n": out.absolute_residual_n.get().tolist(),
                    }
                )
                with args.output.with_suffix(".validation.jsonl").open("a") as trace:
                    trace.write(json.dumps(validation[-1]) + "\n")
                if tick % 50 == 0:
                    print(json.dumps({"verified_frame": tick}), flush=True)
            for repeat in range(0 if args.validation_only else args.repeats):
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
                    minimum_steps = source.frames if args.scope == "stress" else 1
                    while steps < minimum_steps or perf_counter() - start < args.seconds:
                        tick = (steps * 163 + 71) % source.frames if args.abrupt else steps
                        if args.reset_every_frame:
                            recovery.reset()
                            pipeline.reset_count += args.envs
                        if args.scope == "stress":
                            out = pipeline.step(tick)
                            accepted &= pipeline.mapping_accepted
                        else:
                            # An explicitly synthetic changing combination for the isolated recovery ablation.
                            phase = steps * 0.03
                            rhs = snapshots[0] + np.sin(phase) * snapshots[2] + np.cos(phase * 0.7) * snapshots[4]
                            if args.abrupt:
                                rhs += snapshots[(steps * 17) % len(snapshots)]
                            if steps % source.frames == 0 and not args.reset_every_frame:
                                recovery.reset()
                                pipeline.reset_count += args.envs
                            if graph is None:
                                out = recovery.recover(rhs, source.dt_s)
                            else:
                                rhs_buffer[:] = rhs
                                graph.launch(stream)
                                out = graphed
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
            stage_info = None
            if args.stages:
                recovery.reset()
                for tick in range(source.frames):
                    pipeline.step(tick, measure=tick % 12 == 0)
                stream.synchronize()
                stage_info = pipeline.stage_metadata()
        gpu = cp.cuda.runtime.getDeviceProperties(0)
        report = {
            "GPU_tested": True,
            "accepted": True,
            "throughput_measured": bool(rows),
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
            "host": host_metadata(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "CUDA_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "setup_seconds": setup_s,
            "warmup_seconds": warmup_s,
            "warmup_frames": len(warmup_ticks),
            "source_sha256_at_start": hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "CPU_solve_in_timed_loop": False,
            "device_factors": factor_info,
            "GPU_direct_validation": gpu_validation,
            "graph": graph_info,
            "stage_spans": stage_info,
            "assembled_scope_input": "Changing sinusoidal combinations of six complete live RHS snapshots",
            "validation": validation,
            "runs": rows,
            "mean_recovery_transitions_s": (
                sum(row["recovery_transitions"] for row in rows) / sum(row["wall_seconds"] for row in rows)
                if rows
                else None
            ),
            "repeat_rate_std": float(np.std([row["recovery_transitions_s"] for row in rows])) if rows else None,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if rows:
            with args.output.with_suffix(".csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
