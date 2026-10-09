"""Warmed complete rigid grasps with scoped wall/device timing and every-substep stress observations."""

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import genesis as gs

from .cpu import EggConfig, EggRecoveryCPU
from .graph_gpu import CapturedDirectRecoveryGPU
from .panda_scene import PandaConfig, PandaEggScene
from .peak_tensor import P2PeakTensorGPU
from .sparse_gpu import EggRecoveryGPU
from .temporal_gpu import TemporalRecoveryGPU
from .timing import DeviceMemorySampler, factor_metadata, host_metadata, source_hashes
from .trajectory_metrics import PandaTrajectoryMetrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-assets", type=Path, help="Rigid-only scopes can reuse identical exported shell assets")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--cpu-factor", choices=("superlu", "cholmod", "none"), default="superlu")
    parser.add_argument("--factor-backend", choices=("spsm", "cudss"), default="spsm")
    parser.add_argument("--native-order", choices=("auto", "natural"), default="auto")
    parser.add_argument("--scope", choices=("rigid", "live", "policy-rigid", "policy"), default="live")
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
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=610000)
    parser.add_argument("--substeps", type=int, default=1)
    args = parser.parse_args()
    if args.seconds < 10 or args.repeats < 3:
        raise ValueError("Acceptance timings require at least ten seconds and three repetitions")
    if args.method == "direct" and args.precision != "64":
        raise ValueError("The direct baseline is FP64; FP32 uses explicit refinement/fallback")
    if args.recovery_graph and (args.method != "direct" or args.factor_backend != "cudss" or args.peak != "fused"):
        raise ValueError("Captured live recovery requires native FP64 direct factors and the fused full peak")
    if args.recovery_graph and args.native_order != "auto":
        raise ValueError("Captured recovery uses the validated automatic native ordering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    rows = []
    with threadpool_limits(limits=1):
        start = perf_counter()
        assets = args.output.parent / "assets" if args.reuse_assets is None else args.reuse_assets
        config = EggConfig(level=args.level, ordering="column-nd", factor_backend=args.cpu_factor)
        model, asset = None, None
        if args.reuse_assets is not None:
            if args.scope not in ("rigid", "policy-rigid"):
                raise ValueError("Only rigid scopes may omit the elastic operators with exported assets")
            asset = json.loads((assets / "egg_shell.metadata.json").read_text())
            for name, value in asdict(config).items():
                if name not in ("ordering", "factor_backend") and asset["config"][name] != value:
                    raise ValueError(f"Exported assets differ in physical input {name}")
        else:
            model = EggRecoveryCPU(1, config, history=0, direct=True, anchor_to_surface=True)
        gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
        rtol = 1e-6 if args.profile == "strict" else 1e-3
        recovery = None
        if args.scope in ("live", "policy"):
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
                    inertia=args.inertia,
                    body_products=args.body_products,
                    sparse_layout=args.sparse_layout,
                )
            if args.peak == "tensor":
                recovery.peak = P2PeakTensorGPU(
                    model.fem.glambda,
                    model.fem.elements,
                    model.fem.young / (2 * (1 + model.fem.poisson)),
                )
        scene = PandaEggScene(
            model,
            assets,
            PandaConfig(
                environments=args.envs,
                seed=args.seed,
                substeps=args.substeps,
                sampling=args.sampling,
                scatter=args.scatter,
            ),
            recovery,
        )
        policy = None
        if args.scope in ("policy", "policy-rigid"):
            torch.manual_seed(81)
            observation_dimension = scene.observation().shape[1]
            policy = (
                torch.nn.Sequential(
                    torch.nn.Linear(observation_dimension, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 64),
                    torch.nn.Tanh(),
                    torch.nn.Linear(64, 7),
                    torch.nn.Tanh(),
                )
                .to(device=gs.device, dtype=gs.tc_float)
                .eval()
            )
        setup_s = perf_counter() - start
        print(
            json.dumps(
                {"setup_seconds": setup_s, "device_factors": None if recovery is None else factor_metadata(recovery)}
            ),
            flush=True,
        )
        with torch.inference_mode():
            quality = PandaTrajectoryMetrics(scene)
            start = perf_counter()
            warmup_steps = scene.config.episode_steps + int(scene.delay.max()) + 1
            for warmup_tick in range(warmup_steps):
                scene.step(None if policy is None else policy(scene.observation()))
                quality.update()
                if warmup_tick % 32 == 0:
                    print(json.dumps({"warmup_step": warmup_tick}), flush=True)
            torch.cuda.synchronize()
            warmup_s = perf_counter() - start
            quality_report = quality.report()
            for repeat in range(args.repeats):
                scene.restart()
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                accepted = cp.ones(args.envs, dtype=bool)
                relative_max = cp.zeros(args.envs)
                meaningful_relative_max = cp.zeros(args.envs)
                absolute_max = cp.zeros(args.envs)
                peak_max = cp.zeros(args.envs)
                failed, fallback, refinement = (
                    cp.zeros(args.envs, dtype=np.int64),
                    cp.zeros(args.envs, dtype=np.int64),
                    cp.zeros(args.envs, dtype=np.int64),
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_event.record()
                start = perf_counter()
                steps = 0
                # Every environment traverses a whole episode, including independently delayed starts and resets.
                minimum_steps = scene.config.episode_steps + int(scene.delay.max()) + 1
                with DeviceMemorySampler() as memory:
                    while steps < minimum_steps or perf_counter() - start < args.seconds:
                        scene.step(None if policy is None else policy(scene.observation()))
                        if scene.observer is not None:
                            accepted &= scene.observer.is_accepted
                            cp.maximum(relative_max, scene.observer.full_relative_residual_max, out=relative_max)
                            cp.maximum(
                                meaningful_relative_max,
                                scene.observer.meaningful_relative_residual_max,
                                out=meaningful_relative_max,
                            )
                            cp.maximum(absolute_max, scene.observer.absolute_residual_max_n, out=absolute_max)
                            cp.maximum(peak_max, scene.observer.peak_pa, out=peak_max)
                            failed += scene.observer.failed_events
                            fallback += scene.observer.fallback_events
                            refinement += scene.observer.refinement_events
                        steps += 1
                    end_event.record()
                    torch.cuda.synchronize()
                elapsed = perf_counter() - start
                assert cp.all(accepted).item(), "Rejected contact mapping or full equilibrium during timed transition"
                row = {
                    **memory.metadata(),
                    "repeat": repeat,
                    "scope": args.scope,
                    "environments": args.envs,
                    "steps": steps,
                    "environment_transitions": steps * args.envs,
                    "wall_seconds": elapsed,
                    "cuda_span_seconds": start_event.elapsed_time(end_event) / 1000,
                    "environment_transitions_s": steps * args.envs / elapsed,
                    "batch_steps_s": steps / elapsed,
                    "policy_transitions": steps * args.envs if policy is not None else 0,
                    "stress_recoveries": steps * args.envs * args.substeps if recovery is not None else 0,
                    "reset_environments": scene.reset_count,
                    "accepted": True,
                    "full_relative_residual_max_including_near_zero": float(cp.max(relative_max).item()),
                    "full_relative_residual_max_above_absolute_floor": float(cp.max(meaningful_relative_max).item()),
                    "full_absolute_residual_max_n": float(cp.max(absolute_max).item()),
                    "peak_max_pa": float(cp.max(peak_max).item()),
                    "correction_environment_events": int(cp.sum(failed)),
                    "fp64_fallback_events": int(cp.sum(fallback)),
                    "refinement_events": int(cp.sum(refinement)),
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "cupy_pool_total_bytes_after_run": cp.get_default_memory_pool().total_bytes(),
                    "cupy_pool_used_bytes_after_run": cp.get_default_memory_pool().used_bytes(),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        gpu = cp.cuda.runtime.getDeviceProperties(0)
        metadata = {
            "scope": args.scope,
            "profile": args.profile,
            "same_mesh_peak_error_verified_in_timing": False,
            "physical_mesh_converged": False,
            "CPU_or_host_solve_in_timed_loop": False,
            "hardware": {
                **host_metadata(),
                "gpu": gpu["name"].decode(),
                "total_memory_bytes": gpu["totalGlobalMem"],
                "driver_version": cp.cuda.runtime.driverGetVersion(),
                "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            },
            "torch_version": torch.__version__,
            "cupy_version": cp.__version__,
            "quadrants_version": version("quadrants"),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256_at_start": hashes,
            "base_commit": "e9e1214d192914ddec85ca28014c459cfbc6c860",
            "scene": asdict(scene.config),
            "mesh": asdict(config),
            "dofs": model.fem.ndof if model is not None else asset["dofs"],
            "tetrahedra": model.fem.ne if model is not None else asset["tetrahedra"],
            "sampling": "every physical substep; maximum retained across scene step",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "policy": "Fixed FP64 MLP obs->64 tanh->64 tanh->7 tanh; untrained; residual joint amplitude 1e-4 rad",
            "setup_seconds": setup_s,
            "warmup_seconds": warmup_s,
            "warmup_steps": warmup_steps,
            "warmup_quality": quality_report,
            "residual_atol_n": 1e-11,
            "residual_rtol": rtol,
            "mean_environment_transitions_s": sum(row["environment_transitions"] for row in rows)
            / sum(row["wall_seconds"] for row in rows),
            "repeat_rate_std": float(np.std([row["environment_transitions_s"] for row in rows])),
            "runs": rows,
        }
        if recovery is not None:
            metadata["stiffness_nnz"] = recovery.stiffness.nnz
            metadata["device_factors"] = factor_metadata(recovery)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
