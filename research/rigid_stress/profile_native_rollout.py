"""Intrusive CUDA-stream intervals across the unmodified changing-contact rollout."""

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.engine.solvers.rigid.rigid_solver as rigid_module
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy


class Timeline:
    def __init__(self, workload):
        self.workload = workload
        self.records = []

    def wrap(self, function, label, *, reset=False):
        def call(*args, **kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            tick = self.workload.tick
            reset_envs = 0
            if reset:
                ids = args[0] if args else kwargs.get("envs_idx")
                reset_envs = self.workload.n_envs if ids is None else len(ids)
            begin.record()
            started = time.perf_counter()
            with torch.profiler.record_function(label):
                result = function(*args, **kwargs)
            host_ms = 1e3 * (time.perf_counter() - started)
            end.record()
            self.records.append((tick, label, host_ms, reset_envs, begin, end))
            return result

        return call

    def export(self):
        qd.sync()
        return [
            {
                "tick": tick,
                "stage": label,
                "host_submission_ms": host_ms,
                "cuda_stream_interval_ms": begin.elapsed_time(end),
                "reset_envs": reset_envs,
            }
            for tick, label, host_ms, reset_envs, begin, end in self.records
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("live", "policy"), required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=623001)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, output_mode=args.output_mode)
    torch.manual_seed(99173)
    reference = torch.nn.Sequential(
        torch.nn.Linear(26, 128), torch.nn.Tanh(), torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 7)
    ).to(device=gs.device, dtype=torch.float64)
    policy = copy.deepcopy(reference).to(dtype=torch.float32).eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            workload.step(policy(workload.observation().to(dtype=torch.float32)) if args.scope == "policy" else None)
        workload.restart()
        qd.sync()
        assert torch.cuda.current_stream(gs.device) == torch.cuda.default_stream(gs.device)
        timeline = Timeline(workload)
        solver = workload.scene.rigid_solver
        recovery = solver.stress_recovery
        for owner, method, label in (
            (workload, "observation", "observation"),
            (policy, "forward", "policy_inference"),
            (workload.link, "set_stress_contact_radius", "contact_parameter_update"),
            (recovery, "recover", "native_stress_recovery"),
            (rigid_module, "kernel_resolve_stress_contacts", "rigid_contact_force_postprocess"),
            (rigid_module, "kernel_step_2", "rigid_integration_after_stress"),
            (solver, "substep", "rigid_substep_including_recovery"),
            (workload.scene, "step", "scene_step"),
            (workload, "step", "controller_and_scene_step"),
        ):
            setattr(owner, method, timeline.wrap(getattr(owner, method), label))
        workload.reset = timeline.wrap(workload.reset, "partial_reset", reset=True)
        failures_before = qd_to_numpy(recovery.links[0].state.invalid_steps, copy=True)
        profiler = None
        if args.trace:
            profiler = torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
            )
            profiler.start()
        for step in range(args.steps):
            action = policy(workload.observation().to(dtype=torch.float32)) if args.scope == "policy" else None
            workload.step(action)
            if step % 64 == 63:
                qd.sync()
        qd.sync()
        if profiler is not None:
            profiler.stop()
            profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        records = timeline.export()
        failures = qd_to_numpy(recovery.links[0].state.invalid_steps) - failures_before
        solver.check_errno()
    stages = sorted({row["stage"] for row in records})
    summary = {}
    for stage in stages:
        selected = [row for row in records if row["stage"] == stage]
        summary[stage] = {
            "calls": len(selected),
            "cuda_stream_interval_mean_ms": float(np.mean([row["cuda_stream_interval_ms"] for row in selected])),
            "cuda_stream_interval_p50_p90_p99_max_ms": np.quantile(
                [row["cuda_stream_interval_ms"] for row in selected], [0.5, 0.9, 0.99, 1.0]
            ).tolist(),
            "host_submission_mean_ms": float(np.mean([row["host_submission_ms"] for row in selected])),
            "reset_envs_sum": sum(row["reset_envs"] for row in selected),
        }
    result = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summary,
        "records": records,
        "failed_steps_per_environment": failures.tolist(),
        "invalid_environment_steps": int(failures.sum()),
        "contact_delays": workload.delays.tolist(),
        "source_revision": os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                    Path("genesis/engine/solvers/rigid/rigid_solver.py"),
                    Path("genesis/options/rigid_stress.py"),
                    Path("examples/rigid/franka_egg_stress.py"),
                    Path("examples/rigid/egg_stress_controller.py"),
                    Path(__file__),
                ]
            )
        },
        "command": [sys.executable, *sys.argv],
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scope_note": "Intrusive timing of the unchanged live native functions. CUDA default-stream intervals include idle gaps from host preparation and launch; nested stages overlap and must not be added. Synchronization and event/profiler overhead make these diagnostics, not throughput measurements. Policy is FP32 inference, not RL training. Frozen-snapshot stage subdivision is separately provided by the native benchmark.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
