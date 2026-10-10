"""Intrusive recovery cost on changing live contacts, including stress reset/radius work.

Rigid dynamics drive every measured sample. Synchronizing around each native
recovery separates its cost from preceding rigid work; these stage rates are
diagnostic and are not the ordinary unsynchronized live rollout throughput.
"""

import argparse
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
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, output_mode=args.output_mode)
    recovery = workload.scene.rigid_solver.stress_recovery
    durations = {"recover": [], "reset": [], "set_radius": []}
    recording = False

    def wrap(name):
        original = getattr(recovery, name)

        def measured(*call_args, **kwargs):
            if not recording:
                return original(*call_args, **kwargs)
            qd.sync()
            started = time.perf_counter()
            result = original(*call_args, **kwargs)
            qd.sync()
            durations[name].append(time.perf_counter() - started)
            return result

        return measured

    for name in durations:
        setattr(recovery, name, wrap(name))
    recovery.subscriber.callback = recovery.reset
    with torch.inference_mode():
        for _ in range(args.warmup):
            workload.step()
        workload.scene.rigid_solver.check_errno()
        repeats = []
        for _ in range(args.repetitions):
            workload.restart()
            qd.sync()
            for values in durations.values():
                values.clear()
            recording = True
            started = time.perf_counter()
            for _ in range(args.steps):
                workload.step()
            qd.sync()
            wall_seconds = time.perf_counter() - started
            recording = False
            workload.scene.rigid_solver.check_errno()
            assert np.isfinite(tensor_to_array(workload.link.get_max_stress())).all()
            stages = {
                name: {
                    "calls": len(values),
                    "seconds": sum(values),
                    "call_seconds_quantiles": np.quantile(values, [0.5, 0.9, 0.99, 1.0]).tolist() if values else [],
                    "call_seconds": values.copy(),
                }
                for name, values in durations.items()
            }
            recovery_seconds = sum(stage["seconds"] for stage in stages.values())
            repeats.append(
                {
                    "steps": args.steps,
                    "resets": workload.reset_count,
                    "instrumented_live_seconds": wall_seconds,
                    "recovery_with_updates_seconds": recovery_seconds,
                    "diagnostic_recovery_env_steps_per_second": args.envs * args.steps / recovery_seconds,
                    "diagnostic_recovery_batch_steps_per_second": args.steps / recovery_seconds,
                    "stages": stages,
                }
            )
    result = {
        "scope": "synchronized recovery on live changing trajectory",
        "note": "Includes native contact association, pressure retries, complete solve/check/peak, stress reset and radius writes. Rigid/controller/reset motion costs are in instrumented_live_seconds, outside the recovery stage rate. Every-step synchronization is intrusive; use the ordinary live/policy benchmark for actual aggregate throughput.",
        "envs": args.envs,
        "seed": args.seed,
        "warmup": args.warmup,
        "steps": args.steps,
        "output_mode": args.output_mode,
        "load_model": "finite_pad_adaptive_q10",
        "source_revision": os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                    Path("genesis/options/rigid_stress.py"),
                    Path("examples/rigid/franka_egg_stress.py"),
                    Path(__file__),
                ]
            )
        },
        "command": [sys.executable, *sys.argv],
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
        "quadrants": qd.__version__,
        "torch": torch.__version__,
        "repeats": repeats,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Completed synchronized changing-contact recovery profile", args.envs, args.output_mode)


if __name__ == "__main__":
    main()
