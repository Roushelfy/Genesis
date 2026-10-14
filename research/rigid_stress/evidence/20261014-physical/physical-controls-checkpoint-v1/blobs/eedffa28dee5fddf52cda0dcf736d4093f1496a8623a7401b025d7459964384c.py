"""Matched public reset costs and state equality across selection sizes."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.utils.misc import tensor_to_array
from research.rigid_stress.probe_native_reset import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=623001, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=623001, varied=True)
    for _ in range(args.warmup):
        workload.step()
    workload.scene.rigid_solver.check_errno()
    original = RigidSolver.set_state
    configuration = configure(True, False, args.output.with_suffix(".set-state.py"))
    native = RigidSolver.set_state
    sizes = sorted({1, 2, 8, 32, max(1, args.envs // 600), args.envs // 16, args.envs // 4, args.envs})
    rows = []
    for size in sizes:
        selected = np.linspace(0, args.envs - 1, size, dtype=np.int64)
        states = []
        for name, method in (("original", original), ("native", native)):
            RigidSolver.set_state = method
            for _ in range(5):
                workload.reset(selected)
            samples, events = [], []
            for _ in range(args.repetitions):
                qd.sync()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                started = time.perf_counter()
                workload.reset(selected)
                end.record()
                qd.sync()
                samples.append(1e3 * (time.perf_counter() - started))
                events.append(begin.elapsed_time(end))
            state = workload.scene.rigid_solver.get_state(0)
            states.append(
                {
                    key: tensor_to_array(getattr(state, key)).copy()
                    for key in ("qpos", "dofs_vel", "dofs_acc", "links_pos", "links_quat", "friction_ratio")
                }
            )
            row = {
                "route": name,
                "selected": size,
                "envs": args.envs,
                "wall_mean_ms": float(np.mean(samples)),
                "wall_p50_p90_p99_max_ms": np.quantile(samples, [0.5, 0.9, 0.99, 1.0]).tolist(),
                "cuda_interval_mean_ms": float(np.mean(events)),
                "wall_samples_ms": samples,
                "cuda_interval_samples_ms": events,
            }
            rows.append(row)
            print(name, size, row["wall_mean_ms"], flush=True)
        for key in states[0]:
            np.testing.assert_array_equal(states[0][key], states[1][key])
    RigidSolver.set_state = original
    args.output.write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "configuration": configuration,
                "rows": rows,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        Path("research/rigid_stress/probe_native_reset.py"),
                        Path("genesis/engine/solvers/rigid/rigid_solver.py"),
                        Path("examples/rigid/franka_egg_stress.py"),
                    )
                },
                "gpu": torch.cuda.get_device_name(),
                "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
                "note": "Intrusive public reset cost on identical initialized scene/state. Includes cache clearing, state copy, forward kinematics, stress invalidation, simulator and scene restart. Selected and untouched rigid states must match exactly. This is not ordinary rollout throughput; actual trajectory pairs determine retention.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
