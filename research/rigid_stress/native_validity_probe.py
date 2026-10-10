"""Untimed every-step validity audit across warmup and a complete reset trajectory."""

import argparse
import json
from pathlib import Path

import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_audit(state: StressState, failures: qd.Tensor):
    for i_b in range(failures.shape[0]):
        if not state.step_valid[i_b]:
            failures[i_b] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("live", "policy"), required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True, conditions=args.conditions)
    torch.manual_seed(99173)
    policy = (
        torch.nn.Sequential(
            torch.nn.Linear(26, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 7),
        )
        .to(device=gs.device, dtype=gs.tc_float)
        .eval()
    )
    failures = V(dtype=gs.qd_int, shape=(args.envs,))
    failures.fill(0)
    state = workload.scene.rigid_solver.stress_recovery.links[0].state
    with torch.inference_mode():
        for count in (args.warmup, args.steps):
            for _ in range(count):
                action = policy(workload.observation()) if args.scope == "policy" else None
                workload.step(action)
                kernel_audit(state, failures)
            workload.restart()
    counts = qd_to_numpy(failures)
    result = {
        "scope": args.scope,
        "envs": args.envs,
        "warmup_steps": args.warmup,
        "trajectory_steps": args.steps,
        "seed": args.seed,
        "load_model": "finite_pad_adaptive_q10",
        "condition_count": workload.condition_count,
        "conditions": str(args.conditions) if args.conditions is not None else None,
        "every_step_failures": counts.tolist(),
        "failed_environment_steps": int(counts.sum()),
        "note": "Separate untimed native check of every observation, surviving per-environment resets; no rate claimed.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert not counts.any(), "A reset trajectory contained invalid stress observations."
    print("All", args.envs * (args.warmup + args.steps), "environment observations valid.")


if __name__ == "__main__":
    main()
