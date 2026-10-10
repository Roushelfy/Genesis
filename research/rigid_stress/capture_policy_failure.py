"""Capture arbitrary changing policy contacts, including a focused environment reproduction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=49152)
    parser.add_argument("--focused-env", type=int)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    focused = args.focused_env is not None
    workload = FrankaEgg(1 if focused else args.envs, seed=args.seed + (args.focused_env or 0), varied=True)
    if focused:
        workload.delays[:] = (args.focused_env * 37) % 600
        workload.device_delays[:] = torch.as_tensor(workload.delays, device=gs.device)
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
    failure = None
    phase, tick = "initial", 0
    with torch.inference_mode():
        for phase, count in [("warmup", args.warmup), *[(f"repeat-{i}", args.steps) for i in range(args.repetitions)]]:
            if phase != "warmup":
                workload.restart()
            for tick in range(count):
                try:
                    workload.step(policy(workload.observation()))
                except gs.GenesisException as error:
                    failure = str(error)
                    break
                if tick % 100 == 99:
                    print(phase, tick, flush=True)
            if failure is not None:
                break
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    status = qd_to_numpy(entry.contacts.status, transpose=True)
    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
    arrays = {
        name: qd_to_numpy(getattr(entry.contacts, name), transpose=True)
        for name in (
            "position",
            "center",
            "anchor_face",
            "is_refined",
            "candidate_count",
            "weight_sum",
            "force",
            "normal",
            "friction",
            "radius",
            "coefficient",
            "evaluations",
            "force_error",
            "moment_error",
        )
    }
    cases = []
    for env, contact in np.argwhere(valid & (status > 0)):
        cases.append({"env": int(env), "slot": int(contact), "status": int(status[env, contact])})
        cases[-1].update({name: value[env, contact].tolist() for name, value in arrays.items()})
    args.output.write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "phase": phase,
                "tick": tick,
                "error": failure,
                "cases": cases,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [
                        *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                        Path(__file__),
                        Path("examples/rigid/franka_egg_stress.py"),
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("Captured", len(cases), "rejected contacts", phase, tick, flush=True)


if __name__ == "__main__":
    main()
