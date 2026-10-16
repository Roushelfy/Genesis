"""Capture the first rejected actual low-mesh native contact without altering it."""

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    workload = FrankaEgg(args.envs, seed=args.seed, varied=True)
    failure = None
    tick = 0
    phase = "trajectory"
    for phase, count in (("warmup", args.warmup), ("trajectory", args.steps)):
        for tick in range(count):
            try:
                workload.step()
            except gs.GenesisException as error:
                failure = str(error)
                break
        if failure is not None:
            break
        if phase == "warmup" and args.warmup:
            workload.restart()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    status = qd_to_numpy(entry.contacts.status, transpose=True)
    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
    cases = []
    arrays = {
        name: qd_to_numpy(value, transpose=True)
        for name, value in (
            ("position", entry.contacts.position),
            ("center", entry.contacts.center),
            ("anchor_face", entry.contacts.anchor_face),
            ("is_refined", entry.contacts.is_refined),
            ("candidate_count", entry.contacts.candidate_count),
            ("weight_sum", entry.contacts.weight_sum),
            ("force", entry.contacts.force),
            ("normal", entry.contacts.normal),
            ("friction", entry.contacts.friction),
            ("radius", entry.contacts.radius),
            ("coefficient", entry.contacts.coefficient),
            ("evaluations", entry.contacts.evaluations),
            ("force_error", entry.contacts.force_error),
            ("moment_error", entry.contacts.moment_error),
        )
    }
    for i_b, i_c in np.argwhere(valid & (status > 0)):
        cases.append({"env": int(i_b), "slot": int(i_c), "status": int(status[i_b, i_c])})
        cases[-1].update({name: value[i_b, i_c].tolist() for name, value in arrays.items()})
    args.output.write_text(
        json.dumps(
            {"envs": args.envs, "seed": args.seed, "phase": phase, "tick": tick, "error": failure, "cases": cases},
            indent=2,
        )
        + "\n"
    )
    print("Captured", len(cases), "rejected contacts at tick", tick)


if __name__ == "__main__":
    main()
