"""Untimed same-mesh upper-bound probe for temporal load-subspace reuse."""

import argparse
import json
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    rhs = np.lib.format.open_memmap(
        args.output.with_suffix(".rhs.npy"),
        mode="w+",
        dtype=np.float64,
        shape=(args.steps, args.envs, 3 * entry.model.info.vertices.shape[0]),
    )
    for tick in range(args.steps):
        workload.step()
        rhs[tick] = qd_to_numpy(entry.state.rhs, transpose=True).reshape((args.envs, -1))
    workload.scene.rigid_solver.check_errno()
    rhs.flush()
    rows = []
    # Independent offline NumPy diagnostic, outside production and throughput.
    for history in (0, 1, 4):
        hits = 0
        for tick in range(600, args.steps):
            for i_b in range(args.envs):
                current = rhs[tick, i_b]
                prediction = np.zeros_like(current)
                if history:
                    basis = rhs[max(0, tick - history) : tick, i_b].T
                    prediction = basis @ np.linalg.lstsq(basis, current, rcond=1e-12)[0]
                budget = max(1e-11, 1e-8 * np.linalg.norm(current))
                hits += np.linalg.norm(current - prediction) <= budget
        rows.append({"history": history, "accepted": int(hits), "tested": (args.steps - 600) * args.envs})
    args.output.write_text(
        json.dumps(
            {
                "scope": "Untimed best projection of actual complete RHS onto recent exact RHS columns; an optimistic history acceptance bound, not implemented throughput.",
                "envs": args.envs,
                "steps": args.steps,
                "mesh_level": 1,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
