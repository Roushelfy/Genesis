"""Compare saved-state and individual-setter resets on identical changing trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    saved = FrankaEgg(args.envs, varied=True, seed=args.seed, output_mode="full")
    conditions = args.output.with_suffix(".conditions.npz")
    saved.save_conditions(conditions)
    legacy = FrankaEgg(args.envs, conditions=conditions, output_mode="full", saved_reset=False)
    # Conditions preserve the staggered per-environment schedule in both scenes.
    rows = []
    for tick in range(args.steps):
        saved.step()
        legacy.step()
        if tick % 50 != 49:
            continue
        row = {"tick": tick}
        for name, getter, atol in (
            ("egg_position", lambda w: tensor_to_array(w.egg.get_pos()), 1e-12),
            ("egg_quaternion", lambda w: tensor_to_array(w.egg.get_quat()), 1e-12),
            ("robot_qpos", lambda w: tensor_to_array(w.robot.get_qpos()), 1e-12),
        ):
            a, b = getter(saved), getter(legacy)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=atol)
            row[name + "_max_error"] = float(np.max(np.abs(a - b)))
        left = saved.scene.rigid_solver.stress_recovery.links[0]
        right = legacy.scene.rigid_solver.stress_recovery.links[0]
        for name, getter, atol, rtol in (
            ("force", lambda e: qd_to_numpy(e.state.force), 1e-11, 1e-8),
            ("peak", lambda e: qd_to_numpy(e.state.peak), 1e-3, 1e-4),
            ("tensor", lambda e: qd_to_numpy(e.state.stress_tensor), 1e-3, 1e-4),
            ("actual_mu", lambda e: qd_to_numpy(e.contacts.friction), 1e-14, 1e-14),
        ):
            a, b = getter(left), getter(right)
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
            row[name + "_max_error"] = float(np.max(np.abs(a - b)))
        np.testing.assert_array_equal(qd_to_numpy(left.state.valid), qd_to_numpy(right.state.valid))
        saved.scene.rigid_solver.check_errno()
        legacy.scene.rigid_solver.check_errno()
        rows.append(row)
        print("Equivalent tick", tick, "resets", saved.reset_count, flush=True)
    assert saved.reset_count == legacy.reset_count and saved.reset_count > 0
    args.output.write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "resets": saved.reset_count,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path("examples/rigid/franka_egg_stress.py"))
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
