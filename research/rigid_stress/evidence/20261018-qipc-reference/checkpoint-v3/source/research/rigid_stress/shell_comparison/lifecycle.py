"""Original-PR state restore, partial reset and duplicate-environment diagnostics."""

from dataclasses import dataclass

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .application import ShellEgg
from .common import identity, vars_json, write_json
from .rollout import controller_step, make_parser


@dataclass
class Snapshot:
    positions: np.ndarray
    velocities: np.ndarray
    tensors: np.ndarray


def capture(workload):
    return Snapshot(
        tensor_to_array(workload.egg.get_verts_pos()).copy(),
        tensor_to_array(workload.egg.get_verts_vel()).copy(),
        tensor_to_array(workload.observation.tensors).copy(),
    )


def difference(reference, candidate, envs):
    first, second = reference.tensors[:, envs], candidate.tensors[:, envs]
    denominator = np.linalg.norm(first)
    return {
        "maximum_position_difference_m": float(np.abs(reference.positions[envs] - candidate.positions[envs]).max()),
        "maximum_velocity_difference_m_per_s": float(
            np.abs(reference.velocities[envs] - candidate.velocities[envs]).max()
        ),
        "maximum_tensor_difference_pa": float(np.abs(first - second).max()),
        "tensor_relative_difference": float(np.linalg.norm(first - second) / denominator) if denominator else None,
    }


def main():
    parser = make_parser()
    parser.add_argument("--checkpoint-step", type=int, default=40)
    parser.add_argument("--continuation-steps", type=int, default=40)
    args = parser.parse_args()
    assert args.model == "shell" and not args.instrumented
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    workload = ShellEgg(args)
    for _ in range(args.checkpoint_step):
        controller_step(workload, True)
    saved_state = workload.scene.get_state()
    checkpoint = capture(workload)
    for _ in range(args.continuation_steps):
        controller_step(workload, True)
    continuation = capture(workload)
    rows = []
    payload = {
        "checkpoint_positions": checkpoint.positions,
        "checkpoint_velocities": checkpoint.velocities,
        "checkpoint_tensors": checkpoint.tensors,
        "continuation_positions": continuation.positions,
        "continuation_tensors": continuation.tensors,
    }
    for repeat in range(3):
        workload.scene.reset(state=saved_state)
        workload.tick = args.checkpoint_step
        workload.observation.update(workload.scene.sim.shell_solver)
        rows.append(
            {
                "case": "immediate-public-restore",
                "repeat": repeat,
                **difference(checkpoint, capture(workload), slice(None)),
            }
        )
        for _ in range(args.continuation_steps):
            controller_step(workload, True)
        candidate = capture(workload)
        rows.append(
            {
                "case": "continuation-after-public-restore",
                "repeat": repeat,
                **difference(continuation, candidate, slice(None)),
            }
        )
        payload[f"restore{repeat}_positions"] = candidate.positions
        payload[f"restore{repeat}_tensors"] = candidate.tensors
    if args.envs > 1:
        before = capture(workload)
        workload.scene.reset(state=saved_state, envs_idx=[0])
        workload.observation.update(workload.scene.sim.shell_solver)
        after = capture(workload)
        rows.append({"case": "partial-restore-unselected-environments", **difference(before, after, slice(1, None))})
        rows.append({"case": "partial-restore-selected-environment", **difference(checkpoint, after, slice(0, 1))})
    if args.envs >= 16:
        # The frozen bank repeats exactly every eight environments.
        first = Snapshot(checkpoint.positions[:8], checkpoint.velocities[:8], checkpoint.tensors[:, :8])
        second = Snapshot(checkpoint.positions[8:16], checkpoint.velocities[8:16], checkpoint.tensors[:, 8:16])
        rows.append(
            {"case": "duplicate-inputs-within-batch-after-checkpoint", **difference(first, second, slice(None))}
        )
    np.savez_compressed(args.output / "states.npz", **payload)
    write_json(
        args.output / "result.json",
        {
            "kind": "original-shell-lifecycle",
            "args": vars_json(args),
            "identity": identity(),
            "checks": rows,
            "note": "Public state/restore tested as provided. No warm-start/coarse repair is applied.",
        },
    )


if __name__ == "__main__":
    main()
