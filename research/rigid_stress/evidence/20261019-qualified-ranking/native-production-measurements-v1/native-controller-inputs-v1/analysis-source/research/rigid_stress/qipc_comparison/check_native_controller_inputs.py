"""Compare actual controller outputs with stress-radius emission off/on, without dynamics."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.egg_stress_controller import kernel_inputs
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", seed=623016, logging_level="warning")
    with np.load(args.conditions) as bank:
        initial, pick, lift, slide = (
            torch.as_tensor(bank[k], device=gs.device)
            for k in ("initial_joints", "pick_joints", "lift_joints", "slide_joints")
        )
        radius = torch.as_tensor(bank["radius"], dtype=gs.tc_float, device=gs.device)
        delays = torch.as_tensor(bank["delays"], dtype=gs.tc_float, device=gs.device)
    batch = len(delays)
    arm = torch.arange(7, dtype=gs.tc_int, device=gs.device)
    zero_action = torch.zeros((batch, 7), dtype=gs.tc_float, device=gs.device)
    histories = []
    for stress in (False, True):
        phase = torch.zeros_like(radius)
        target = torch.empty_like(initial)
        grip = torch.empty(batch, device=gs.device, dtype=torch.get_default_dtype())
        limit = torch.empty_like(grip)
        output_radius = torch.empty_like(radius)
        history = initial.new_empty((1200, batch, initial.shape[1] + 3))
        for tick in range(1200):
            kernel_inputs(
                tick,
                delays,
                initial,
                pick,
                lift,
                slide,
                radius,
                arm,
                zero_action,
                phase,
                target,
                grip,
                limit,
                output_radius,
                False,
                False,
                stress,
            )
            history[tick].copy_(torch.cat((target, phase[:, None], grip[:, None], limit[:, None]), dim=1))
        histories.append(history)
    qd.sync()
    first, second = (tensor_to_array(t) for t in histories)
    delta = np.abs(first - second)
    rows = []
    for name, start, end in (("joint_target", 0, 9), ("phase", 9, 10), ("grip", 10, 11), ("force_limit", 11, 12)):
        difference = delta[..., start:end]
        nonzero = np.argwhere(difference != 0)
        rows.append(
            {
                "field": name,
                "maximum_absolute_difference": float(difference.max()),
                "different_values": len(nonzero),
                "first_different_index": nonzero[0].tolist() if len(nonzero) else None,
            }
        )
    np.savez_compressed(args.output / "inputs.npz", stress_disabled=first, stress_enabled=second)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "check_completed": True,
            "bitwise_identical": bool((delta == 0).all()),
            "rows": rows,
            "conditions": file_id(args.conditions),
            "inputs": file_id(args.output / "inputs.npz"),
            "scope": "Exact1200-step controller outputs for the frozen eight-condition bank; only the existing stress template differs. No dynamics or recovery runs in this control.",
        },
    )


if __name__ == "__main__":
    main()
