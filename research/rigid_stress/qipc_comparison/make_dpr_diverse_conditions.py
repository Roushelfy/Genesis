"""Save a small diverse Franka bank with overlapping, staggered approach phases."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=510127)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    workload = FrankaEgg(args.count, stress=False, seed=args.seed, varied=True)
    # This instance only generates conditions; simulation constructs fresh controller buffers from the saved bank.
    workload.delays[:] = 3 * np.arange(args.count)
    path = args.output / "conditions.npz"
    workload.save_conditions(path)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=True),
            "genesis_module": gs.__file__,
            "seed": args.seed,
            "count": args.count,
            "conditions": file_id(path),
            "position_m": tensor_to_array(workload.initial_position).tolist(),
            "quaternion": tensor_to_array(workload.initial_quaternion).tolist(),
            "friction_ratio": tensor_to_array(workload.friction).tolist(),
            "manual_radius_m": tensor_to_array(workload.base_radius).tolist(),
            "delay_steps": workload.delays.tolist(),
            "scope": "Independent seeded positions, yaw, friction and manual radii; fresh IK targets. No filtering by stress admission.",
        },
    )


if __name__ == "__main__":
    main()
