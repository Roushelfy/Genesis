"""Capture an actual observer input before a costly failing recovery call."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import plate_native_pad
from .artifacts import file_id, runtime_identity, write_json
from .native_pad_recovery import NativePadRecovery


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", type=Path, required=True)
    cli.add_argument("--observation", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    task = json.loads(args.task.read_text())
    assert task["module"] == plate_native_pad.__name__ and args.observation > 0
    assert "--output" not in task["args"] and "--primary-cost" in task["args"]
    args.output.mkdir(parents=True, exist_ok=False)
    original = NativePadRecovery.sample
    count, previous_force = 0, None

    def sample(observer, force_world, rotation, angular_world, friction, load, path=None):
        nonlocal count, previous_force
        count += 1
        if count == args.observation:
            force = force_world @ rotation
            selected = np.flatnonzero(np.any(force != 0, axis=1))
            normal = np.zeros_like(force_world)
            normal[:, 2] = np.where(observer.points[:, 2] > 0, -1.0, 1.0)
            capture = args.output / "input.npz"
            np.savez_compressed(
                capture,
                source_points_m=observer.points[selected],
                source_forces_n=force[selected],
                source_normals_material=(normal @ rotation)[selected],
                original_contact_forces_world_n=force_world,
                previous_contact_forces_world_n=previous_force if previous_force is not None else np.zeros_like(force),
                rotation_material_to_world=rotation,
                angular_velocity_world_rad_per_s=angular_world,
                friction=friction,
                load_n=load,
                observation=count,
            )
            # Written after the immutable input file is complete; downstream
            # diagnostics may inspect this leaf while the recovery is running.
            write_json(
                args.output / "input-ready.json",
                {
                    "identity": runtime_identity(),
                    "task": file_id(args.task),
                    "input": file_id(capture),
                    "observation": count,
                    "primary_timing": False,
                    "scope": "Exact caller-owned force/state input captured before the requested observer call. Physical dynamics and recovery arithmetic are unchanged. This intrusive diagnostic cannot supply primary timings or admit the unfinished trajectory.",
                },
            )
        result = original(observer, force_world, rotation, angular_world, friction, load, path)
        previous_force = force_world.copy()
        return result

    previous_argv = sys.argv
    NativePadRecovery.sample = sample
    try:
        sys.argv = [
            task["module"],
            *task["args"],
            "--intrusive-phase-profile",
            "--output",
            str(args.output / "run"),
        ]
        plate_native_pad.main()
    finally:
        NativePadRecovery.sample = original
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
