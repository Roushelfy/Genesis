"""Separate changes in actual centroid loads from changes in sparse source packets."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def load(path):
    with np.load(path) as saved:
        points = saved["original_contact_points_m"]
        forces = saved["original_contact_forces_n"]
        sources = saved["source_points_m"]
        values = saved["source_forces_n"]
    lookup = {tuple(point): index for index, point in enumerate(points)}
    indices = np.array([lookup[tuple(point)] for point in sources])
    reduced = np.zeros_like(forces)
    np.add.at(reduced, indices, values)
    with np.load(path.with_name(path.name.removeprefix("stress-"))) as saved:
        rotation = saved["rotation"]
        velocity = saved["velocity_m_per_s"]
        angular = saved["angular_velocity_rad_per_s"]
    return points, forces, indices, reduced, rotation, velocity, angular


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--frames", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    points, force, indices, reduced, rotation, velocity, angular = load(args.reference)
    reference_parameters = json.loads((args.reference.parent.parent / "start.json").read_text())["args"]
    cases = []
    for path in args.frames:
        actual = load(path)
        np.testing.assert_array_equal(actual[0], points)
        cases.append(
            {
                "field": file_id(path),
                "changed_parameters": {
                    key: [reference_parameters.get(key), value]
                    for key, value in json.loads((path.parent.parent / "start.json").read_text())["args"].items()
                    if key not in ("output", "loads") and value != reference_parameters.get(key)
                },
                "original_force_relative_difference": float(np.linalg.norm(actual[1] - force) / np.linalg.norm(force)),
                "compressed_force_relative_difference": float(
                    np.linalg.norm(actual[3] - reduced) / np.linalg.norm(reduced)
                ),
                "reference_source_indices": indices.tolist(),
                "actual_source_indices": actual[2].tolist(),
                "rotation_difference_norm": float(np.linalg.norm(actual[4] - rotation)),
                "velocity_difference_m_per_s": (actual[5] - velocity).tolist(),
                "angular_velocity_difference_rad_per_s": (actual[6] - angular).tolist(),
            }
        )
    write_json(
        args.output / "result.json",
        {"identity": runtime_identity(), "reference": file_id(args.reference), "cases": cases},
    )


if __name__ == "__main__":
    main()
