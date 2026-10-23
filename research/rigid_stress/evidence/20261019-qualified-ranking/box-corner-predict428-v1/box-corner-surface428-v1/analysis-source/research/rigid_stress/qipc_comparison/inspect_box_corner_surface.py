"""Inspect discrete surface orientation and ordering of the two-face elastic controls."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def inspect(path):
    with np.load(path) as saved:
        points = saved["points_m"]
        position = saved["deformed_relative_m"]
        displacement = saved["displacement_m"]
        labels, force = saved["labels"], saved["normal_force_n"]
        rest = saved["rest_m"]
    x, y = np.unique(points[:, 0]), np.unique(points[:, 1])
    assert len(x) * len(y) == len(points)
    nx, ny = len(x), len(y)
    position, displacement, rest = [a.reshape(nx, ny, 3) for a in [position, displacement, rest]]
    force, labels = force.reshape(nx, ny), labels.reshape(nx, ny)
    active = force > 1e-10 * force.sum()
    pair_active = active[:-1] & active[1:]
    increments = np.diff(position, axis=0)
    is_reversed = pair_active & (increments[..., 0] <= 0)
    dx = np.diff(x)[:, None]
    longitudinal_rows = []
    for j in range(ny):
        index = np.flatnonzero(active[:, j])
        if len(index) < 2:
            continue
        current = labels[index, j]
        transition = np.diff(current)
        longitudinal_rows.append(
            {
                "y_m": float(y[j]),
                "active_cells": len(index),
                "normal_face_changes": int(np.count_nonzero(transition)),
                "reversed_face_changes": int(np.count_nonzero(transition > 0)),
                "minimum_loaded_x_increment_m": float(np.diff(position[index, j, 0]).min()),
            }
        )
    tangent_x = np.gradient(position, x, axis=0, edge_order=2)
    tangent_y = np.gradient(position, y, axis=1, edge_order=2)
    normal = np.cross(tangent_x, tangent_y)
    rest_normal = np.cross(np.gradient(rest, x, axis=0, edge_order=2), np.gradient(rest, y, axis=1, edge_order=2))
    stretch = np.linalg.norm(normal, axis=-1) / np.linalg.norm(rest_normal, axis=-1)
    reversed_orientation = normal[..., 2] <= 0
    gradient_x = np.linalg.norm(np.gradient(displacement, x, axis=0, edge_order=2), axis=-1)
    pair_weight = (force[:-1] + force[1:]) / 2
    examples = []
    for i, j in np.argwhere(is_reversed)[:12]:
        examples.append(
            {
                "material_x_m": [float(x[i]), float(x[i + 1])],
                "material_y_m": float(y[j]),
                "deformed_relative_m": position[i : i + 2, j].tolist(),
                "faces": labels[i : i + 2, j].tolist(),
                "normal_forces_n": force[i : i + 2, j].tolist(),
            }
        )
    return {
        "input": file_id(path),
        "grid": [nx, ny],
        "active_cells": int(active.sum()),
        "adjacent_loaded_pairs": int(pair_active.sum()),
        "reversed_adjacent_loaded_pairs": int(is_reversed.sum()),
        "minimum_adjacent_loaded_x_ratio": float((increments[..., 0] / dx)[pair_active].min()),
        "reversed_pair_force_weight_fraction": float(pair_weight[is_reversed].sum() / pair_weight[pair_active].sum()),
        "negative_projected_orientation_active_cells": int(np.count_nonzero(active & reversed_orientation)),
        "negative_projected_orientation_force_fraction": float(
            force[active & reversed_orientation].sum() / force.sum()
        ),
        "maximum_active_displacement_x_gradient": float(gradient_x[active].max()),
        "active_surface_area_ratio_range": [float(stretch[active].min()), float(stretch[active].max())],
        "longitudinal_rows": longitudinal_rows,
        "reversed_pair_examples": examples,
        "scope": "Deformed cell-average positions and nonuniform-grid finite differences. Both rigid face tangents have positive material-chart x projection in the intended single crossing of this convex corner. Negative increments between adjacent loaded cells or reversed face order contradict that discrete orientation. Gradient and area estimates depend on resolution and are diagnostics, rather than pointwise continuum deformation-gradient bounds.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fields", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=False), "inputs": [file_id(p) for p in args.fields]},
    )
    rows = [inspect(path) for path in args.fields]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "scope": "Orientation and ordering audit of existing frictionless local reference fields. Numerical contact residuals and cell-mean finite-face membership are separate from injective, small-gradient surface geometry.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
