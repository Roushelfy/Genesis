"""Measure Q4 corner angles and edge ratios on captured generated assets, independently of solver comparisons."""

import argparse
from pathlib import Path

import numpy as np

from research.rigid_stress.qipc_comparison.artifacts import file_id, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.asset:
        with np.load(path) as asset:
            vertices, quads = asset["vertices"], asset["quads"]
        points = vertices[quads]
        edges = np.roll(points, -1, axis=1) - points
        length = np.linalg.norm(edges, axis=-1)
        previous = np.roll(edges, 1, axis=1)
        norm = length * np.roll(length, 1, axis=1)
        angles = np.degrees(np.arccos(np.clip(-np.sum(previous * edges, axis=-1) / norm, -1, 1)))
        sine = np.linalg.norm(np.cross(previous, edges), axis=-1) / norm
        rows.append(
            {
                "asset": file_id(path),
                "nodes": len(vertices),
                "quads": len(quads),
                "minimum_corner_angle_degrees": float(angles.min()),
                "maximum_corner_angle_degrees": float(angles.max()),
                "minimum_corner_sine": float(sine.min()),
                "maximum_element_edge_ratio": float((length.max(axis=1) / length.min(axis=1)).max()),
            }
        )
    write_json(args.output / "quality.json", rows)


if __name__ == "__main__":
    main()
