"""Prepare fixed device-load inputs for a common prescribed-load accuracy/cost experiment."""

import argparse
from pathlib import Path

import numpy as np

from ..skfem_reference import SkfemElasticReference
from .common import file_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    for level in (1, 2):
        path = args.inputs / f"egg-level{level}.npz"
        with np.load(path) as saved:
            vertices, tets = saved["vertices"], saved["tetrahedra"]
        with np.load(args.loads / f"level{level}.npz") as saved:
            force, shell_force = saved["volume_force_n"], saved["shell_force_n"]
        reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000.0)
        edge_vertices = np.unique(np.sort(tets[:, edges].reshape(-1, 2), axis=1), axis=0)
        coordinates = np.concatenate((vertices, vertices[edge_vertices].mean(axis=1)))
        permutation = reference.native_dof_permutation(coordinates)
        np.savez_compressed(
            args.output / f"level{level}.npz",
            native_coordinates_m=coordinates,
            native_force_n=force[permutation].reshape(-1, 3),
            shell_force_n=shell_force,
        )
        rows.append(
            {
                "level": level,
                "mesh": file_identity(path),
                "load": file_identity(args.loads / f"level{level}.npz"),
                "native_nodes": len(coordinates),
                "shell_nodes": len(shell_force),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "kind": "static-cost-inputs",
            "cases": rows,
            "scope": "Same integrated finite compression, topology mapping only; no fitted load or stress surrogate.",
        },
    )


if __name__ == "__main__":
    main()
