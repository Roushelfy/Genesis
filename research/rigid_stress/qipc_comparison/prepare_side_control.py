"""Rotate the existing full free egg into a declared horizontal-long-axis plate control."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--shell-rest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    with np.load(args.mesh) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key in ("vertices", "mid_vertices", "outer_vertices", "inner_vertices"):
        arrays[key] = arrays[key] @ rotation.T
    np.savez_compressed(args.output / "solid-side.npz", **arrays)
    with np.load(args.shell_rest) as saved:
        mid, faces = saved["vertices"], saved["faces"]
    trimesh.Trimesh(mid @ rotation.T, faces, process=False).export(args.output / "shell-side.obj", digits=17)
    write_json(
        args.output / "result.json",
        {
            "source_mesh": file_id(args.mesh),
            "source_shell_rest": file_id(args.shell_rest),
            "rotation": rotation.tolist(),
            "solid": file_id(args.output / "solid-side.npz"),
            "shell": file_id(args.output / "shell-side.obj"),
            "scope": "Explicit additional orientation control: same complete geometry, wall, material and mass, long axis horizontal; free egg and unchanged force-controlled plates. Original upright history remains separate.",
        },
    )


if __name__ == "__main__":
    main()
