"""Author small shell assets with the unchanged hollow-egg collision exterior."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def make_asset(source, destination):
    metadata = json.loads((source / "egg_shell.metadata.json").read_text())
    layers = metadata["config"]["layers"]
    thickness = metadata["config"]["thickness_m"]
    with np.load(source / "elastic.npz") as asset:
        vertices = asset["vertices"]
    n_original = len(vertices) // (layers + 1)
    outer = vertices[:n_original]
    normals = (outer - vertices[layers * n_original :]) / thickness
    faces = []
    for line in (source / "egg_outer.obj").read_text().splitlines():
        if line.startswith("f "):
            faces.append([int(word.split("/")[0]) - 1 for word in line.split()[1:]])
    faces = np.array(faces)
    assert faces.shape[1] == 3 and faces.max() < n_original
    edges, edge_indices = np.unique(
        np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1), axis=0, return_inverse=True
    )
    edge_indices = edge_indices.reshape((-1, 3)) + n_original
    outer = np.concatenate((outer, outer[edges].mean(axis=1), outer[faces].mean(axis=1)))
    normals = np.concatenate((normals, normals[edges].mean(axis=1), normals[faces].mean(axis=1)))
    interpolated_length = np.linalg.norm(normals, axis=1)
    normals /= interpolated_length[:, None]
    centers = n_original + len(edges) + np.arange(len(faces))
    quads = np.stack(
        [np.column_stack((faces[:, i], edge_indices[:, i], centers, edge_indices[:, (i + 2) % 3])) for i in range(3)],
        axis=1,
    ).reshape((-1, 4))
    midsurface = outer - 0.5 * thickness * normals
    np.savez(destination, vertices=midsurface, quads=quads, directors=normals)
    return {
        "source": file_id(source / "elastic.npz"),
        "collision": file_id(source / "egg_outer.obj"),
        "asset": file_id(destination),
        "quads": len(quads),
        "dofs": 6 * len(midsurface),
        "thickness_m": thickness,
        "outer_geometry": "Each original collision triangle is subdivided into three quads; outer position is unchanged.",
        "maximum_inner_difference_from_volume_m": float(np.max(thickness * (1.0 - interpolated_length))),
        "inner_geometry": "Unit nodal directors retain the declared wall thickness; new edge/center nodes differ from the volume's linearly interpolated inner surface.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = [make_asset(args.source / f"level{level}", args.output / f"level{level}-shell.npz") for level in (1, 2)]
    write_json(
        args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False), "results": results}
    )


if __name__ == "__main__":
    main()
