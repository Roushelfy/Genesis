"""Diagnose failed material query locations without moving or discarding them."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skfem import MeshTet

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sampling", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.sampling), file_id(args.mesh)],
            "scope": "Locate original query refusals and inspect candidate barycentrics; no admission change.",
        },
    )
    with np.load(args.mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    mesh = MeshTet(vertices.T, tets.T, sort_t=False)
    corners = vertices[tets]
    inverse = np.linalg.inv((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
    tree = cKDTree(corners.mean(axis=1))
    rows = []
    with np.load(args.sampling) as saved:
        for bank in ["peak_points_m", "column_points_m", "volume_points_m"]:
            points = saved[bank]
            failed = []
            for start in range(0, len(points), 64):
                current = points[start : start + 64]
                try:
                    mesh.element_finder()(*current.T)
                except ValueError:
                    for offset, point in enumerate(current):
                        try:
                            mesh.element_finder()(*point[:, None])
                        except ValueError:
                            candidates = tree.query(point, 128)[1]
                            local = np.einsum("eij,ej->ei", inverse[candidates], point - corners[candidates, 0])
                            bary = np.column_stack((1 - local.sum(axis=1), local))
                            best = int(bary.min(axis=1).argmax())
                            element, weights = int(candidates[best]), bary[best]
                            projected = np.maximum(weights, 0)
                            projected /= projected.sum()
                            distance = np.linalg.norm(projected @ corners[element] - point)
                            failed.append(
                                {
                                    "query": start + offset,
                                    "position_m": point.tolist(),
                                    "candidate_element": element,
                                    "barycentric": weights.tolist(),
                                    "projected_distance_m": float(distance),
                                    "inverse_norm_per_m": float(np.linalg.norm(inverse[element], ord=np.inf)),
                                }
                            )
                    if len(failed) >= 12:
                        break
            rows.append(
                {
                    "bank": bank,
                    "points": len(points),
                    "inspected_through": min(start + 64, len(points)),
                    "failures": failed,
                }
            )
            write_json(args.output / "progress.json", {"rows": rows})
    write_json(args.output / "result.json", {"execution_completed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
