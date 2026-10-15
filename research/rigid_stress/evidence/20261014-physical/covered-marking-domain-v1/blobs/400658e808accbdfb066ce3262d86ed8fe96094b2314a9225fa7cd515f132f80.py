"""Refine tetrahedra with scikit-fem edge midpoints while retaining the seed polyhedral geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from skfem import MeshTet


def geometric_integrals(vertices, tetrahedra, boundary):
    edges = vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]]
    volumes = abs(np.linalg.det(edges)) / 6
    triangles = vertices[boundary]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    centroids = vertices[tetrahedra].mean(axis=1)
    return volumes, areas, (centroids * volumes[:, None]).sum(axis=0) / volumes.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.seed) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
    seed_metadata = json.loads(args.seed.with_suffix(".json").read_text())
    seed_volume, seed_area, seed_com = geometric_integrals(mesh.p.T, mesh.t.T, mesh.facets[:, mesh.boundary_facets()].T)
    rows = []
    for level in range(args.levels):
        vertices, tetrahedra = mesh.p.T.copy(), mesh.t.T.copy()
        determinant = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
        flipped = determinant < 0
        tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
        boundary = mesh.facets[:, mesh.boundary_facets()].T
        volumes, areas, com = geometric_integrals(vertices, tetrahedra, boundary)
        np.testing.assert_allclose(volumes.sum(), seed_volume.sum(), rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(areas.sum(), seed_area.sum(), rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(com, seed_com, rtol=1e-12, atol=1e-15)
        output = args.output / f"level{level}.npz"
        np.savez_compressed(output, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary)
        row = {
            "geometry": seed_metadata["geometry"],
            "geometry_mode": "Frozen seed polyhedron; scikit-fem uniform edge-midpoint refinement, no CAD projection",
            "seed_sha256": hashlib.sha256(args.seed.read_bytes()).hexdigest(),
            "mesh_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "uniform_refinement_level": level,
            "tetrahedra": len(tetrahedra),
            "nodes": len(vertices),
            "surface_triangles": len(boundary),
            "volume_m3": float(volumes.sum()),
            "surface_area_m2": float(areas.sum()),
            "com_m": com.tolist(),
            "minimum_tetrahedron_volume_m3": float(volumes.min()),
        }
        output.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        if level + 1 < args.levels:
            mesh = mesh.refined()
    (args.output / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    print(json.dumps({"levels": args.levels, "tetrahedra": [row["tetrahedra"] for row in rows]}))


if __name__ == "__main__":
    main()
