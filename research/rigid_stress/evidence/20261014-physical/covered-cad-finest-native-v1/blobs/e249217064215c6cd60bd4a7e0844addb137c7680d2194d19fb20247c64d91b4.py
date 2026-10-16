"""Refine diagnosed bulk regions while preserving the seed's complete polyhedral domain."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skfem import MeshTet

from .refine_fixed_geometry import geometric_integrals
from .summarize_fine_references import closed_source


def verify_parent_partition(parent, child):
    vertices = parent.p.T
    tetrahedra = parent.t.T
    old = vertices[tetrahedra]
    inverse = np.linalg.inv((old[:, 1:] - old[:, :1]).transpose(0, 2, 1))
    tree = cKDTree(old.mean(axis=1))
    current = child.p.T[child.t.T]
    owners = np.full(len(current), -1, dtype=int)
    for first in range(0, len(current), 512):
        section = current[first : first + 512]
        unresolved = np.arange(len(section))
        for count in [16, 64, 256]:
            candidates = tree.query(section[unresolved].mean(axis=1), k=min(count, len(old)))[1]
            differences = section[unresolved, None] - old[candidates, :1]
            coordinates = np.einsum("nkij,nkvj->nkvi", inverse[candidates], differences)
            barycentric = np.concatenate((1 - coordinates.sum(axis=-1, keepdims=True), coordinates), axis=-1)
            contained = (barycentric >= -1e-10).all(axis=(-1, -2)) & (barycentric <= 1 + 1e-10).all(axis=(-1, -2))
            valid = contained.any(axis=1)
            indices = np.flatnonzero(valid)
            owners[first + unresolved[valid]] = candidates[indices, np.argmax(contained[valid], axis=1)]
            unresolved = unresolved[~valid]
            if not len(unresolved):
                break
        assert not len(unresolved), "A child tetrahedron is not contained in a parent tetrahedron"
    old_volume = abs(np.linalg.det(old[:, 1:] - old[:, :1])) / 6
    new_volume = abs(np.linalg.det(current[:, 1:] - current[:, :1])) / 6
    accumulated = np.bincount(owners, weights=new_volume, minlength=len(old_volume))
    np.testing.assert_allclose(accumulated, old_volume, rtol=1e-10, atol=1e-24)
    return {
        "all_child_tetrahedra_contained_in_declared_parent": True,
        "each_parent_volume_preserved": True,
        "maximum_parent_volume_relative_difference": float(np.max(abs(accumulated - old_volume) / old_volume)),
        "parent_ids": owners,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.radius > 0 and args.levels >= 3
    closed_source(args.diagnosis)
    assert int((args.diagnosis / "returncode.txt").read_text()) == 0
    report_path = args.diagnosis / "results/summary.json"
    report = json.loads(report_path.read_text())
    assert Path(report["current"]) / "mesh.npz" == args.seed
    anchors = np.unique(np.concatenate([row["top_five_positions_m"] for row in report["rows"]]), axis=0)
    assert anchors.shape == (5, 3)
    seed_metadata = json.loads(args.seed.with_suffix(".json").read_text())
    with np.load(args.seed) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
    args.output.mkdir(parents=True, exist_ok=False)
    seed_vertices = mesh.p.T.copy()
    seed_volume, seed_area, seed_com = geometric_integrals(mesh.p.T, mesh.t.T, mesh.facets[:, mesh.boundary_facets()].T)
    rows, partition = [], None
    for level in range(args.levels):
        np.testing.assert_array_equal(mesh.p.T[: len(seed_vertices)], seed_vertices)
        vertices, tetrahedra = mesh.p.T.copy(), mesh.t.T.copy()
        determinants = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
        assert np.all(determinants != 0)
        flipped = determinants < 0
        tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
        boundary = mesh.facets[:, mesh.boundary_facets()].T
        volume, area, com = geometric_integrals(vertices, tetrahedra, boundary)
        np.testing.assert_allclose(volume.sum(), seed_volume.sum(), rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(area.sum(), seed_area.sum(), rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(com, seed_com, rtol=0, atol=1e-14)
        path = args.output / f"level{level}.npz"
        np.savez_compressed(path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary)
        row = {
            "geometry": seed_metadata["geometry"],
            "geometry_mode": "Fixed seed polyhedral domain; longest-edge bisection without CAD projection",
            "seed_sha256": hashlib.sha256(args.seed.read_bytes()).hexdigest(),
            "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "diagnosis_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "local_refinement_level": level,
            "marking_radius_m": args.radius,
            "marking_anchors_m": anchors.tolist(),
            "tetrahedra": len(tetrahedra),
            "nodes": len(vertices),
            "surface_triangles": len(boundary),
            "volume_m3": float(volume.sum()),
            "surface_area_m2": float(area.sum()),
            "com_m": com.tolist(),
            "minimum_tetrahedron_volume_m3": float(volume.min()),
            "parent_partition": None,
            "goal_completion": False,
            "limitations": "Adaptive marking changes numerical bulk/trace resolution while retaining the seed's polyhedral physical domain, material and mass. Boundary triangle subdivisions change; no boundary quadrature identity or solved FEM/stress convergence is inferred. All original36 convergence probes remain required. The marking radius is numerical and does not change the physical load footprint.",
        }
        if partition is not None:
            np.savez_compressed(args.output / f"level{level}.parents.npz", parent_ids=partition["parent_ids"])
            row["parent_partition"] = {k: v for k, v in partition.items() if k != "parent_ids"}
        path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(json.dumps({"level": level, "tetrahedra": len(tetrahedra)}), flush=True)
        if level + 1 < args.levels:
            distances = cKDTree(anchors).query(vertices[tetrahedra].mean(axis=1))[0]
            marked = np.flatnonzero(distances <= args.radius)
            assert len(marked) > 0
            child = mesh.refined(marked)
            assert child.nelements > mesh.nelements
            partition = verify_parent_partition(mesh, child)
            mesh = child
    result = {"scope": __doc__, "rows": rows, "goal_completion": False}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
