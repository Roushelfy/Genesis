"""Refine audited finite-load hotspots without changing their seed polyhedral domain or physical loads."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skfem import MeshTet

from .refine_fixed_geometry import geometric_integrals
from .refine_thickness_tail_regions import verify_parent_partition
from .summarize_fine_references import CASES, closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=["ellipsoid", "capsule"], required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--passes-per-level", type=int, default=2)
    parser.add_argument("--marking-radius", type=float, default=0.003)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.levels >= 3 and 1 <= args.passes_per_level <= 3 and args.marking_radius > 0
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return digest

    closed_source(args.diagnosis)
    assert int((args.diagnosis / "returncode.txt").read_text()) == 0
    report_path = args.diagnosis / "results/summary.json"
    retain(report_path)
    report = json.loads(report_path.read_text())
    assert report["family"] == "fixed_polyhedron" and len(report["rows"]) == 30
    selected = [row for row in report["rows"] if row["geometry"] == args.geometry and row["level"] == 2]
    assert [row["case"] for row in selected] == list(CASES)
    seed_hash = retain(args.seed)
    assert all(row["mesh_sha256"] == seed_hash for row in selected)
    stress_anchors = np.unique([row["raw_global_peak_position_m"] for row in selected], axis=0)
    load_anchors = []
    for row in selected:
        run = Path(row["run"])
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        path = run / f"results/{row['case']}/snapshot.json"
        retain(path)
        snapshot = json.loads(path.read_text())
        assert snapshot["mesh_sha256"] == seed_hash and snapshot["patch_radius_m"] == 0.003
        load_anchors.extend(patch["physical_anchor_m"] for patch in snapshot["patches"])
    load_anchors = np.unique(load_anchors, axis=0)
    anchors = np.unique(np.concatenate((stress_anchors, load_anchors)), axis=0)
    tree = cKDTree(anchors)
    metadata_path = args.seed.with_suffix(".json")
    retain(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["geometry"]["name"] == args.geometry
    with np.load(args.seed) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
    args.output.mkdir(parents=True, exist_ok=False)
    seed_vertices = mesh.p.T.copy()
    original_volume, original_area, original_com = geometric_integrals(
        mesh.p.T, mesh.t.T, mesh.facets[:, mesh.boundary_facets()].T
    )
    rows = []
    partition_records = []
    for level in range(args.levels):
        np.testing.assert_array_equal(mesh.p.T[: len(seed_vertices)], seed_vertices)
        vertices, tetrahedra = mesh.p.T.copy(), mesh.t.T.copy()
        determinant = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
        assert np.all(determinant != 0)
        flipped = determinant < 0
        tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
        boundary = mesh.facets[:, mesh.boundary_facets()].T
        volume, area, com = geometric_integrals(vertices, tetrahedra, boundary)
        np.testing.assert_allclose(volume.sum(), original_volume.sum(), rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(area.sum(), original_area.sum(), rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(com, original_com, rtol=0, atol=1e-14)
        path = args.output / f"level{level}.npz"
        np.savez_compressed(path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary)
        row = {
            "geometry": metadata["geometry"],
            "geometry_mode": "Frozen seed polyhedron; local longest-edge bisection, no CAD projection",
            "seed_sha256": seed_hash,
            "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "local_refinement_level": level,
            "passes_per_level": args.passes_per_level,
            "numerical_marking_radius_m": args.marking_radius,
            "unchanged_physical_load_radius_m": 0.003,
            "audited_raw_peak_marking_anchors_m": stress_anchors.tolist(),
            "all_original_load_marking_anchors_m": load_anchors.tolist(),
            "tetrahedra": len(tetrahedra),
            "nodes": len(vertices),
            "surface_triangles": len(boundary),
            "volume_m3": float(volume.sum()),
            "surface_area_m2": float(area.sum()),
            "com_m": com.tolist(),
            "minimum_tetrahedron_volume_m3": float(volume.min()),
            "parent_partition_records": partition_records,
            "new_equilibrium_states_solved": 0,
            "goal_completion": False,
        }
        path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(json.dumps({"level": level, "tetrahedra": len(tetrahedra)}), flush=True)
        partition_records = []
        if level + 1 < args.levels:
            for pass_index in range(args.passes_per_level):
                distance = tree.query(mesh.p.T[mesh.t.T].mean(axis=1))[0]
                marked = np.flatnonzero(distance <= args.marking_radius)
                assert len(marked) > 0
                child = mesh.refined(marked)
                assert child.nelements > mesh.nelements
                partition = verify_parent_partition(mesh, child)
                partition_file = f"level{level + 1}-pass{pass_index}.parents.npz"
                np.savez_compressed(args.output / partition_file, parent_ids=partition["parent_ids"])
                partition_records.append(
                    {
                        **{key: value for key, value in partition.items() if key != "parent_ids"},
                        "parent_tetrahedra": mesh.nelements,
                        "child_tetrahedra": child.nelements,
                        "marked_parent_tetrahedra": len(marked),
                        "parent_ids_file": partition_file,
                    }
                )
                mesh = child
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "geometry": args.geometry,
        "rows": rows,
        "input_sha256": inputs,
        "new_equilibrium_states_solved": 0,
        "goal_completion": False,
        "limitations": "Numerical marking uses all five audited seed raw maxima and every original physical load anchor, with a declared fixed physical-space marking radius. No original load/material/footprint, seed boundary geometry or original convergence probe is changed. Every child tetrahedron is contained in a parent and every parent volume is checked, alongside total volume/area/COM. Volume and surface-basis refinement and boundary quadrature still couple. Marking uses measured seed hotspots and cannot establish a universal error estimator or exclude another global peak. All original field budgets, probes and unaveraged global maxima must be retained in subsequent solves and independent audits. Asset generation solves zero new equilibria.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
