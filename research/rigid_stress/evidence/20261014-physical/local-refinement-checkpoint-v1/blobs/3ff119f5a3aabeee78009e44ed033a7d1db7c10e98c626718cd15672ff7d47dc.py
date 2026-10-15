"""Audit contact-centered local refinement geometry before assembling larger FEM solves."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .franka_contact_reference import build_from_capture
from .localized_franka_mesh import contact_centres, localized_mesh


def geometry(mesh):
    vertices = mesh.p[:, mesh.t].transpose(2, 1, 0)
    edges = vertices[:, 1:] - vertices[:, :1]
    determinant = np.linalg.det(edges)
    volume = abs(determinant) / 6
    assert np.all(volume > 0)
    return float(volume.sum()), np.einsum("e,ej->j", volume, vertices.mean(axis=1)) / volume.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-run", type=Path, required=True)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.001)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15, 18])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.closure_run / "results/configuration.json").read_text())
    system, _ = build_from_capture(Path(config["capture"]), config["dhat"])
    centres = contact_centres(system, args.contact_run)
    seed_volume, seed_com = geometry(system.body.mesh)
    rows = []
    for level in args.levels:
        mesh, history = localized_mesh(system.body.mesh, centres, args.radius, level)
        volume, com = geometry(mesh)
        np.testing.assert_allclose(volume, seed_volume, rtol=1e-13, atol=0)
        np.testing.assert_allclose(com, seed_com, rtol=0, atol=1e-14)
        distance, _ = cKDTree(mesh.p.T).query(system.body.mesh.p.T)
        assert distance.max() == 0
        mesh_edges = mesh.p[:, mesh.edges].transpose(2, 1, 0)
        lengths = np.linalg.norm(mesh_edges[:, 1] - mesh_edges[:, 0], axis=1)
        # Retain all original facets as coplanar subdivisions; compare boundary area too.
        facets = mesh.p[:, mesh.facets[:, mesh.boundary_facets()]].transpose(2, 1, 0)
        area = np.linalg.norm(np.cross(facets[:, 1] - facets[:, 0], facets[:, 2] - facets[:, 0]), axis=1).sum() / 2
        row = {
            "level": level,
            "tetrahedra": int(mesh.nelements),
            "vertices": int(mesh.nvertices),
            "p2_body_dofs": int(3 * (mesh.nvertices + mesh.nedges)),
            "min_edge_m": float(lengths.min()),
            "max_edge_m": float(lengths.max()),
            "volume_m3": volume,
            "com_m": com.tolist(),
            "boundary_area_m2": float(area),
            "history": history,
        }
        if rows:
            np.testing.assert_allclose(area, rows[0]["boundary_area_m2"], rtol=1e-12, atol=0)
        np.savez_compressed(args.output / f"mesh{level:02d}.npz", vertices=mesh.p.T, tetrahedra=mesh.t.T)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)
    result = {
        "scope": __doc__,
        "radius_m": args.radius,
        "centres_m": centres.tolist(),
        "rows": rows,
        "goal_completion": False,
        "limitations": "Geometry and mesh-size qualification only. Fixed polyhedral geometry, material and physical contact protocol are unchanged. Stress/contact convergence and a resolved physical patch require solved fields and independent tails.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
