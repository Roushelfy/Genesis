"""Check whether fixed-polyhedron refinement resolves the original contact geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from skfem import MeshTet

from .audit_objective_frame import finite_frame
from .franka_contact_reference import build_from_capture
from .summarize_fine_references import closed_source


def boundary_normals(mesh):
    indices = mesh.boundary_facets()
    faces = mesh.facets[:, indices].T
    points = mesh.p.T[faces]
    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    area = np.linalg.norm(normal, axis=1) / 2
    assert np.all(area > 0)
    normal /= 2 * area[:, None]
    cells = mesh.f2t[0, indices]
    assert np.all(cells >= 0)
    outward = points.mean(axis=1) - mesh.p.T[mesh.t.T[cells]].mean(axis=1)
    normal[np.einsum("ij,ij->i", normal, outward) < 0] *= -1
    return faces, normal, area


def unique_normals(normals):
    result = []
    for normal in normals:
        if not result or min(np.linalg.norm(normal - old) for old in result) > 1e-8:
            result.append(normal)
    return np.array(result)


def describe(mesh, anchor, direction):
    distance, vertex = cKDTree(mesh.p.T).query(anchor)
    assert distance < 1e-12
    faces, normals, area = boundary_normals(mesh)
    incident = np.any(faces == vertex, axis=1)
    selected = unique_normals(normals[incident])
    assert len(selected) >= 3
    neighbours = np.setdiff1d(np.unique(faces[incident]), [vertex])
    edge = mesh.p.T[neighbours] - anchor
    length = np.linalg.norm(edge, axis=1)
    gap = -edge @ direction
    tangent = edge + gap[:, None] * direction
    tangent_length = np.linalg.norm(tangent, axis=1)
    assert np.all(tangent_length > 0)
    matrix = np.column_stack((-np.eye(len(selected)), np.ones(len(selected))))
    result = linprog(
        np.r_[np.zeros(len(selected)), -1],
        A_ub=matrix,
        b_ub=np.zeros(len(selected)),
        A_eq=np.column_stack((selected.T, np.zeros(3))),
        b_eq=direction,
        bounds=(0, None),
        options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    )
    assert result.success
    assert np.linalg.norm(selected.T @ result.x[:-1] - direction) < 1e-12
    full_gap = -(mesh.p.T - anchor) @ direction
    row = {
        "anchor_m": anchor.tolist(),
        "outward_contact_direction": direction.tolist(),
        "body_vertex": int(vertex),
        "incident_boundary_triangles": int(incident.sum()),
        "distinct_incident_plane_normals": selected.tolist(),
        "maximum_incident_normal_angle_deg": float(
            np.rad2deg(np.arccos(np.clip((selected @ selected.T).min(), -1, 1)))
        ),
        "positive_normal_cone_coefficient_margin": float(result.x[-1]),
        "incident_edge_min_m": float(length.min()),
        "incident_edge_max_m": float(length.max()),
        "neighbour_support_gap_min_m": float(gap.min()),
        "neighbour_gap_over_tangent_distance_min": float((gap / tangent_length).min()),
        "neighbour_gap_over_tangent_distance_max": float((gap / tangent_length).max()),
        "whole_body_support_plane_min_gap_m": float(full_gap.min()),
        "incident_lumped_area_m2": float(area[incident].sum() / 3),
    }
    return row, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-run", type=Path, required=True)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for run in [args.mesh_run, args.contact_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    config = json.loads((args.contact_run / "results/configuration.json").read_text())
    source_config = config
    seen = set()
    while "capture" not in source_config:
        parent = Path(source_config["closure_run"])
        assert parent not in seen
        seen.add(parent)
        closed_source(parent)
        source_config = json.loads((parent / "results/configuration.json").read_text())
    system, _ = build_from_capture(Path(source_config["capture"]), source_config["dhat"])
    n = system.body.basis.N
    with np.load(args.contact_run / "results/reference_geometry.npz") as saved:
        np.testing.assert_array_equal(saved["vertices"], system.body.mesh.p.T)
        np.testing.assert_array_equal(saved["tetrahedra"], system.body.mesh.t.T)
    field_path = args.contact_run / f"results/step{config['steps']:04d}.npz"
    with np.load(field_path) as saved:
        u, saved_force = saved["displacement"], saved["contact_force_N"]
    normal, current = system.normal_collisions(u)
    normal_force = -system.mesh.to_full_dof(system.barrier.gradient(normal, system.mesh, current))[:n].reshape(-1, 3)
    active = np.flatnonzero(np.linalg.norm(normal_force, axis=1) > 1e-8)
    assert len(active) == 2
    assert set(active) == set(np.flatnonzero(np.linalg.norm(saved_force[:n].reshape(-1, 3), axis=1) > 1e-8))
    vector = np.empty(n)
    vector[system.parts[0].permutation] = u[:n]
    rotation = finite_frame(system.body, vector)[0]
    anchors = system.rest[active]
    directions = -normal_force[active] @ rotation
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    rows, original_normals, inputs = [], None, {}
    for level in args.levels:
        path = args.mesh_run / f"results/mesh{level:02d}.npz"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
        contacts, normals = [], []
        for anchor, direction in zip(anchors, directions):
            row, selected = describe(mesh, anchor, direction)
            contacts.append(row)
            normals.append(selected)
        if original_normals is None:
            original_normals = normals
        for contact, selected, seed in zip(contacts, normals, original_normals):
            distance = np.linalg.norm(selected[:, None] - seed[None], axis=2)
            error = max(distance.min(axis=1).max(), distance.min(axis=0).max())
            assert error < 1e-9
            contact["normal_set_distance_from_seed"] = float(error)
        rows.append({"level": level, "tetrahedra": int(mesh.nelements), "contacts": contacts})
    inputs[str(field_path)] = hashlib.sha256(field_path.read_bytes()).hexdigest()
    report = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Incident plane normals and the supporting vertex describe the fixed seed polyhedron, not a refined smooth egg. Normal-cone/support margins diagnose the geometric contact type; they do not prove a stress singularity or establish contact/pressure/continuum convergence. Unmodified fixed-geometry refinement remains a valid volume axis. A smooth-surface geometry axis is a separately declared approximation to the original physical geometry, not permission to tune rounding or replace raw maxima.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"levels": len(rows), "contacts_per_level": len(anchors), "seed_contacts": rows[0]["contacts"]}))


if __name__ == "__main__":
    main()
