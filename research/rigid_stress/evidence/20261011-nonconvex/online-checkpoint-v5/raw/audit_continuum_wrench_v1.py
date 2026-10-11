"""Offline FP64 outer bound for all six wrench components on a finite material footprint.

Each true planar face is intersected with a circumscribed polygon around its
Euclidean R-ball section. This is a superset of the intrinsic footprint. Coulomb
cones are also circumscribed for the LP. A verified separating dual therefore
proves physical incompatibility; outer feasibility alone makes no such claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def closest_points(anchor, triangles):
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    local = np.einsum("fij,fj->fi", np.linalg.pinv(edges), anchor - triangles[:, 0])
    inside = (local >= 0).all(axis=1) & (local.sum(axis=1) <= 1)
    points = triangles[:, 0] + np.einsum("fij,fj->fi", edges, local)
    first, second = triangles, triangles[:, [1, 2, 0]]
    sides = second - first
    fraction = np.einsum("fai,fai->fa", anchor - first, sides) / np.einsum("fai,fai->fa", sides, sides)
    candidates = first + np.clip(fraction, 0, 1)[..., None] * sides
    nearest = np.argmin(np.linalg.norm(candidates - anchor, axis=2), axis=1)
    points[~inside] = candidates[np.arange(len(triangles)), nearest][~inside]
    return points


def clip_polygon(polygon, direction, limit):
    values = polygon @ direction - limit
    end = np.roll(polygon, -1, axis=0)
    end_values = np.roll(values, -1)
    retained = []
    for first, second, a, b in zip(polygon, end, values, end_values):
        if a <= 0:
            retained.append(first)
        if (a < 0 < b) or (b < 0 < a):
            retained.append(first + a / (a - b) * (second - first))
    return np.array(retained).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=Path("examples/rigid/assets/hollow_egg/level1/elastic.npz"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.mesh) as saved:
        vertices, faces, tets = (saved[key] for key in ("vertices", "surface_triangles", "tetrahedra"))
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    contains = (faces[:, None, :, None] == tets[None, :, None, :]).any(axis=3).all(axis=2)
    assert (contains.sum(axis=1) == 1).all()
    owners = tets[contains.argmax(axis=1)]
    opposite = owners[(owners[:, :, None] != faces[:, None]).all(axis=2)]
    toward_owner = np.einsum("fi,fi->f", normals, vertices[opposite] - triangles[:, 0]) > 0
    normals[~toward_owner] *= -1
    records = []
    for snapshot in args.snapshots:
        with np.load(snapshot) as saved:
            failed = saved["valid"] & (saved["traction_status"] == 3) & (saved["source_status"] == 0)
            for env, contact in zip(*np.nonzero(failed)):
                point, anchor, force = (saved[key][env, contact] for key in ("position", "anchor", "force"))
                radius, friction = (float(saved[key][env, contact]) for key in ("radius", "friction"))
                moment = np.cross(point - anchor, force)
                scale = max(np.linalg.norm(force), np.linalg.norm(moment) / radius)
                target = np.concatenate((force, moment / radius)) / scale
                candidates = np.nonzero(np.linalg.norm(closest_points(anchor, triangles) - anchor, axis=1) <= radius)[0]
                record = {"snapshot": str(snapshot), "env": int(env), "contact": int(contact),
                          "radius_m": radius, "friction": friction, "candidate_faces": candidates.tolist(),
                          "bounds": []}
                for sides in (32, 128):
                    positions, sample_normals, sample_faces = [], [], []
                    for face in candidates:
                        triangle, normal = triangles[face], normals[face]
                        first = triangle[1] - triangle[0]
                        first /= np.linalg.norm(first)
                        second = np.cross(normal, first)
                        basis = np.column_stack((first, second))
                        plane_offset = normal @ (anchor - triangle[0])
                        center = anchor - plane_offset * normal
                        local = (triangle - center) @ basis
                        section_radius = np.sqrt(max(0, radius**2 - plane_offset**2))
                        angles = np.arange(sides) * (2 * np.pi / sides)
                        for angle in angles:
                            local = clip_polygon(local, np.array([np.cos(angle), np.sin(angle)]), section_radius)
                            if len(local) == 0:
                                break
                        if len(local):
                            positions.extend(center + local @ basis.T)
                            sample_normals.extend(np.repeat(normal[None], len(local), axis=0))
                            sample_faces.extend([face] * len(local))
                    positions, sample_normals = np.array(positions), np.array(sample_normals)
                    relative = (positions - anchor) / radius
                    axis = np.eye(3)[np.argmin(np.abs(sample_normals), axis=1)]
                    first = np.cross(sample_normals, axis)
                    first /= np.linalg.norm(first, axis=1)[:, None]
                    second = np.cross(sample_normals, first)
                    rays = sample_normals[:, None] + friction / np.cos(np.pi / sides) * (
                        first[:, None] * np.cos(angles)[None, :, None]
                        + second[:, None] * np.sin(angles)[None, :, None])
                    columns = np.concatenate((rays, np.cross(relative[:, None], rays)), axis=2).reshape(-1, 6).T
                    feasibility = linprog(np.zeros(columns.shape[1]), A_eq=columns, b_eq=target,
                                         bounds=(0, None), method="highs",
                                         options={"primal_feasibility_tolerance": 1e-9})
                    dual = linprog(target, A_ub=-columns.T, b_ub=np.zeros(columns.shape[1]),
                                   bounds=[(-1, 1)] * 6, method="highs")
                    assert dual.success
                    vector = dual.x[:3] + np.cross(dual.x[3:], relative)
                    compression = np.einsum("qi,qi->q", vector, sample_normals)
                    tangent = vector - compression[:, None] * sample_normals
                    margin = compression - friction * np.linalg.norm(tangent, axis=1)
                    result = {"sides": sides, "outer_vertices": len(positions),
                              "feasibility_status": int(feasibility.status), "separator": dual.x.tolist(),
                              "target_separator_dot": float(target @ dual.x),
                              "exact_coulomb_dual_margin_min": float(margin.min()),
                              "certified_physical_incompatibility": bool(target @ dual.x < -1e-8 and margin.min() >= -1e-12)}
                    if feasibility.success:
                        force_witness = (feasibility.x.reshape(-1, sides, 1) * rays).sum(axis=1)
                        result["outer_wrench_error"] = float(np.linalg.norm(columns @ feasibility.x - target))
                        result["used_faces"] = np.unique(np.array(sample_faces)[np.linalg.norm(force_witness, axis=1) > 1e-12]).tolist()
                    record["bounds"].append(result)
                records.append(record)
                print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": __doc__, "contacts": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
