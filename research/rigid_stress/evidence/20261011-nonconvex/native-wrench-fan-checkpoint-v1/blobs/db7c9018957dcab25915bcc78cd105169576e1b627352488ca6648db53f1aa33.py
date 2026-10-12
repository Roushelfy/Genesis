"""Independent CPU FP64 aggregate-wrench outer proof on all original finite disks and true material cones."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from audit_continuum_wrench_v2 import clip_polygon, closest_points

root = Path(__file__).parent
capture = root / "native-wrench-fan-failure-capture-v1/b4096-623103"
output = root / "wrench-fan-continuum-audit-v1"
assert output.is_dir()
with np.load(capture / "failure/before-rigid-integration.npz") as saved:
    data = {key: saved[key].copy() for key in saved.files}
with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as saved:
    vertices, faces, tets = (saved[key].copy() for key in ("vertices", "surface_triangles", "tetrahedra"))
triangles = vertices[faces]
normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
normals /= np.linalg.norm(normals, axis=1)[:, None]
contains = (faces[:, None, :, None] == tets[None, :, None, :]).any(axis=3).all(axis=2)
assert np.all(contains.sum(axis=1) == 1)
owners = tets[contains.argmax(axis=1)]
opposite = owners[(owners[:, :, None] != faces[:, None]).all(axis=2)]
toward_owner = np.einsum("fi,fi->f", normals, vertices[opposite] - triangles[:, 0]) > 0
normals[~toward_owner] *= -1
valid = data["fit_contacts.valid"][0]
indices = np.flatnonzero(valid)
centers, radii, mus = (data[f"fit_contacts.{key}"][0, indices] for key in ("center", "radius", "friction"))
original_force = data["tractions.priors.force"][0, indices]
original_moment = data["tractions.priors.anchor_moment"][0, indices]
anchor = centers[0]
force = original_force.sum(axis=0)
moment = (original_moment + np.cross(centers - anchor, original_force)).sum(axis=0)
np.testing.assert_allclose(force, data["fit_contacts.force"][0, 0], rtol=0, atol=1e-14)
np.testing.assert_allclose(moment, data["fit_sources.anchor_moment"][0, 0], rtol=0, atol=1e-14)
radius = float(radii[0])
scale = max(np.linalg.norm(force), np.linalg.norm(moment) / radius)
target = np.concatenate((force, moment / radius)) / scale
records = []
for sides in (32, 128):
    positions, point_normals, friction, source, boundary_faces = [], [], [], [], []
    angles = np.arange(sides) * (2 * np.pi / sides)
    for i_s, (center, footprint_radius, mu) in enumerate(zip(centers, radii, mus)):
        geometric_roundoff = 256 * np.finfo(float).eps * (np.linalg.norm(center) + np.linalg.norm(vertices, axis=1).max() + footprint_radius)
        outer_radius = footprint_radius + geometric_roundoff
        candidates = np.flatnonzero(np.linalg.norm(closest_points(center, triangles) - center, axis=1) <= outer_radius)
        for face in candidates:
            triangle, normal = triangles[face], normals[face]
            first = triangle[1] - triangle[0]
            first /= np.linalg.norm(first)
            second = np.cross(normal, first)
            basis = np.column_stack((first, second))
            plane_offset = normal @ (center - triangle[0])
            section_center = center - plane_offset * normal
            local = (triangle - section_center) @ basis
            section_radius = np.sqrt(max(0, outer_radius**2 - plane_offset**2))
            for angle in angles:
                local = clip_polygon(local, np.array([np.cos(angle), np.sin(angle)]), section_radius)
                if len(local) == 0:
                    break
            if len(local):
                positions.extend(section_center + local @ basis.T)
                point_normals.extend(np.repeat(normal[None], len(local), axis=0))
                friction.extend([mu] * len(local))
                source.extend([i_s] * len(local))
                boundary_faces.extend([int(face)] * len(local))
    positions, point_normals, friction = np.array(positions), np.array(point_normals), np.array(friction)
    relative = (positions - anchor) / radius
    axes = np.eye(3)[np.argmin(abs(point_normals), axis=1)]
    first = np.cross(point_normals, axes)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(point_normals, first)
    rays = point_normals[:, None] + (friction / np.cos(np.pi / sides))[:, None, None] * (
        np.cos(angles)[None, :, None] * first[:, None] + np.sin(angles)[None, :, None] * second[:, None])
    columns = np.concatenate((rays, np.cross(relative[:, None], rays)), axis=-1).reshape(-1, 6).T
    primal = linprog(np.zeros(columns.shape[1]), A_eq=columns, b_eq=target, bounds=(0, None),
                     options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    dual = linprog(target, A_ub=-columns.T, b_ub=np.zeros(columns.shape[1]), bounds=[(-1, 1)] * 6,
                   options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    assert dual.success
    ray = -dual.x / np.linalg.norm(dual.x) if np.linalg.norm(dual.x) else dual.x
    values = ray[:3] + np.cross(ray[3:], relative)
    compression = np.einsum("qi,qi->q", values, point_normals)
    polar = compression + friction * np.linalg.norm(values - compression[:, None] * point_normals, axis=1)
    margin = float(target @ ray)
    maximum = float(polar.max())
    native_ray = data["tractions.certificate"][0, 0]
    native_ray /= np.linalg.norm(native_ray)
    values = native_ray[:3] + np.cross(native_ray[3:], relative)
    compression = np.einsum("qi,qi->q", values, point_normals)
    native_polar = compression + friction * np.linalg.norm(values - compression[:, None] * point_normals, axis=1)
    row = {"sides": sides, "vertices": len(positions), "primal_status": int(primal.status),
           "ray": ray.tolist(), "target_margin": margin, "continuous_polar_upper_bound": maximum,
           "certified_physical_incompatibility": margin > 1e-8 and maximum < -1e-12,
           "native_ray_target_margin": float(target @ native_ray),
           "native_ray_continuous_polar_upper_bound": float(native_polar.max()),
           "native_reported_bound": float(data["wrench_search.bound"][0, 0])}
    records.append(row)
    np.savez_compressed(output / f"outer-{sides}.npz", positions=positions, normals=point_normals, friction=friction,
                        source=source, face=boundary_faces, ray=ray, target=target)
    print(json.dumps(row), flush=True)
summary = {"scope": __doc__, "selected_environment": int(data["environment"]), "source_count": len(indices),
           "original_force": original_force.tolist(), "original_moment": original_moment.tolist(),
           "centers": centers.tolist(), "radii": radii.tolist(), "friction": mus.tolist(),
           "aggregate_force": force.tolist(), "aggregate_moment_about_anchor": moment.tolist(),
           "bounds": records,
           "interpretation": "Each outer polygon contains the Euclidean disk/triangle intersection,which contains the intrinsic support. A256epsilon coordinate envelope makes only the offline outer proof conservative;original physical radii are unchanged. The true cone polar is convex in position. Negative vertex maxima plus a positive target margin therefore certify the complete unchanged continuous load model. Outer feasibility alone provides no physical-feasibility claim."}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
