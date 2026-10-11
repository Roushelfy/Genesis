"""Offline geometry/cone certificate; never used by production recovery."""

import json
from pathlib import Path

import numpy as np

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
output = root / "continuum-cone-v5"
output.mkdir(exist_ok=True)
with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as mesh:
    vertices = mesh["vertices"]
    faces = mesh["surface_triangles"]
    tetrahedra = mesh["tetrahedra"]
triangles = vertices[faces]
normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
normals /= np.linalg.norm(normals, axis=1)[:, None]
for i, face in enumerate(faces):
    owner = np.nonzero(np.sum(np.isin(tetrahedra, face), axis=1) == 3)[0]
    assert len(owner) == 1
    opposite = np.setdiff1d(tetrahedra[owner[0]], face)
    if normals[i] @ (vertices[opposite[0]] - triangles[i, 0]) < 0:
        normals[i] *= -1

records = []
for tick, folder in [(147, "online-mesh64-v4/results/failure")]:
    with np.load(root / folder / f"snapshot-{tick}.npz") as saved:
        keys = list(saved.files)
        env, contact = 1, 2
        anchor, position, force = (saved[key][env, contact] for key in ("anchor", "position", "force"))
        radius, friction = (float(saved[key][env, contact]) for key in ("radius", "friction"))
        unit_force = force / np.linalg.norm(force)
        midpoint_direction = position - saved["witness"][env, contact]
        midpoint_direction /= np.linalg.norm(midpoint_direction)
        midpoint_compression = unit_force @ midpoint_direction
        midpoint_cone_margin = friction * midpoint_compression - np.linalg.norm(unit_force - midpoint_compression * midpoint_direction)
        # AABB intersection is a superset of the Euclidean radius ball and thus
        # a superset of every intrinsic footprint with distance < radius.
        low, high = triangles.min(axis=1), triangles.max(axis=1)
        aabb_distance = np.linalg.norm(np.maximum(np.maximum(low - anchor, anchor - high), 0), axis=1)
        closest_distances = np.full(len(faces), np.inf)
        for face, triangle in enumerate(triangles):
            origin = triangle[0]
            edges = (triangle[1:] - origin).T
            local = np.linalg.solve(edges.T @ edges, edges.T @ (anchor - origin))
            if min(local[0], local[1], 1 - local.sum()) >= 0:
                closest_distances[face] = np.linalg.norm(origin + edges @ local - anchor)
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge = triangle[b] - triangle[a]
                t = np.clip(edge @ (anchor - triangle[a]) / (edge @ edge), 0, 1)
                closest_distances[face] = min(closest_distances[face], np.linalg.norm(triangle[a] + t * edge - anchor))
        eligible = np.nonzero(closest_distances < radius)[0]
        projections = []
        for face in eligible:
            n = normals[face]
            compression = n @ unit_force
            tangent = unit_force - compression * n
            r = np.linalg.norm(tangent)
            projected_compression = max(0, (compression + friction * r) / (1 + friction**2))
            projected = projected_compression * n + friction * projected_compression * tangent / r
            residual = unit_force - projected
            distance = np.linalg.norm(residual)
            if distance > 1e-15:
                ray = (tangent / r - friction * n) / np.sqrt(1 + friction**2)
                rn = normals[eligible] @ ray
                polar_values = rn + friction * np.linalg.norm(ray[None] - rn[:, None] * normals[eligible], axis=1)
                projections.append({
                    "face": int(face), "distance": float(distance), "polar_ray": ray.tolist(),
                    "polar_max": float(polar_values.max()), "polar_values": polar_values.tolist(),
                    "force_dot_ray": float(unit_force @ ray),
                    "cone_margin": float(friction * compression - r),
                })
        best = min(projections, key=lambda x: x["distance"])
        # Bound the floating-point residual in a nominally tangent polar ray.
        # Perturb it inward toward -normal: strict support <= 0 then provides
        # an explicit admissible separating ray (cone polar is scale-free).
        ray = np.asarray(best["polar_ray"]) - 2e-12 * normals[best["face"]]
        ray /= np.linalg.norm(ray)
        rn = normals[eligible] @ ray
        support = rn + friction * np.linalg.norm(ray[None] - rn[:, None] * normals[eligible], axis=1)
        records.append({
            "tick": tick, "snapshot_keys": keys, "force": force.tolist(), "anchor": anchor.tolist(),
            "position": position.tolist(), "radius": radius, "friction": friction,
            "midpoint_witness_direction": midpoint_direction.tolist(), "midpoint_direction_cone_margin": float(midpoint_cone_margin),
            "midpoint_direction_reference_normal_difference": float(np.linalg.norm(midpoint_direction - normals[int(saved["anchor_face"][env, contact])])),
            "eligible_superset_faces": eligible.tolist(), "projection": best,
            "face_closest_distances": closest_distances[eligible].tolist(),
            "strict_polar_ray": ray.tolist(), "strict_polar_support": support.tolist(),
            "strict_polar_force_lower_bound": float(unit_force @ ray),
            "anchor_face": int(saved["anchor_face"][env, contact]),
            "witness": saved["witness"][env, contact].tolist(),
        })

with np.load(root / "online-mesh64-v4/results/source-vertices.npz") as saved:
    collision = saved["positions"]
    offsets = saved["offsets"]
    distances = np.linalg.norm(collision[:, None] - vertices[None], axis=2)
    nearest = np.argmin(distances, axis=1)
    geometry = {
        "keys": saved.files, "collision_shape": list(collision.shape),
        "status": saved["status"].tolist(), "offsets": offsets.tolist(),
        "nearest_reference_vertex": nearest.tolist(),
        "nearest_reference_distance": distances[np.arange(len(collision)), nearest].tolist(),
        "collision_positions": collision.tolist(),
    }
    for record in records:
        face = record["anchor_face"]
        matching = [int(np.nonzero(nearest == i)[0][0]) for i in faces[face]]
        triangle = collision[matching]
        collision_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        collision_normal /= np.linalg.norm(collision_normal)
        if collision_normal @ normals[face] < 0:
            collision_normal *= -1
        force = np.asarray(record["force"])
        force /= np.linalg.norm(force)
        compression = collision_normal @ force
        tangent = force - collision_normal * compression
        record["collision_facet_normal"] = collision_normal.tolist()
        record["reference_facet_normal"] = normals[face].tolist()
        record["facet_normal_difference"] = float(np.linalg.norm(collision_normal - normals[face]))
        record["collision_facet_cone_margin"] = float(record["friction"] * compression - np.linalg.norm(tangent))
result = {"scope": __doc__, "contacts": records, "source_geometry": geometry}
(output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
for record in records:
    print(json.dumps(record), flush=True)
print(json.dumps({"source_geometry_max_error": max(geometry["nearest_reference_distance"])}), flush=True)
