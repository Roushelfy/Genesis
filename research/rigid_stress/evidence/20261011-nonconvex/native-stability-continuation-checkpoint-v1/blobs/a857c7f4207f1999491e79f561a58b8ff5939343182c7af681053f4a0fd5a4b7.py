"""Independent frame audit: original collision cone versus owner-oriented material boundary cone."""
import json
from pathlib import Path

import numpy as np

from audit_continuum_wrench_v2 import closest_points

root = Path(__file__).parent
with np.load(root / "native-wrench-fan-failure-capture-v1/b4096-623103/failure/before-rigid-integration.npz") as asset:
    data = {key: asset[key].copy() for key in asset.files}
with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
    vertices, faces, tets = (asset[key].copy() for key in ("vertices", "surface_triangles", "tetrahedra"))
triangles = vertices[faces]
normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
normals /= np.linalg.norm(normals, axis=1)[:, None]
contains = (faces[:, None, :, None] == tets[None, :, None, :]).any(axis=3).all(axis=2)
assert np.all(contains.sum(axis=1) == 1)
owners = tets[contains.argmax(axis=1)]
opposite = owners[(owners[:, :, None] != faces[:, None]).all(axis=2)]
normals[np.einsum("fi,fi->f", normals, vertices[opposite] - triangles[:, 0]) < 0] *= -1
rows = []
for index in np.flatnonzero(data["contacts.valid"][0]):
    point, force, normal, anchor = (data[f"contacts.{key}"][0, index] for key in ("position", "force", "normal", "center"))
    mu = float(data["contacts.friction"][0, index])
    face = int(data["contacts.anchor_face"][0, index])
    material_normal = normals[face]
    witness = data["raw_sources.witness"][0, index]
    distances = np.linalg.norm(closest_points(witness, triangles) - witness, axis=1)
    nearby_faces = np.flatnonzero(distances < 1e-12)
    assert face in nearby_faces
    offset = point - anchor
    row = {"raw_contact": int(index), "point": point.tolist(), "force": force.tolist(),
           "normal": normal.tolist(), "material_inward_normal": material_normal.tolist(),
           "material_collision_normal_dot": float(normal @ material_normal), "mu": mu,
           "anchor": anchor.tolist(), "witness": witness.tolist(),
           "witness_method": int(data["raw_sources.method"][0, index]),
           "vertex": int(data["raw_sources.vertex"][0, index]), "pair": int(data["raw_sources.pair"][0, index]),
           "geom": int(data["raw_sources.geom"][0, index]), "anchor_face": face,
           "all_witness_faces": nearby_faces.tolist(), "witness_boundary_distance_m": float(distances.min()),
           "point_anchor_distance_m": float(np.linalg.norm(offset)),
           "point_normal_offset_m": float(offset @ normal),
           "point_material_inward_offset_m": float(offset @ material_normal)}
    for name, direction in (("original_collision", normal), ("anchor_material", material_normal)):
        compression = float(force @ direction)
        tangent = float(np.linalg.norm(force - compression * direction))
        row[f"{name}_compression_N"] = compression
        row[f"{name}_tangent_N"] = tangent
        row[f"{name}_cone_excess_N"] = tangent - mu * compression
    rows.append(row)
summary = {"scope": __doc__, "environment": int(data["environment"]), "contacts": rows,
           "interpretation": "No normal/force/friction/anchor edits. The owner-oriented material inward normal is determined from the unique owning tetrahedron, independently of collision normal and link COM. Collision-cone feasibility alone does not establish feasibility on the witnessed material boundary. Rigid producer geometry requires separate review if directions disagree."}
output = root / "failed-rigid-contact-frame-audit-v2"
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
