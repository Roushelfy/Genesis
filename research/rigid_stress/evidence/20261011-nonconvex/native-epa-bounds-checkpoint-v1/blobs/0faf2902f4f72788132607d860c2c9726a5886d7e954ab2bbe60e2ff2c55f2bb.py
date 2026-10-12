"""Publish byte-exact captured low convex collision geometry/poses and independent SAT targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

directory = Path(__file__).parent
with np.load(directory / "native-certified-failure-capture-v2/b4096-623103/physical-failure/before-rigid-integration.npz") as asset:
    saved = {key:asset[key].copy() for key in asset.files}
reference = json.loads((directory / "captured-convex-sat-audit-v1/summary.json").read_text())
start,end = (int(saved[f"rigid.geom_info.vert_{key}"][24]) for key in ("start","end"))
vertices = saved["rigid.vert_info.init_pos"][start:end]
hull = ConvexHull(vertices)
triangles = hull.simplices.copy()
edge1 = vertices[triangles[:,1]]-vertices[triangles[:,0]]
edge2 = vertices[triangles[:,2]]-vertices[triangles[:,0]]
flipped = np.einsum("ij,ij->i",np.cross(edge1,edge2),hull.equations[:,:3]) < 0
triangles[flipped] = triangles[flipped][:,[0,2,1]]
path = Path("tests/rigid/assets/nonconvex_epa_shallow_bound.npz")
np.savez_compressed(path,
    mesh_vertices=vertices,mesh_faces=triangles,
    mesh_position=saved["rigid.geoms.pos"][0,24],mesh_quaternion=saved["rigid.geoms.quat"][0,24],
    box_positions=saved["rigid.geoms.pos"][0,[14,15]],box_quaternions=saved["rigid.geoms.quat"][0,[14,15]],
    box_size=np.array([.006,.004,.006]),
    reference_normals=np.array([row["minimum_translation_A_direction"] for row in reference["contacts"]]),
    reference_penetration=np.array([row["minimum_translation_A_m"] for row in reference["contacts"]]),
    environment=saved["environment"],source_geoms=np.array([14,15,24]))
print(json.dumps({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"vertices":len(vertices),"faces":len(triangles)},indent=2),flush=True)
