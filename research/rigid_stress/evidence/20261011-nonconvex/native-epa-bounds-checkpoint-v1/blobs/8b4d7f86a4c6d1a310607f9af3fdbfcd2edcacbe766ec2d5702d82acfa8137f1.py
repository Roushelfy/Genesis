"""Independent FP64 convex polytope minimum separating translation for captured rigid geom pairs."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

directory = Path(__file__).parent
output = directory / "captured-convex-sat-audit-v1"
with np.load(directory / "native-certified-failure-capture-v2/b4096-623103/physical-failure/before-rigid-integration.npz") as asset:
    data = {key:asset[key].copy() for key in asset.files}
rows = []
for contact in np.flatnonzero(data["contacts.valid"][0]):
    slot = int(data["collider.contact_sort_idx"][0,contact])
    geom_a,geom_b = (int(data[f"collider.contact_data.geom_{side}"][0,slot]) for side in ("a","b"))
    origin = 0.5 * (data["rigid.geoms.pos"][0,geom_a]+data["rigid.geoms.pos"][0,geom_b])
    vertices,normals,edges = [],[],[]
    for geom in (geom_a,geom_b):
        start,end = (int(data[f"rigid.geom_info.vert_{name}"][geom]) for name in ("start","end"))
        quaternion = data["rigid.geoms.quat"][0,geom]
        world = Rotation.from_quat(quaternion[[1,2,3,0]]).apply(data["rigid.vert_info.init_pos"][start:end]) + data["rigid.geoms.pos"][0,geom] - origin
        hull = ConvexHull(world)
        vertices.append(world)
        normals.append(hull.equations[:,:3])
        labels = np.unique(np.sort(np.concatenate([hull.simplices[:,[0,1]],hull.simplices[:,[1,2]],hull.simplices[:,[2,0]]]),axis=1),axis=0)
        edges.append(world[labels[:,1]]-world[labels[:,0]])
    crosses = np.cross(edges[0][:,None],edges[1][None,:]).reshape(-1,3)
    lengths = np.linalg.norm(crosses,axis=1)
    # Extra triangulation diagonals contribute valid separating-axis candidates and cannot lower the true MTD.
    axes = np.concatenate([normals[0],normals[1],crosses[lengths>1e-24]/lengths[lengths>1e-24,None]])
    projections = [world @ axes.T for world in vertices]
    along = projections[1].max(axis=0)-projections[0].min(axis=0)
    opposite = projections[0].max(axis=0)-projections[1].min(axis=0)
    assert min(along.min(),opposite.min()) > 0
    candidates = np.concatenate([along,opposite])
    winner = int(np.argmin(candidates))
    normal = axes[winner%len(axes)] * (1 if winner<len(axes) else -1)
    original = data["collider.contact_data.normal"][0,slot]
    row = {"geom_a":geom_a,"geom_b":geom_b,"axis_count":len(axes),
           "minimum_translation_A_m":float(candidates[winner]),"minimum_translation_A_direction":normal.tolist(),
           "reported_normal":original.tolist(),"normal_dot_reference":float(normal @ original),
           "reported_penetration_m":float(data["collider.contact_data.penetration"][0,slot])}
    rows.append(row)
summary = {"scope":__doc__,"environment":int(data["environment"]),"contacts":rows,
           "limitations":"Reference is for the exact captured discrete convex collision geoms, not the hollow material footprint. It preserves rigid inputs and derives support directions from all face/edge candidates independently of GJK/EPA."}
(output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)
