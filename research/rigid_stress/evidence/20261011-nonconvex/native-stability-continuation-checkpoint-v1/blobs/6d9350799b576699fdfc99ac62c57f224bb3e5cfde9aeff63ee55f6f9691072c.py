"""Independent captured convex-vertex interval/witness audit along each original collision normal."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

directory = Path(__file__).parent
with np.load(directory / "native-certified-failure-capture-v2/b4096-623103/physical-failure/before-rigid-integration.npz") as asset:
    data = {key:asset[key].copy() for key in asset.files}
rows = []
for index in np.flatnonzero(data["contacts.valid"][0]):
    slot = int(data["collider.contact_sort_idx"][0, index])
    normal = data["collider.contact_data.normal"][0, slot]
    row = {"raw_contact":int(index), "normal":normal.tolist(), "reported_penetration_m":float(data["collider.contact_data.penetration"][0, slot])}
    projections = {}
    for side in ("a", "b"):
        geom = int(data[f"collider.contact_data.geom_{side}"][0, slot])
        start, end = (int(data[f"rigid.geom_info.vert_{name}"][geom]) for name in ("start", "end"))
        vertices = data["rigid.vert_info.init_pos"][start:end]
        assert len(vertices)
        quaternion = data["rigid.geoms.quat"][0, geom]
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        world = rotation.apply(vertices) + data["rigid.geoms.pos"][0, geom]
        projections[side] = world @ normal
        witness = data[f"collider.contact_data.witness.point_{side}"][0, slot]
        row[side] = {"geom":geom, "type":int(data["rigid.geom_info.type"][geom]), "convex":bool(data["rigid.geom_info.is_convex"][geom]), "vertices":len(vertices), "local_bounds_m":[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()], "normal_interval_m":[float(projections[side].min()),float(projections[side].max())], "witness_projection_m":float(witness @ normal), "point":witness.tolist()}
    row["translation_A_along_collision_normal_m"] = float(projections["b"].max() - projections["a"].min())
    row["translation_A_opposite_collision_normal_m"] = float(projections["a"].max() - projections["b"].min())
    row["witness_B_minus_A_along_normal_m"] = float((np.array(row["b"]["point"]) - np.array(row["a"]["point"])) @ normal)
    rows.append(row)
summary = {"scope":__doc__, "environment":int(data["environment"]), "contacts":rows,
           "limitations":"Vertex intervals are exact supports for the captured discrete convex geoms. Positive interval overlap alone is not a full intersection proof;negative overlap is separation. Original forces,normals,witnesses and rigid state are unchanged."}
(directory / "producer-support-audit-v1/summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)
