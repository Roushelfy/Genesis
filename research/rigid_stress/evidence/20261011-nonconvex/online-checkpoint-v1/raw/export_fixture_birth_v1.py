import json
from pathlib import Path

import numpy as np

base = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
with np.load(base / "online-failure-capture-v1/snapshot-110.npz") as saved:
    i_b, i_c = 0, 1
    face = saved["anchor_face"][i_b, i_c]
    values = {key: saved[key][i_b, i_c] for key in ("position", "anchor", "force", "radius", "friction", "anchor_face", "geom", "pair", "method", "witness")}
with np.load(base / "online-failure-capture-v1/snapshot-110-env0-contact1.npz") as fitted:
    normals = np.unique(fitted["inward_normals"], axis=0)
    force = fitted["force"]
    compression = normals @ force
    tangent = force[None] - compression[:, None] * normals
    margins = (fitted["friction"] * compression - np.linalg.norm(tangent, axis=1)) / np.linalg.norm(force)
np.savez_compressed(base / "nonconvex_sliding_tick110.npz", **values)
print(json.dumps({"anchor_face": int(face), "normal_count": len(normals), "normal_cone_margins": margins.tolist(), "sample_normals": normals.tolist()}), flush=True)
