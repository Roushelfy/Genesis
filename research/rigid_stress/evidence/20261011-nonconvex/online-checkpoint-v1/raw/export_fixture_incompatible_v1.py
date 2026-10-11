from pathlib import Path

import numpy as np

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
with np.load(root / "online-v2/results/failure/snapshot-111.npz") as saved:
    values = {key: saved[key][0, 1] for key in ("position", "anchor", "force", "radius", "friction", "anchor_face", "geom", "pair", "method", "witness")}
np.savez_compressed(root / "nonconvex_urdf_tick111.npz", **values)
