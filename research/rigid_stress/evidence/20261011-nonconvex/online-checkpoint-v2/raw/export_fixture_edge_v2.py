from pathlib import Path

import numpy as np

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
with np.load(root / "online-mesh64-v3/results/failure/snapshot-114.npz") as saved:
    values = {key: saved[key][0, [4]] for key in ("position", "force", "radius", "friction", "geom", "pair", "method", "witness")}
np.savez_compressed(root / "nonconvex_mesh64_tick114.npz", **values)
