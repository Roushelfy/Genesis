from pathlib import Path
import numpy as np

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
destination = Path('tests/rigid/assets/nonconvex_mesh64_tick147.npz')
with np.load(root / 'online-mesh64-gjk-v1/results/failure/snapshot-147.npz') as saved:
    values = {key: saved[key][1, 2] for key in ['position', 'anchor', 'force', 'radius', 'friction', 'normal', 'witness', 'anchor_face', 'geom', 'pair', 'method']}
if destination.exists():
    with np.load(destination) as old:
        for key, value in values.items():
            np.testing.assert_array_equal(old[key], value)
else:
    np.savez_compressed(destination, **values)
print(f'Exact original snapshot contact extracted: {destination}', flush=True)
