from pathlib import Path
import numpy as np

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
cases = [(147, 'online-mesh64-gjk-v1', 1, 2), (427, 'online-surface-fan-v1', 2, 0), (454, 'online-surface-fan-v2', 2, 0)]
for tick, folder, environment, contact in cases:
    destination = Path(f'tests/rigid/assets/nonconvex_mesh64_tick{tick}.npz')
    with np.load(root / folder / 'results/failure' / f'snapshot-{tick}.npz') as saved:
        values = {key: saved[key][environment, contact] for key in ['position', 'anchor', 'force', 'radius', 'friction', 'normal', 'witness', 'anchor_face', 'geom', 'pair', 'method']}
    if destination.exists():
        with np.load(destination) as old:
            for key, value in values.items():
                np.testing.assert_array_equal(old[key], value)
    else:
        np.savez_compressed(destination, **values)
    print(f'Exact original snapshot contact extracted: {destination}', flush=True)
