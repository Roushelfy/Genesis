"""Preserve exact public direct-patch trajectory failures as small native regressions."""
from pathlib import Path
import numpy as np

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
destination = Path('tests/rigid/assets')
for version, tick, env, contact in [('v2', 1076, 13, 1), ('v3', 594, 18, 1)]:
    original = root / f'online-surface-fan-gjk-patch-{version}/results/failure/snapshot-{tick}.npz'
    output = destination / f'nonconvex_mesh64_tick{tick}.npz'
    if output.exists():
        raise FileExistsError(output)
    with np.load(original) as saved:
        assert saved['valid'][env, contact] and saved['source_status'][env, contact] == 0
        assert saved['traction_status'][env, contact] == 3
        arrays = {key: saved[key][env, contact] for key in (
            'position', 'anchor', 'force', 'radius', 'friction', 'anchor_face')}
    np.savez_compressed(output, **arrays)
    print(output, flush=True)
