from pathlib import Path
import numpy as np
root = Path(__file__).parent
for run, tick, env, failed_slot in [('online-square-fan-v1', 710, 5, 0), ('warp-candidate-online-v1', 1160, 2, 1)]:
    with np.load(root / run / 'results/failure' / f'snapshot-{tick}.npz') as saved:
        pair = saved['pair'][env, failed_slot]
        selected = np.flatnonzero(saved['valid'][env] & (saved['pair'][env] == pair))
        assert len(selected) == 2
        data = {name: saved[name][env, selected].copy() for name in ('position', 'anchor', 'force', 'normal', 'radius', 'friction', 'pair', 'geom', 'method', 'anchor_face', 'source_status')}
    destination = Path('tests/rigid/assets') / f'nonconvex_plane_pair_tick{tick}.npz'
    assert not destination.exists(), destination
    np.savez_compressed(destination, **data)
    print(destination)
