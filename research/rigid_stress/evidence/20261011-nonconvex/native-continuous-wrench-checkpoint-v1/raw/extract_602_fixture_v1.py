"""Extract the exact saved two-source component-wrench incompatibility regression."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
original = root / 'warp-v7-online-b1024-v1/b1024-seed623201/results/failure/snapshot-602.npz'
destination = Path('tests/rigid/assets/nonconvex_mesh64_tick602.npz')
assert not destination.exists()
with np.load(original) as saved:
    indices = np.flatnonzero(saved['valid'][66])
    indices = indices[np.lexsort((saved['pair'][66, indices], saved['geom'][66, indices]))]
    assert len(indices) == 2
    fields = {key: saved[key][66, indices] for key in
              ('position', 'anchor', 'force', 'radius', 'friction', 'anchor_face', 'witness', 'normal', 'geom', 'pair', 'method')}
    assert (fields['geom'] == 24).all() and (fields['pair'] == 22).all()
    assert (fields['witness'] == fields['anchor']).all()
    np.savez_compressed(destination, **fields)
print(json.dumps({'original': str(original), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                  'destination': str(destination), 'bytes': destination.stat().st_size,
                  'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}), flush=True)
