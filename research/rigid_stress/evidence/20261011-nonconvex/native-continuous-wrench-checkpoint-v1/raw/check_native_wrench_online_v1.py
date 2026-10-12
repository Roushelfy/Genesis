"""Validate the live original-component failure, unchanged exact input and every completed CPU capture."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
run = root / 'native-wrench-online-b1024-v1'
summary = json.loads((run / 'results/summary.json').read_text())
assert summary['completed_steps'] == 602 and len(summary['oracle_captures']) == 4
assert summary['source_sha256'] == summary['source_sha256_after']
assert (run / 'source.sha256').read_bytes() == (run / 'source-after.sha256').read_bytes()
original = root / 'warp-v7-online-b1024-v1/b1024-seed623201/results/failure/snapshot-602.npz'
current = run / 'results/failure/snapshot-602.npz'
with np.load(original) as saved, np.load(current) as new:
    failure = np.argwhere(new['raw_status'] != 0)
    assert failure.tolist() == [[66, 1]]
    assert new['raw_status'][66, 1] == 36
    assert new['traction_status'][66, 1] == 6 and new['source_status'][66, 1] == 0
    active = np.flatnonzero(new['valid'][66])
    assert np.array_equal(active, np.flatnonzero(saved['valid'][66]))
    comparisons = {key: np.array_equal(new[key][66, active], saved[key][66, active]) for key in
                   ('position','force','normal','radius','friction','witness','anchor','anchor_face','geom','pair','method')}
    assert all(comparisons.values()), comparisons
    ray = new['certificate'][66, 1]
    point, anchor, force, radius = (new[key][66, 1] for key in ('position','anchor','force','radius'))
    moment = np.cross(point-anchor, force)
    scale = max(np.linalg.norm(force), np.linalg.norm(moment)/radius)
    target = np.concatenate((force,moment/radius))/scale
    assert target @ ray >= new['certificate_margin'][66, 1] > 1e-3
captures = []
for tick in (0, 160, 320, 480):
    with np.load(run / f'results/snapshot-{tick}.npz') as saved:
        assert (saved['status'][saved['valid']] == 0).all()
        assert (saved['full_residual'] <= saved['full_residual_budget']).all()
        captures.append({'tick': tick, 'environments': len(saved['full_residual']),
                         'maximum_full_residual_N': float(saved['full_residual'].max())})
result = {'completed_steps': 602, 'completed_env_steps': 602*1024, 'failure': {'environment':66,'contact':1,'status':36},
          'original_input_bitwise_checks':comparisons,'cpu_full_fields':sum(c['environments'] for c in captures),
          'captures':captures,'failure_sha256':hashlib.sha256(current.read_bytes()).hexdigest(),
          'scope':'Native every-step diagnostic validation; abort and all completed fields disclosed, no throughput or complete B1024 acceptance claim.'}
(run / 'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
