import json
from pathlib import Path

import numpy as np


raw = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex/live-mapping-v1')
asset_path = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery/examples/rigid/assets/hollow_egg/level1/elastic.npz')
with np.load(asset_path) as asset:
    triangle_positions = asset['vertices'][asset['surface_triangles']]
origin = triangle_positions[:, 0]
edges = np.stack((triangle_positions[:, 1] - origin, triangle_positions[:, 2] - origin), axis=-1)
dual = np.linalg.pinv(edges)
reports = []
for filename in sorted(raw.glob('snapshot-*.npz')):
    with np.load(filename) as snapshot:
        loaded = snapshot['valid'] & (np.linalg.norm(snapshot['force'], axis=2) > 1e-12)
        for env, contact in np.argwhere(loaded & (snapshot['traction_status'] != 0)):
            witness = snapshot['witness'][env, contact]
            local = np.einsum('fji,fi->fj', dual, witness - origin)
            projected = origin + np.einsum('fij,fj->fi', edges, local)
            barycentric = np.column_stack((1.0 - local.sum(axis=1), local))
            distances = np.where((barycentric >= 0).all(axis=1), np.linalg.norm(projected - witness, axis=1), np.inf)
            segments = triangle_positions[:, [[0, 1], [1, 2], [2, 0]]]
            direction = segments[:, :, 1] - segments[:, :, 0]
            fraction = np.clip(np.einsum('fei,fei->fe', witness - segments[:, :, 0], direction) / (direction**2).sum(axis=2), 0.0, 1.0)
            closest = segments[:, :, 0] + fraction[:, :, None] * direction
            distances = np.minimum(distances, np.linalg.norm(closest - witness, axis=2).min(axis=1))
            row = {
                'snapshot': filename.name,
                'env': int(env), 'contact': int(contact),
                'geom': int(snapshot['geom'][env, contact]),
                'pair': int(snapshot['pair'][env, contact]),
                'method': int(snapshot['method'][env, contact]),
                'source_status': int(snapshot['source_status'][env, contact]),
                'traction_status': int(snapshot['traction_status'][env, contact]),
                'evaluations': int(snapshot['evaluations'][env, contact]),
                'witness': witness.tolist(),
                'original_point': snapshot['position'][env, contact].tolist(),
                'force': snapshot['force'][env, contact].tolist(),
                'radius': float(snapshot['radius'][env, contact]),
                'mu': float(snapshot['friction'][env, contact]),
                'nearest_distance_m': float(distances.min()),
                'nearest_face': int(distances.argmin()),
                'anchor': snapshot['anchor'][env, contact].tolist(),
            }
            reports.append(row)
(raw / 'failure-diagnosis.json').write_text(json.dumps(reports, indent=2) + '\n')
print(json.dumps(reports, indent=2), flush=True)
