import json
from pathlib import Path
import numpy as np
from research.rigid_stress.diagnose_native_traction_snapshot import cone_lp
root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex/fit-diagnosis-v1')
records = []
for path in sorted(root.glob('snapshot-*-env*-contact*.npz')):
    with np.load(path) as data:
        normals, counts = np.unique(data['inward_normals'], axis=0, return_counts=True)
        direction = data['target'][:3]
        compression = normals @ direction
        tangent = direction - compression[:, None] * normals
        cone_gap = np.linalg.norm(tangent, axis=1) - data['friction'] * compression
        force_target = data['target'].copy()
        force_target[3:] = 0.0
        force_lp, _ = cone_lp(np.zeros((len(normals),3)), normals, float(data['friction']), force_target, 128, True)
        record = {'file': path.name, 'normals': normals.tolist(), 'sample_counts': counts.tolist(), 'cone_gaps': cone_gap.tolist(), 'compression': compression.tolist(), 'force_only_outer_lp': force_lp, 'position_minus_anchor_m': (data['position']-data['anchor']).tolist(), 'dual_moment_target': data['target'][3:].tolist()}
        records.append(record)
        print(json.dumps(record), flush=True)
(root/'force-feasibility.json').write_text(json.dumps(records, indent=2)+'\n')
