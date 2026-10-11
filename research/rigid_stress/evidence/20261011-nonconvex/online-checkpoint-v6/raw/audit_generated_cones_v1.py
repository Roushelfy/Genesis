"""Audit actual generated normals, friction feasibility and witness support-plane gaps."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
with np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz') as mesh:
    vertices = mesh['vertices']
records = []
for run, tick in [('online-square-fan-v1', 710), ('warp-candidate-online-v1', 1160)]:
    with np.load(root / run / 'results/failure' / f'snapshot-{tick}.npz') as saved:
        failed = saved['valid'] & (saved['traction_status'] == 3) & (saved['source_status'] == 0)
        for env, contact in zip(*np.nonzero(failed)):
            normal, force, witness, anchor = (saved[key][env, contact] for key in ('normal', 'force', 'witness', 'anchor'))
            friction = float(saved['friction'][env, contact])
            unit_force = force / np.linalg.norm(force)
            compression = normal @ unit_force
            tangent = unit_force - compression * normal
            gap = witness @ normal - (vertices @ normal).min()
            records.append({'run': run, 'tick': tick, 'env': int(env), 'contact': int(contact),
                'normal': normal.tolist(), 'normal_norm_error': float(abs(np.linalg.norm(normal)-1)),
                'force': force.tolist(), 'friction': friction,
                'produced_normal_normalized_cone_margin': float(friction*compression-np.linalg.norm(tangent)),
                'witness_support_gap_m': float(gap), 'anchor_witness_difference_m': float(np.linalg.norm(anchor-witness)),
                'anchor_face': int(saved['anchor_face'][env, contact])})
result = {'scope': __doc__, 'contacts': records}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
