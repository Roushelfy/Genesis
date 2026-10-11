import json
from pathlib import Path
import numpy as np
from research.rigid_stress.diagnose_native_traction_snapshot import cone_lp
root=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex/fit-diagnosis-v1')
live=root.parent/'live-mapping-v3'
asset=np.load('examples/rigid/assets/hollow_egg/level1/elastic.npz')
vertices=asset['vertices']; triangles=asset['surface_triangles']
records=[]
for path in sorted(root.glob('snapshot-*-env*-contact*.npz')):
    with np.load(path) as data:
        tick=int(path.name.split('-')[1]); saved=np.load(live/f'snapshot-{tick}.npz')
        face=int(saved['anchor_face'][2,0]); xyz=vertices[triangles[face]]
        edge=np.stack((xyz[1]-xyz[0],xyz[2]-xyz[0]),axis=1)
        normal=np.cross(edge[:,0],edge[:,1]); normal/=np.linalg.norm(normal)
        normal=-normal if (data['inward_normals']@normal).max()<0.5 else normal
        force=data['force']; height=np.dot(data['position']-data['anchor'], normal)
        cop=data['position']-(height/(normal@force))*force
        local=np.linalg.lstsq(edge, cop-xyz[0],rcond=None)[0]
        bary=np.r_[1-local.sum(),local]
        anchor_local=np.linalg.lstsq(edge, data['anchor']-xyz[0],rcond=None)[0]
        anchor_bary=np.r_[1-anchor_local.sum(),anchor_local]
        record={'snapshot':path.name, 'face':face,'triangle':xyz.tolist(),'inward_normal':normal.tolist(),'anchor_barycentric':anchor_bary.tolist(),'cop_barycentric':bary.tolist(),'cop_shift_m':(cop-data['anchor']).tolist(),'cop_distance_m':float(np.linalg.norm(cop-data['anchor'])),'cop_on_original_face':bool((bary>0).all()),'height_m':float(height)}
        records.append(record); print(json.dumps(record),flush=True)
(root/'continuum-inspection.json').write_text(json.dumps(records,indent=2)+'\n')
