"""Isolate atlas-switch jumps from actual source motion on fixed-force loads."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import gc,hashlib,json,logging
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .model import *
from .study import ANCHORS,common_probes,blended_weights,weighted_error,cases
logging.getLogger('skfem').setLevel(logging.ERROR)
ROOT=Path(__file__).resolve().parent

def run():
    points,ng=common_probes();contacts=[]
    for b in (-.0015,.0015):
        for e in (1e-6,1e-7):
            for side in (-1,1):contacts.append((b,e,side,Contact((b+side*e,.0002),.001,(.012,.004,-.05))))
        contacts.append((b,0.,0,Contact((b,.0002),.001,(.012,.004,-.05))))
    fields={};stats={}
    for name,anchors in [('joint',ANCHORS)]+[(str(i),[a]) for i,a in enumerate(ANCHORS)]:
        model=BoxModel(make_mesh(anchors,hmin=.0011,core=.0022)[0]);S=model.stress_map(points)
        stats[name]=model.storage();fields[name]={}
        for b,e,s,c in contacts:
            f,_=model.load(c,order=32);fields[name][(b,e,s)]=np.asarray(S@model.solve(f)).ravel()
        del model,S;gc.collect()
    rows=[]
    for b in (-.0015,.0015):
        left=0 if b<0 else 1
        key=(b,0.,0)
        samepoint=weighted_error(fields[str(left+1)][key],fields[str(left)][key])
        for e in (1e-6,1e-7):
            km,kp=(b,e,-1),(b,e,1)
            hardm,hardp=fields[str(left)][km],fields[str(left+1)][kp]
            blend=[]
            for key in (km,kp):
                c=Contact((key[0]+key[1]*key[2],.0002),.001,(.012,.004,-.05))
                blend.append(sum(w*fields[str(j)][key] for j,w in blended_weights(c)))
            jointdelta=weighted_error(fields['joint'][kp],fields['joint'][km])
            rows.append(dict(boundary_x_m=b,epsilon_m=e,same_point_adjacent_mesh_field_difference=samepoint,hard_relative_change=weighted_error(hardp,hardm),blend_relative_change=weighted_error(blend[1],blend[0]),joint_physical_motion_relative_change=jointdelta))
    coverage=[]
    for name,cs in cases():
        for c in cs:
            j=int(np.argmin([abs(c.center[0]-a[0]) for a in ANCHORS]));d=np.linalg.norm(np.array(c.center)-ANCHORS[j])
            coverage.append(dict(case=name,selected_mesh=j,contact_center_distance_m=float(d),radius_m=c.radius,core_radius_m=.0022,pressure_disk_within_nominal_target_core=bool(d+c.radius<=.0022),outer_support_margin_m=float(.0022-d-c.radius)))
    out=dict(scope='Same F,y,a on either side of a selector boundary; joint mesh supplies the physical-motion control, NOT continuum GT. Exact samepoint comparison isolates finite hard-switch jump. Smooth blend is an approximation, not a common-FEM equilibrium solve.',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'model.py',Path(__file__))},models=stats,rows=rows,coverage=coverage)
    (ROOT/'switch_control.json').write_text(json.dumps(out,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()))
    print(json.dumps(out['rows']),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
