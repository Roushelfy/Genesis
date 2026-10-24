"""Additional finer-union and factor-strategy controls after the bounded suite."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import gc,hashlib,json,logging,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .model import *
from .study import ANCHORS,cases,common_probes,compare,weighted_error
logging.getLogger('skfem').setLevel(logging.ERROR)
ROOT=Path(__file__).resolve().parent

def run():
    out=ROOT/'refined_reference';out.mkdir(exist_ok=True)
    points,ng=common_probes();mesh,h=make_mesh(ANCHORS,hmin=.00075,core=.0022)
    print('fine mesh',mesh.nelements,flush=True)
    model=BoxModel(mesh,symmetric_factor=True);print(model.storage(),flush=True)
    S=model.stress_map(points)
    report=dict(scope='Additional finer union mesh, not continuum-contact qualification. The underlying geometry/BC/material and common probes remain unchanged. Primary atlas/coarse fields use archived general-pivot code; refined solve uses equivalent SPD-aware LU.',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'model.py',Path(__file__))},hmin_m=.00075,model=model.storage(),rows=[])
    for label,cs in cases():
        fs=[];info=[]
        for c in cs:
            f,i=model.load(c,order=48);fs.append(f);info.append(i)
        f=sum(fs);q=model.solve(f);sigma=np.asarray(S@q).ravel()
        old=np.load(ROOT/'results_bounded'/(label+'.npz'))
        row=dict(case=label,checks=info,residual=model.residual(q,f),reference_change=compare(old['reference'],sigma,ng),coarse=compare(old['coarse'],sigma,ng),hard_atlas=compare(old['hard_atlas'],sigma,ng),blended_atlas=compare(old['blended_atlas'],sigma,ng),union=compare(old['union'],sigma,ng))
        if label in ['radius_0.00055','three_asymmetric']:
            f96=sum(model.load(c,order=96)[0] for c in cs);s96=np.asarray(S@model.solve(f96)).ravel()
            row['quadrature48_vs96']=dict(load_relative=float(np.linalg.norm(f-f96)/np.linalg.norm(f96)),stress_relative=weighted_error(sigma,s96))
        report['rows'].append(row)
        np.savez_compressed(out/(label+'.npz'),points=points,reference=sigma)
        (out/'results.json').write_text(json.dumps(report,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()))
        print(label,row['reference_change']['full_sampled_tensor_relative'],row['hard_atlas']['full_sampled_tensor_relative'],flush=True)
    report['all_numerical_checks_passed']=all(r['residual']<1e-9 and all(c['force_relative']<1e-12 and c['moment_normalized']<1e-10 for c in r['checks']) for r in report['rows'])
    report['all_reference_changes_below_5_percent']=all(r['reference_change']['full_sampled_tensor_relative']<.05 for r in report['rows'])
    (out/'results.json').write_text(json.dumps(report,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()))
    return report

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
