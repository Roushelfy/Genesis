"""Compute the missing finer-reference case receipt without rewriting its old partial record."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,logging
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .model import *
from .study import ANCHORS,cases,common_probes,compare,weighted_error
logging.getLogger('skfem').setLevel(logging.ERROR)
ROOT=Path(__file__).resolve().parent

def run():
    paths=[ROOT/'model.py',ROOT/'study.py',Path(__file__)]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    partial_path=ROOT/'refined_reference/results.json';partial=json.loads(partial_path.read_text())
    assert len(partial['rows'])==15 and partial['rows'][-1]['case']=='two_near'
    points,ng=common_probes();mesh,_=make_mesh(ANCHORS,hmin=.00075,core=.0022)
    model=BoxModel(mesh,symmetric_factor=True);S=model.stress_map(points)
    label='three_asymmetric';cs=dict(cases())[label]
    fs=[];info=[]
    for c in cs:
        f,i=model.load(c,order=48);fs.append(f);info.append(i)
    f=sum(fs);q=model.solve(f);sigma=np.asarray(S@q).ravel()
    f96=sum(model.load(c,order=96)[0] for c in cs);s96=np.asarray(S@model.solve(f96)).ravel()
    old=np.load(ROOT/'results_bounded'/(label+'.npz'))
    row=dict(case=label,checks=info,residual=model.residual(q,f),reference_change=compare(old['reference'],sigma,ng),coarse=compare(old['coarse'],sigma,ng),hard_atlas=compare(old['hard_atlas'],sigma,ng),blended_atlas=compare(old['blended_atlas'],sigma,ng),union=compare(old['union'],sigma,ng),quadrature48_vs96=dict(load_relative=float(np.linalg.norm(f-f96)/np.linalg.norm(f96)),stress_relative=weighted_error(sigma,s96)))
    cached=np.load(ROOT/'refined_reference/three_asymmetric.npz')
    receipt=dict(scope='Fresh actual recomputation of formerly unpersisted case16; original15rowpartialreport retained unchanged. No synthetic/inferred numerical pass.',source_sha256=hashes,source_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name]for p in paths),partial_report_sha256=hashlib.sha256(partial_path.read_bytes()).hexdigest(),model=model.storage(),probe_array_sha256=hashlib.sha256(points.tobytes()).hexdigest(),cached_field_difference=weighted_error(sigma,cached['reference']),row=row)
    receipt['all_selected_numerical_checks_passed']=bool(receipt['source_unchanged'] and row['residual']<1e-9 and all(i['force_relative']<1e-12 and i['moment_normalized']<1e-10 for i in info))
    receipt_path=ROOT/'finer_case16_receipt.json';receipt_path.write_text(json.dumps(receipt,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()))
    np.savez_compressed(ROOT/'refined_reference/three_asymmetric_verified.npz',points=points,reference=sigma)
    combined=dict(partial);combined['rows']=list(partial['rows'])+[row]
    combined['completion_receipt']='../finer_case16_receipt.json';combined['original_partial_report']='refined_reference/results.json'
    combined['all_numerical_checks_passed']=all(r['residual']<1e-9 and all(i['force_relative']<1e-12 and i['moment_normalized']<1e-10 for i in r['checks'])for r in combined['rows'])
    combined['all_reference_changes_below_5_percent']=all(r['reference_change']['full_sampled_tensor_relative']<.05 for r in combined['rows'])
    combined['scope']='Complete16case finer-reference numerical record, combining15originalrows with one freshly recomputed receipt. All physical-reference uncertainty remains explicit.'
    (ROOT/'refinement_complete.json').write_text(json.dumps(combined,indent=2,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()))
    print(json.dumps(receipt,default=lambda a:int(a) if isinstance(a,np.integer) else a.tolist()),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
