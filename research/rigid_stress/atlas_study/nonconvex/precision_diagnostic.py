"""Bounded extended-residual diagnostic on the frozen 2D assembled problem.

This leaves primary study/results untouched. The assembled K, loads and stresses
remain FP64; only residual accumulation is longdouble for this diagnostic.
"""
from pathlib import Path
import hashlib,json,numpy as np
from scipy.sparse.linalg import norm
from atlas_study.nonconvex.study import Model,grid,ANCHORS,frob,vm
OUT=Path(__file__).resolve().parent
results=json.loads((OUT/'results.json').read_text())
worst=max(results['cases'],key=lambda c:c['reference_residual'])
p=np.load(OUT/'fields.npz')['points']
m=Model(grid(.01,list(ANCHORS),True,4),p,'reference_precision')
s0,q,f=m.solve(worst['contacts']);q0=q.copy()
A=m.Kfree;AA=A.astype(np.longdouble);ff=f[m.free].astype(np.longdouble);rows=[]
for it in range(3):
 r=ff-AA@q[m.free].astype(np.longdouble)
 r64=f[m.free]-A@q[m.free]
 correction=np.zeros(m.b.N);correction[m.free]=m.scale*m.lu.solve(m.scale*np.asarray(r,float))
 ds=np.asarray(m.S@correction);s=np.asarray(m.S@q)
 rows.append(dict(iteration=it,relative_residual_fp64=float(np.linalg.norm(r64)/np.linalg.norm(ff)),relative_residual_longdouble=float(np.linalg.norm(r)/np.linalg.norm(ff)),normwise_backward=float(np.linalg.norm(r)/(norm(A)*np.linalg.norm(q[m.free])+np.linalg.norm(ff))),max_u=float(abs(q).max()),correction_full_sampled_stress_relative=float(frob(ds)/frob(s)),vm_peak_relative_change=float(abs(vm(s+ds).max()-vm(s).max())/vm(s).max()),cumulative_stress_change_relative=float(frob(np.asarray(m.S@(q-q0)))/frob(s0))))
 q+=correction
out=dict(scope='Worst primary raw-residual case, complete identical physical probe field. Diagnostic uses longdouble residual accumulation on frozen FP64 K; not a new FEM/contact model or changed primary acceptance threshold.',case=worst['name'],contacts=worst['contacts'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),study_sha256=hashlib.sha256((OUT/'study.py').read_bytes()).hexdigest(),rows=rows)
print(json.dumps(out,indent=2),flush=True)
(OUT/'precision_diagnostic.json').write_text(json.dumps(out,indent=2))
