"""Exact static-condensation check on the independent 3D P2 box model.

The retained DOFs are the complete loaded top boundary. No reduction/truncation
of the interface is performed. The particular control has zero interior load
and a fixed bottom clamp; free-body inertia-relief loads would need their
corresponding particular response rather than assuming zero interior RHS.
"""
from pathlib import Path
import hashlib
import json
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from volume.model import BoxModel, Contact, make_mesh, HEIGHT


def run():
    source_paths=[Path(__file__),PACKAGE/'volume/model.py']
    before={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths}
    model=BoxModel(make_mesh()[0],name='coarse_condensation_control')
    top=np.flatnonzero(np.isclose(model.basis.doflocs[2],HEIGHT,rtol=0.,atol=1e-14))
    interior=np.setdiff1d(model.free,top)
    K=model.K
    A=K[interior][:,interior].tocsc()
    di=1/np.sqrt(A.diagonal())
    factor=splu((sparse.diags(di)@A@sparse.diags(di)).tocsc(),permc_spec='MMD_AT_PLUS_A')
    Kit=K[interior][:,top].toarray()
    harmonic=-di[:,None]*factor.solve(di[:,None]*Kit)
    schur=K[top][:,top].toarray()+K[top][:,interior]@harmonic
    rng=np.random.default_rng(20261024)
    cases=[]
    for j in range(6):
        c=Contact(tuple(rng.uniform(-.004,.004,2)),float(rng.uniform(.0008,.0018)),
                  (float(rng.uniform(-.02,.02)),float(rng.uniform(-.02,.02)),-.05))
        f,admit=model.load(c,order=24)
        direct=model.solve(f)
        trace=np.linalg.solve(schur,f[top])
        condensed=np.zeros_like(f);condensed[top]=trace;condensed[interior]=harmonic@trace
        outside=float(np.linalg.norm(f[interior])/max(np.linalg.norm(f),1e-30))
        cases.append(dict(contact=dict(center=c.center,radius=c.radius,force=c.force),
            force_info=admit,interior_load_relative=outside,
            displacement_relative=float(np.linalg.norm(condensed-direct)/np.linalg.norm(direct)),
            physical_residual=model.residual(condensed,f)))
    stable=all(hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p.relative_to(PACKAGE))]
               for p in source_paths)
    result=dict(scope=__doc__,dofs=model.N,retained_top_dofs=len(top),eliminated_interior_dofs=len(interior),
        schur_symmetry_relative=float(np.linalg.norm(schur-schur.T)/np.linalg.norm(schur)),
        schur_smallest_eigenvalue=float(np.linalg.eigvalsh(schur)[0]),
        rows=cases,source_sha256=before,source_unchanged_during_run=stable,
        passed=stable and all(c['interior_load_relative']<1e-12 and c['displacement_relative']<1e-9
                   and c['physical_residual']<1e-9 for c in cases))
    path=Path(__file__).with_name('condensation.json')
    path.write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
    print(json.dumps(result,default=lambda x:x.item()),flush=True)
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
