#!/usr/bin/env python3
"""Unrestricted-load fixed-factor ordering, RHS batch and refinement checks."""
import json,sys,gc
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
import arbitrary_contact_test as test
import general_peak as gp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import eggshell_p2_converged as nd


def main():
    with threadpool_limits(limits=1):
        xyz,tet,outer,_=test.egg.egg_mesh(2,2)
        f=test.conv.FEM(xyz,tet,outer,2,quarter=False,label='fullbody-uncorrelated-RHS-solvers')
        f.wall_layers=2;g=test.SurfaceGeometry(f,10)
        seq=test.workloads(f,16)['random_unrestricted_patches'];rhs=[];truth=[]
        peak=gp.P2Peak(f.glambda,f.elements,f.young/(2*(1+f.poisson)));peak.warmup(np.zeros(f.ndof))
        for d in seq:
            contact,_=test.load_shared(g,d['patches']);raw=contact-test.centrifugal_inertia(f,np.asarray(d['omega_rad_s']))
            u,b,_=f.solve(raw);rhs.append(b);truth.append(peak(u)[0])
        Bfull=np.column_stack(rhs);K=f.k[f.free][:,f.free].tocsc();B=np.asfortranarray(Bfull[f.free])
        result={'GPU_tested':False,'scope':'Full P2 egg9630DOF, random arbitrary contacts; coarse/unconverged. RHS construction and peak evaluation excluded from solve timing.',
            'records':[],'FP32_refinement':None}
        for kind in ['MMD_AT_PLUS_A','COLAMD','surface_column_ND']:
            t=perf_counter()
            factor=nd.OrderedFactor(f) if kind=='surface_column_ND' else splu(K,permc_spec=kind,diag_pivot_thresh=0.,options={'SymmetricMode':True})
            setup=perf_counter()-t
            def solve(x):return factor.solve(np.asfortranarray(x))
            solve(B[:,:1])
            for count in [1,8,32]:
                ids=np.arange(count)%B.shape[1];bb=np.asfortranarray(B[:,ids]);expected=np.asarray(truth)[ids]
                solve(bb);times=[]
                for _ in range(7):
                    t=perf_counter();u_free=solve(bb);times.append(perf_counter()-t)
                u=np.zeros((f.ndof,count));u[f.free]=u_free
                residual=f.k@u-Bfull[:,ids]
                rel=float(np.max(np.linalg.norm(residual,axis=0)/np.linalg.norm(Bfull[:,ids],axis=0)))
                values=np.array([peak(u[:,i])[0] for i in range(count)]);error=float(np.max(abs(values/expected-1)))
                assert rel<1e-8 and error<1e-8
                stats=test.stats(times);stats['env_solves_per_second']=count/(stats['mean_ms']/1000)
                record={'ordering':kind,'batch':count,'timing':stats,'relative_full_residual':rel,'peak_relative_error':error,
                    'factor_setup_s':setup,'factor_nnz':int(factor.nnz),'precision':'FP64'}
                result['records'].append(record);print(json.dumps(record),flush=True)
            del factor;gc.collect()
        # Single precision is a measured candidate; ill-conditioned shell may
        # fail iterative refinement, which must be recorded rather than hidden.
        t=perf_counter()
        try:
            factor=splu(K.astype(np.float32),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
            setup=perf_counter()-t;records=[]
            for j in range(8):
                b=B[:,j];start=perf_counter();y=factor.solve(b.astype(np.float32)).astype(np.float64);history=[]
                for iteration in range(9):
                    r=b-K@y;rel=float(np.linalg.norm(r)/np.linalg.norm(b));history.append(rel)
                    if rel<1e-9 or iteration==8:break
                    y+=factor.solve(r.astype(np.float32)).astype(np.float64)
                elapsed=perf_counter()-start;u=np.zeros(f.ndof);u[f.free]=y
                err=abs(peak(u)[0]/truth[j]-1.)
                records.append({'RHS':j,'residual_history':history,'peak_relative_error':err,'elapsed_ms':elapsed*1000,
                    'passed':rel<1e-9 and err<1e-7})
            result['FP32_refinement']={'factor_setup_s':setup,'max_corrections':8,'records':records,
                'all_passed':all(r['passed'] for r in records),'scope':'Random arbitrary loads; zero unsupported convergence claims'}
        except Exception as exc:
            result['FP32_refinement']={'all_passed':False,'factor_error':str(exc)}
        (HERE/'arbitrary_contact_solve_results.json').write_text(json.dumps(result,indent=2))
        print('Full-load solve tests completed.',flush=True)


if __name__=='__main__':main()
