"""Bounded CPU precision study of the fixed shell linear solve.

Geometry, Hertz/DPR loads, inertia relief and gauge are fixed in FP64.  The
FP32 cases round the SAME assembled stiffness/load and evaluate the linear
stress operator in FP32.  They do not test FP32 geometry/contact assembly.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu,gmres,LinearOperator
from threadpoolctl import threadpool_limits

from .core import ShellModel
from .geometry import egg_mesh
from .contacts import ShellDPRTransfer
from .performance import _contact_bank


def metrics(model,q,balanced,baseline_field,S):
    residual=model.K@np.asarray(q,dtype=np.float64)-balanced
    length=float(np.sqrt(np.mean(np.sum((model.vertices-model.center_of_mass)**2,axis=1))))
    scales=np.tile([1.,1.,1.,1./length,1./length,1./length],len(model.vertices))
    scaled_relative=np.linalg.norm(residual*scales[:,None],axis=0)/np.maximum(
        np.linalg.norm(balanced*scales[:,None],axis=0),1e-300)
    field=S@q
    field_relative=np.linalg.norm(np.asarray(field,dtype=np.float64)-baseline_field,axis=0)/np.maximum(
        np.linalg.norm(baseline_field,axis=0),1e-300)
    return dict(full_stress_relative_by_rhs=field_relative.tolist(),
                full_stress_relative_max=float(field_relative.max()),
                physical_residual_scaled_by_rhs=scaled_relative.tolist(),
                physical_residual_scaled_max=float(scaled_relative.max()),
                translation_residual_max_N=float(np.abs(residual.reshape(-1,6,residual.shape[-1])[:,:3]).max()),
                rotation_residual_max_Nm=float(np.abs(residual.reshape(-1,6,residual.shape[-1])[:,3:]).max()))


def elapsed(function,repeats=5):
    function();times=[]
    for _ in range(repeats):
        start=time.perf_counter();function();times.append((time.perf_counter()-start)*1000)
    return dict(median_ms=float(np.median(times)),range_ms=[float(min(times)),float(max(times))])


def study(n,count=8):
    start=time.perf_counter();vertices,faces=egg_mesh(n)
    model=ShellModel(vertices,faces,E=1e10,nu=.3,h=.0005)
    assembly=time.perf_counter()-start
    transfer=ShellDPRTransfer(model)
    bank=_contact_bank(model,count=count)
    forces=[];contact_info=[]
    for c in bank:
        f,info=transfer.transfer(c,order=24,method='auto')
        forces.append(f);contact_info.append(info)
    force=np.column_stack(forces)
    balanced,_=model.balance_load(force)
    factor64=model.factorize()
    free=factor64.free_dofs
    K=model.K[free][:,free].tocsc();b=balanced[free]
    q64=model.solve(force)
    S=model.stress_matrix();S32=S.astype(np.float32)
    baseline=S@q64
    result=dict(resolution=n,nodes=len(vertices),elements=len(faces),dofs=model.ndof,rhs=count,
                assembly_seconds=assembly,fp64_factor_seconds=factor64.seconds,
                contact_radius_range_m=[float(min(x['radius_m'] for x in contact_info)),float(max(x['radius_m'] for x in contact_info))],
                contact_methods=[x['method'] for x in contact_info],rows=[])
    def apply64():
        q=np.zeros_like(force);q[free]=factor64.lu.solve(np.asfortranarray(b));return q
    base=metrics(model,q64,balanced,baseline,S)
    base.update(name='fp64_baseline',solve_and_stress_time=elapsed(lambda:S@apply64()))
    result['rows'].append(base)
    D=1/np.sqrt(K.diagonal())
    K_eq=(sparse.diags(D)@K@sparse.diags(D)).tocsc()
    factors={}
    for name,matrix in [('raw_fp32',K),('equilibrated_fp32',K_eq)]:
        begun=time.perf_counter()
        try:
            factor=splu(matrix.astype(np.float32),permc_spec='MMD_AT_PLUS_A',options={'Equil':False})
            factors[name]=factor
            factor_seconds=time.perf_counter()-begun
            def apply():
                q=np.zeros(force.shape,dtype=np.float32)
                if name=='raw_fp32': q[free]=factor.solve(b.astype(np.float32))
                else: q[free]=D.astype(np.float32)[:,None]*factor.solve((D[:,None]*b).astype(np.float32))
                return q
            q=apply();row=metrics(model,q,balanced,baseline,S32)
            row.update(name=name,factor_seconds=factor_seconds,
                       factor_nonzeros=int(factor.L.nnz+factor.U.nnz),
                       solve_and_stress_time=elapsed(lambda:S32@apply()))
            result['rows'].append(row)
        except Exception as exc:
            result['rows'].append(dict(name=name,error=repr(exc)))
    if 'equilibrated_fp32' in factors:
        factor=factors['equilibrated_fp32']
        def refine():
            q=np.zeros(force.shape,dtype=np.float64)
            q[free]=D[:,None]*factor.solve((D[:,None]*b).astype(np.float32)).astype(np.float64)
            history=[]
            for iteration in range(17):
                residual=b-K@q[free]
                norm=np.linalg.norm(residual,axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-300)
                history.append(float(norm.max()))
                if norm.max()<1e-11 or iteration==16:break
                correction=factor.solve((D[:,None]*residual).astype(np.float32)).astype(np.float64)
                q[free]+=D[:,None]*correction
            return q,history
        q,history=refine();row=metrics(model,q,balanced,baseline,S)
        row.update(name='fp64_residual_refinement_with_equilibrated_fp32_factor',
                   refinement_iterations=len(history)-1,free_residual_history=history,
                   solve_and_stress_time=elapsed(lambda:S@refine()[0],repeats=3))
        result['rows'].append(row)
        # A stationary correction can diverge even when the same inaccurate
        # factor remains useful as a GMRES preconditioner.
        preconditioner=LinearOperator(K_eq.shape,
            matvec=lambda x:factor.solve(np.asarray(x,dtype=np.float32)).astype(np.float64),dtype=np.float64)
        qg=np.zeros_like(force);counts=[];infos=[];residuals=[]
        begun=time.perf_counter()
        for j in range(count):
            b_eq=D*b[:,j];counter=[0]
            def callback(value):counter[0]+=1
            y0=factor.solve(b_eq.astype(np.float32)).astype(np.float64)
            y,info=gmres(K_eq,b_eq,x0=y0,M=preconditioner,rtol=1e-10,atol=0.,restart=40,maxiter=10,
                         callback=callback,callback_type='pr_norm')
            qg[free,j]=D*y;counts.append(counter[0]);infos.append(int(info))
            residuals.append(float(np.linalg.norm(b_eq-K_eq@y)/np.linalg.norm(b_eq)))
        seconds=time.perf_counter()-begun
        row=metrics(model,qg,balanced,baseline,S)
        row.update(name='fp64_equilibrated_GMRES_with_fp32_LU_preconditioner',
                   iterations_by_rhs=counts,info_by_rhs=infos,equilibrated_relative_residual_by_rhs=residuals,
                   solve_plus_checks_single_ms=1000*seconds,restart=40,maxiter=10,rtol=1e-10)
        result['rows'].append(row)
    if n==4:
        begun=time.perf_counter();response=S@model.solve(np.eye(model.ndof));build=time.perf_counter()-begun
        response32=np.asarray(response,dtype=np.float32);force32=force.astype(np.float32)
        response64field=response@force
        projected32=response32@force32
        active=np.flatnonzero(np.any(force!=0.,axis=1))
        projected_mixed=response32[:,active].astype(np.float64)@force[active]
        def field_error(field):
            return (np.linalg.norm(np.asarray(field,dtype=np.float64)-baseline,axis=0)
                    /np.linalg.norm(baseline,axis=0)).tolist()
        result['precomputed_stress_response']=dict(
            scope='Exact FP64 response is precomputed once, then only its storage/product is rounded. This response-only output experiment does not by itself verify a recovered displacement/FEM residual.',
            build_seconds=build,shape=list(response.shape),fp64_bytes=response.nbytes,fp32_bytes=response32.nbytes,
            active_generalized_load_rows=int(len(active)),
            fp64_projection_relative_by_rhs=field_error(response64field),
            fp32_storage_fp32_product_relative_by_rhs=field_error(projected32),
            fp32_storage_fp64_selected_column_product_relative_by_rhs=field_error(projected_mixed),
            fp32_product_time=elapsed(lambda:response32@force32),
            selected_fp64_accumulation_time=elapsed(lambda:response32[:,active].astype(np.float64)@force[active]))
    return result


def main():
    import platform
    paths=('shell_cpu/core.py','shell_cpu/contacts.py','shell_cpu/performance.py','shell_cpu/precision_study.py')
    before={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    out=dict(scope='CPU linear-solve/stress precision only; fixed FP64 geometry, contact loads, inertia relief, stiffness assembly and gauge. Not FP32 end-to-end geometry or GPU TF32.',
             numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),source_sha256=before,cases=[])
    with threadpool_limits(limits=1):
        for n in (4,8):
            out['cases'].append(study(n))
            print(json.dumps(out['cases'][-1]),flush=True)
            Path('shell_cpu/review_results/precision.json').write_text(json.dumps(out,indent=2))
    out['source_unchanged_during_run']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    Path('shell_cpu/review_results/precision.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
