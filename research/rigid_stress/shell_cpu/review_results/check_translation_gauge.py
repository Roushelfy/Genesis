"""Optional, physically equivalent translation gauge and mixed precision."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,LinearOperator,onenormest,norm
from threadpoolctl import threadpool_limits

from shell_cpu.core import ShellModel
from shell_cpu.geometry import egg_mesh
from shell_cpu.contacts import ShellDPRTransfer
from shell_cpu.performance import _contact_bank
from shell_cpu.precision_study import metrics,elapsed
from shell_cpu.review_checks import test_invariants,test_flat_patches,test_sphere


def factory(*args,**kwargs):return ShellModel(*args,**kwargs,gauge='translation')


def main():
    files=['shell_cpu/core.py','shell_cpu/contacts.py','shell_cpu/precision_study.py',__file__]
    before={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    out={'scope':'OPTIONAL distributed translation-only gauge. Stiffness/load/mass projection are unchanged; legacy remains default. This is a separate experiment from historical primary performance results.',
         'source_sha256':before,'invariants':test_invariants(factory),'patches':test_flat_patches(factory),
         'sphere':test_sphere(factory,resolutions=(4,8)),'cases':[]}
    for n in (4,8):
        v,faces=egg_mesh(n);legacy=ShellModel(v,faces,E=1e10,nu=.3,h=.0005)
        model=factory(v,faces,E=1e10,nu=.3,h=.0005)
        transfer=ShellDPRTransfer(model)
        F=np.column_stack([transfer.transfer(c,order=24)[0] for c in _contact_bank(model,count=8)])
        B,_=model.balance_load(F);factor=model.factorize();free=factor.free_dofs
        K=model.K[free][:,free].tocsc();b=B[free]
        D=1/np.sqrt(K.diagonal());A=(sparse.diags(D)@K@sparse.diags(D)).tocsc()
        lu64=splu(A,permc_spec='MMD_AT_PLUS_A');lu32=splu(A.astype(np.float32),permc_spec='MMD_AT_PLUS_A',options={'Equil':False})
        inverse=LinearOperator(A.shape,matvec=lu64.solve,rmatvec=lambda x:lu64.solve(x,trans='T'),matmat=lu64.solve,dtype=float)
        S=model.stress_matrix();ql=legacy.solve(F);q64=model.solve(F);base=S@q64
        case={'resolution':n,'dofs':model.ndof,'stiffness_identical':(model.K!=legacy.K).nnz==0,
              'mass_projection_identical':bool(np.array_equal(model.mass_modes,legacy.mass_modes)),
              'legacy_gauge':legacy.gauge_dofs.tolist(),'translation_gauge':model.gauge_dofs.tolist(),
              'equilibrated_condition_one_norm_lower_estimate':float(norm(A,1)*onenormest(inverse)),
              'legacy_vs_translation_fp64_stress_relative':float(np.linalg.norm(S@ql-base)/np.linalg.norm(base)),
              'rows':[]}
        def apply64():
            q=np.zeros_like(F);q[free]=factor.lu.solve(np.asfortranarray(b));return q
        case['fp64_solve_and_stress_time']=elapsed(lambda:S@apply64())
        def apply64equilibrated():
            q=np.zeros_like(F);q[free]=D[:,None]*lu64.solve(np.asfortranarray(D[:,None]*b));return q
        case['equilibrated_fp64_solve_and_stress_time']=elapsed(lambda:S@apply64equilibrated())
        case['equilibrated_fp64_accuracy']=metrics(model,apply64equilibrated(),B,base,S)
        case['factor_nonzeros']={
            'fp64_original_units':int(factor.lu.L.nnz+factor.lu.U.nnz),
            'fp64_equilibrated':int(lu64.L.nnz+lu64.U.nnz),
            'fp32_equilibrated':int(lu32.L.nnz+lu32.U.nnz)}
        def mixed(iterations):
            q=np.zeros_like(F);q[free]=D[:,None]*lu32.solve((D[:,None]*b).astype(np.float32)).astype(float)
            for _ in range(iterations):
                residual=b-K@q[free]
                q[free]+=D[:,None]*lu32.solve((D[:,None]*residual).astype(np.float32)).astype(float)
            return q
        for iterations in (0,1,2):
            q=mixed(iterations);row=metrics(model,q,B,base,S)
            row.update(refinement_iterations=iterations,solve_and_stress_time=elapsed(lambda:S@mixed(iterations),repeats=3))
            case['rows'].append(row)
        out['cases'].append(case)
        Path('shell_cpu/review_results/gauge_checks.json').write_text(json.dumps(out,indent=2))
        print(json.dumps(case),flush=True)
    out['source_unchanged_during_run']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    out['selected_checks_passed']=out['source_unchanged_during_run'] and all(
        c['stiffness_identical'] and c['mass_projection_identical'] and c['legacy_vs_translation_fp64_stress_relative']<1e-8
        and c['rows'][-1]['full_stress_relative_max']<1e-8 and c['rows'][-1]['physical_residual_scaled_max']<1e-8
        for c in out['cases']) and out['invariants']['near_zero_eigenvalues']==6 and out['invariants']['negative_eigenvalues']==0
    Path('shell_cpu/review_results/gauge_checks.json').write_text(json.dumps(out,indent=2))
    print({'selected_checks_passed':out['selected_checks_passed'],'stable':out['source_unchanged_during_run']},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
