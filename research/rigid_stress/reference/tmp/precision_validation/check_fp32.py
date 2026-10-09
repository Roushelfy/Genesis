#!/usr/bin/env python3
"""CPU single-precision recovery checks, not a GPU or TF32 benchmark.

Geometry, stiffness, mass, and contact mapping are prepared in FP64 offline.
The runtime FP32 path casts assembled external forces to FP32, balances the
six rigid modes in FP32, solves with an FP32 LU, and scans every P2 corner
in FP32. Thus this does not test FP32 element assembly or contact mapping.
"""
import json, sys, zipfile
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu
from numba import njit
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'output'),str(ROOT/'output/arbitrary_contact_cpu'),str(ROOT/'output/smooth_friction_cpu')]
import arbitrary_contact_test as test
import smooth_loads
from indexed_contacts import IndexedContacts
from general_peak import P2Peak, MID


@njit(cache=True)
def peak32(G,E,U,mu):
    best=np.float32(0.)
    two=np.float32(2.);three=np.float32(3.);four=np.float32(4.)
    for e in range(len(E)):
        origin=U[E[e,0]].copy()
        base=np.zeros(6,np.float32)
        for j in range(4):
            v=U[E[e,j]]-origin;g=G[e,j]
            base[0]-=v[0]*g[0];base[1]-=v[1]*g[1];base[2]-=v[2]*g[2]
            base[3]-=v[0]*g[1]+v[1]*g[0]
            base[4]-=v[1]*g[2]+v[2]*g[1]
            base[5]-=v[0]*g[2]+v[2]*g[0]
        for c in range(4):
            strain=base.copy()
            for j in range(4):
                node=E[e,c] if j==c else E[e,MID[c,j]]
                v=U[node]-origin;g=G[e,j]
                strain[0]+=four*v[0]*g[0];strain[1]+=four*v[1]*g[1];strain[2]+=four*v[2]*g[2]
                strain[3]+=four*(v[0]*g[1]+v[1]*g[0])
                strain[4]+=four*(v[1]*g[2]+v[2]*g[1])
                strain[5]+=four*(v[0]*g[2]+v[2]*g[0])
            a=strain[0]-strain[1];b=strain[1]-strain[2];c2=strain[2]-strain[0]
            value=two*(a*a+b*b+c2*c2)+three*(strain[3]*strain[3]+strain[4]*strain[4]+strain[5]*strain[5])
            if value>best:best=value
    return mu*np.sqrt(best)


def summarize(rows):
    active=[r for r in rows if r['reference_peak_Pa']>1.]
    errors=np.array([r['peak_relative_error_1Pa_floor'] for r in active])
    residuals=np.array([r['full_relative_residual'] for r in active])
    worst=max(active,key=lambda r:r['peak_relative_error_1Pa_floor'])
    return {'frames':len(rows),'frames_reference_peak_above_1Pa':len(active),
        'peak_error_max_percent':float(errors.max()*100),
        'peak_error_max_1Pa_floor_percent':float(max(r['peak_relative_error_1Pa_floor'] for r in rows)*100),
        'zero_reference_spurious_peak_max_Pa':float(max([r['peak_Pa'] for r in rows if r['reference_peak_Pa']==0.] or [0.])),
        'peak_error_p95_percent':float(np.percentile(errors,95)*100),
        'peak_error_mean_percent':float(errors.mean()*100),
        'fraction_above_0_1_percent':float(np.mean(errors>1e-3)),
        'fraction_above_1_percent':float(np.mean(errors>1e-2)),
        'full_residual_max_active':float(residuals.max()),
        'full_residual_p95_active':float(np.percentile(residuals,95)),
        'worst_frame':worst}


def main():
    out=Path(__file__).resolve().parent
    with threadpool_limits(limits=1):
        xyz,tet,outer,meta=test.egg.egg_mesh(2,2)
        f=test.conv.FEM(xyz,tet,outer,2,quarter=False,label='FP32-recovery-validation')
        geometry=test.SurfaceGeometry(f,10);mapper=IndexedContacts(geometry)
        p64=P2Peak(f.glambda,f.elements,f.young/(2*(1+f.poisson)))
        p64.warmup(np.zeros(f.ndof))
        G32=f.glambda.astype(np.float32);mu32=np.float32(f.young/(2*(1+f.poisson)))
        peak32(G32,f.elements,np.zeros((len(f.xyz),3),np.float32),mu32)
        K=f.k[f.free][:,f.free].tocsc();Kfull=f.k.tocsr()
        K32=K.astype(np.float32)
        factor32=splu(K32,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        # Scale the rounded FP32 matrix using diagonal congruence. Test rather
        # than assume that scaling improves this thin-shell problem.
        scale=(1/np.sqrt(K.diagonal())).astype(np.float32)
        S=sparse.diags(scale,format='csc')
        scaled_matrix=(S@K32@S).tocsc()
        scaled_factor32=splu(scaled_matrix,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        R32=f.r.astype(np.float32);MR32=f.mr.astype(np.float32)
        gram32=R32.T@MR32;gram_factor32=linalg.cho_factor(gram32,lower=True)
        def balance32(raw):
            v=raw.astype(np.float32)
            return v-MR32@linalg.cho_solve(gram_factor32,R32.T@v)
        gravity=f.m@np.tile([0.,0.,-9.81],len(f.xyz))
        cases=smooth_loads.prepare_cases(f,geometry,frames=256)
        descriptors=[]
        for name,sequence in cases.items():
            for i,desc in enumerate(sequence):
                traction,_,_=smooth_loads.assemble_descriptor(geometry,desc)
                ids=np.flatnonzero(np.any(traction!=0.,axis=1))
                raw=mapper.load_tractions(traction[ids],ids)+gravity
                descriptors.append((name,i,raw))
        random=test.workloads(f,64)['random_unrestricted_patches']
        for i,desc in enumerate(random):
            force,_=test.load_shared(geometry,desc['patches'])
            raw=force-test.centrifugal_inertia(f,np.asarray(desc['omega_rad_s']))
            descriptors.append(('random_positions_directions_counts',i,raw))
        # Amplitude scaling isolates precision issues without changing geometry
        # or assuming fixed load locations in the preceding cases.
        amplitude_seeds=[v[2] for v in descriptors[-64:-56]]
        for amp in [1e-6,1e-3,1.,1e3]:
            for i in range(8):
                descriptors.append((f'amplitude_{amp:g}',i,amplitude_seeds[i]*amp))
        paths=['fp32_LU_rhs64_rounded_peak32','fp32_runtime_balance_LU_peak',
               'fp32_scaled_runtime_balance_LU_peak','fp32_LU_one_FP64_residual_correction']
        result={'CPU':test.egg.cpu_name(),'GPU_tested':False,'TF32_tested':False,'threads':1,
            'mesh':{'DOFs':f.ndof,'tetrahedra':f.ne,'stress_converged':False,'thickness_m':.0005},
            'scope':'Full-body affine P2 shell; arbitrary patches and synthetic friction-cone-admissible snapshots, not Genesis/Coulomb dynamics.',
            'offline_precision':'Geometry/K/M/contact integration FP64. FP32 LU computed from rounded K. Runtime FP32 includes six-mode inertia relief, LU solve and all-corner peak arithmetic. FP32 element assembly/contact mapping not tested.',
            'diagnostic_precision':'Reference and reported full residuals evaluated in FP64 against original K and reference balanced RHS. Peak reference uses original FP64 LU; no refinement on pure-FP32 paths.',
            'peak_metric':'abs(peak-reference)/max(reference,1Pa); relative-error summary excludes reference below 1Pa except maximum which retains the floor.',
            'factor_nnz':{'FP64':f.factor.nnz,'FP32':factor32.nnz,'scaled_FP32':scaled_factor32.nnz},
            'cases':{},'timing':{}}
        timing_B=[]
        for j,(name,idx,raw) in enumerate(descriptors):
            rhs=f.balance(raw);active=np.linalg.norm(rhs)>1e-12
            u64=np.zeros(f.ndof)
            if active:u64[f.free]=f.factor.solve(rhs[f.free])
            reference=p64(u64)[0]
            if active and len(timing_B)<32:timing_B.append(rhs[f.free])
            b_runtime32=balance32(raw)
            for path in paths:
                b32=rhs.astype(np.float32) if path!='fp32_runtime_balance_LU_peak' and path!='fp32_scaled_runtime_balance_LU_peak' else b_runtime32
                u=np.zeros(f.ndof,np.float32)
                if np.linalg.norm(b32.astype(np.float64))>1e-12:
                    if path=='fp32_scaled_runtime_balance_LU_peak':
                        u[f.free]=scale*scaled_factor32.solve(scale*b32[f.free])
                    else:u[f.free]=factor32.solve(b32[f.free])
                if path=='fp32_LU_one_FP64_residual_correction':
                    u=u.astype(np.float64)
                    if active:
                        residual=rhs[f.free]-K@u[f.free]
                        u[f.free]+=factor32.solve(residual.astype(np.float32)).astype(np.float64)
                    peak=p64(u)[0]
                else:peak=float(peak32(G32,f.elements,u.reshape(-1,3),mu32))
                diff=Kfull@u.astype(np.float64)-rhs
                rel=float(np.linalg.norm(diff)/max(np.linalg.norm(rhs),1e-12))
                row={'frame':idx,'reference_peak_Pa':reference,'peak_Pa':peak,
                     'peak_relative_error_1Pa_floor':abs(peak-reference)/max(reference,1.),
                     'full_relative_residual':rel,
                     'FP32_balance_error_vs_rhs64':float(np.linalg.norm(b_runtime32.astype(np.float64)-rhs)/max(np.linalg.norm(rhs),1e-12))}
                result['cases'].setdefault(name,{}).setdefault(path,[]).append(row)
            if j%128==0:print(json.dumps({'processed':j,'total':len(descriptors)}),flush=True)
        aggregate={path:[] for path in paths}
        for name,data in result['cases'].items():
            for path,rows in data.items():aggregate[path]+=rows
            result['cases'][name]={'summaries':{p:summarize(rows) for p,rows in data.items()},'rows':data}
        result['overall']={p:summarize(rows) for p,rows in aggregate.items()}
        B=np.asfortranarray(np.column_stack(timing_B));B32=B.astype(np.float32,order='F')
        # Matched ordering, timings exclude mapping/balancing/postprocessing.
        for count in [1,32]:
            bb=B[:,:count];bb32=B32[:,:count]
            methods={'FP64':lambda:f.factor.solve(bb),'FP32':lambda:factor32.solve(bb32),
                'FP32_plus_one_FP64_residual':lambda:(lambda y:y+factor32.solve((bb-K@y).astype(np.float32)).astype(np.float64))(factor32.solve(bb32).astype(np.float64))}
            for label,fn in methods.items():
                fn();times=[]
                for repeat in range(9):
                    t=perf_counter();fn();times.append(perf_counter()-t)
                result['timing'][f'{label}_B{count}']=test.stats(times)
        (out/'fp32_precision_results.json').write_text(json.dumps(result,indent=2))
        lines=[result['scope'],result['offline_precision'],'Not a GPU or TF32 measurement.','']
        for path,summary in result['overall'].items():lines.append(path+'\n'+json.dumps(summary,indent=2))
        lines.append('\nSolve-only timings:\n'+json.dumps(result['timing'],indent=2))
        (out/'fp32_precision_report.txt').write_text('\n'.join(lines))
        destination=ROOT/'output/fp32_stress_validation_cpu.zip'
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
            for filename in ['check_fp32.py','fp32_precision_results.json','fp32_precision_report.txt']:
                z.write(out/filename,'tmp/precision_validation/'+filename)
            for p in ['output/arbitrary_contact_cpu/arbitrary_contact_test.py','output/arbitrary_contact_cpu/general_peak.py',
                      'output/arbitrary_contact_cpu/indexed_contacts.py','output/smooth_friction_cpu/smooth_loads.py',
                      'output/eggshell_gripper_cpu.py','output/eggshell_convergence_cpu.py',
                      'output/rigid_stress_reference.py','output/rigid_stress_temporal_cpu_test.py']:
                file=ROOT/p
                if file.exists():z.write(file,p)
            z.writestr('README.txt','CPU precision validation. Install numpy scipy numba threadpoolctl.\nRun: python tmp/precision_validation/check_fp32.py\nOffline geometry/stiffness/contact mapping remain FP64; runtime recovery paths are identified separately. No TF32 or GPU test.\nFor peak_error_max_percent/p95/mean, only frames with reference peak >1Pa are included. Zero-reference false stress is reported in Pa.\n')
        print(json.dumps({'overall':result['overall'],'timing':result['timing'],'zip':str(destination)}),flush=True)


if __name__=='__main__':main()
