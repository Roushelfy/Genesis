#!/usr/bin/env python3
"""CPU ablations: shared-factor batches, failed-env compaction, cached history.

All pipeline paths use FP64 and the same residual tolerance. The pure-FP32
microbenchmarks only measure batching/compaction, not an accuracy guarantee.
Independent environment histories; no fixed contact basis or stress hotspot.
"""
import argparse, json, sys, platform, zipfile
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'output'),str(ROOT/'output/arbitrary_contact_cpu'),str(ROOT/'output/smooth_friction_cpu')]
import temporal_benchmark as tb
from general_peak import P2Peak


class CachedHistory:
    """FIFO slots, fixed storage, cached (KQ)^T KQ; residual-metric projection."""
    def __init__(self,K,capacity):
        self.K=K;self.capacity=capacity;self.order=[]
        self.Q=np.zeros((K.shape[0],capacity),order='F')
        self.KQ=np.zeros_like(self.Q,order='F')
        self.gram=np.zeros((capacity,capacity))
        self.gram_rebuilds=0;self.gram_column_updates=0;self.copy_shift_bytes=0

    def coefficients(self,right):
        coefficients=np.zeros(self.capacity)
        if self.order:
            ids=np.asarray(self.order,dtype=np.int32)
            coefficients[ids]=linalg.solve(self.gram[np.ix_(ids,ids)],right[ids],assume_a='sym')
        return coefficients

    def project(self,u,residual):
        if self.order:u+=self.Q@self.coefficients(self.KQ.T@residual)

    def append_direction(self,delta):
        if not self.capacity:return
        if len(self.order)==self.capacity:slot=self.order.pop(0)
        else:slot=next(i for i in range(self.capacity) if i not in self.order)
        q=delta.copy();aq=self.K@q
        original=float(aq@aq)
        if not np.isfinite(original) or original<=0:return
        for _ in range(2):
            if self.order:
                coeff=self.coefficients(self.KQ.T@aq)
                q-=self.Q@coeff;aq-=self.KQ@coeff
        norm2=float(aq@aq)
        if norm2<=1e-12*original or not np.isfinite(norm2):return
        norm=np.sqrt(norm2)
        self.Q[:,slot]=q/norm;self.KQ[:,slot]=aq/norm
        column=self.KQ.T@self.KQ[:,slot]
        self.gram[:,slot]=column;self.gram[slot,:]=column
        self.order.append(slot);self.gram_column_updates+=1


class LegacyHistory:
    def __init__(self,f,peak,capacity):
        self.solver=tb.TemporalRecovery(f,peak,'extrapolate',capacity,1e-3,projection_metric='residual')
        self.gram_rebuilds=0;self.copy_shift_bytes=0;self.gram_column_updates=0

    def project(self,u,residual):
        s=self.solver
        if s.Q.shape[1]:
            self.gram_rebuilds+=1
            gram=s.KQ.T@s.KQ
            coefficient=linalg.solve(gram,s.KQ.T@residual,assume_a='sym')
            u+=s.Q@coefficient

    def append_direction(self,delta):
        s=self.solver
        if s.Q.shape[1]>=s.capacity:self.copy_shift_bytes+=2*s.f.ndof*max(s.capacity-1,0)*8
        self.gram_rebuilds+=2 if s.Q.shape[1] else 0
        s.append_direction(delta)


class BatchRecovery:
    def __init__(self,f,peak,environments,capacity=8,rtol=1e-3,history='cached',strategy='compact',direct=False):
        self.f=f;self.K=f.k.tocsr();self.peak=peak;self.B=environments
        self.capacity=capacity;self.rtol=rtol;self.strategy=strategy;self.direct=direct
        self.history=[CachedHistory(self.K,capacity) if history=='cached' else LegacyHistory(f,peak,capacity) for _ in range(environments)]
        self.last=self.before=None

    def step(self,rhs):
        start=perf_counter();stage={name:0. for name in ['predict_project_check','pack','factor','scatter','history','peak']}
        norms=np.linalg.norm(rhs,axis=0);active=norms>1e-12
        tolerance=np.maximum(1e-12,self.rtol*norms)
        t=perf_counter()
        if self.direct or self.last is None:u=np.zeros_like(rhs,order='F')
        elif self.before is None:u=self.last.copy(order='F')
        else:u=np.asfortranarray(2*self.last-self.before)
        u[:,~active]=0
        residual=rhs if self.direct else rhs-self.K@u
        if not self.direct:
            for env in np.flatnonzero(active):self.history[env].project(u[:,env],residual[:,env])
            residual=rhs-self.K@u
        failed=active if self.direct else active&(np.linalg.norm(residual,axis=0)>tolerance)
        ids=np.flatnonzero(failed)
        stage['predict_project_check']=perf_counter()-t
        corrected_columns=0
        if len(ids):
            t=perf_counter()
            selected=np.arange(self.B) if self.strategy=='dense_zero' else ids
            packed=np.asfortranarray(residual[np.ix_(self.f.free,selected)])
            if self.strategy=='dense_zero':packed[:,~failed]=0
            stage['pack']=perf_counter()-t
            t=perf_counter();free_delta=self.f.factor.solve(packed);stage['factor']=perf_counter()-t
            corrected_columns=len(selected)
            t=perf_counter();delta=np.zeros_like(rhs,order='F')
            delta[np.ix_(self.f.free,selected)]=free_delta
            u+=delta;stage['scatter']=perf_counter()-t
            if not self.direct:
                t=perf_counter()
                for env in ids:self.history[env].append_direction(delta[:,env])
                stage['history']=perf_counter()-t
        t=perf_counter();peaks=np.asarray([self.peak(u[:,env])[0] for env in range(self.B)])
        stage['peak']=perf_counter()-t
        elapsed=perf_counter()-start
        self.before,self.last=self.last,u
        return u,peaks,{'seconds':elapsed,'stage_seconds':stage,'failed':len(ids),'active':int(active.sum()),
                        'solved_columns':corrected_columns,'factor_calls':int(bool(len(ids)))}


def measure(fn,repeats=5):
    fn();values=[]
    for _ in range(repeats):
        t=perf_counter();fn();values.append(perf_counter()-t)
    return {'median_ms':float(np.median(values)*1000),'min_ms':float(min(values)*1000),
            'max_ms':float(max(values)*1000),'samples_ms':[float(v*1000) for v in values]}


def solve_difference(f,peak,expected,actual):
    relative=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
    stress_errors=[]
    for i in range(expected.shape[1]):
        a=np.zeros(f.ndof);a[f.free]=expected[:,i]
        b=np.zeros(f.ndof);b[f.free]=actual[:,i]
        pa=peak(a)[0];pb=peak(b)[0]
        stress_errors.append(abs(pb-pa)/max(pa,1.))
    return {'relative_displacement_l2':relative,'max_peak_difference_percent':float(max(stress_errors,default=0.)*100)}


def microbenchmarks(f,peak,Bbase,Ubase,quick):
    result={'batch':[],'compaction':[],'history':[]}
    K=f.k[f.free][:,f.free].tocsc()
    factor32=splu(K.astype(np.float32),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
    rng=np.random.default_rng(321)
    # Select changing loaded snapshots; do not repeat a single RHS to fill B.
    source=Bbase[:,np.linalg.norm(Bbase,axis=0)>1e-12]
    for precision,factor,dtype in [('FP64',f.factor,np.float64),('FP32',factor32,np.float32)]:
        for count in ([1,8,32] if quick else [1,4,8,16,32,64,128]):
            ids=rng.choice(source.shape[1],count,replace=False)
            rhs=np.asfortranarray(source[f.free][:,ids],dtype=dtype)
            serial=lambda:np.column_stack([factor.solve(rhs[:,i]) for i in range(count)])
            batch=lambda:factor.solve(rhs)
            expected=serial();actual=batch()
            difference=solve_difference(f,peak,expected,actual)
            if precision=='FP64':assert difference['max_peak_difference_percent']<1e-5,difference
            serial_time=measure(serial,3 if quick else 5);batch_time=measure(batch,3 if quick else 5)
            result['batch'].append({'precision':precision,'B':count,'serial':serial_time,'batch':batch_time,
                'speedup':serial_time['median_ms']/batch_time['median_ms'],
                'batch_RHS_per_second':1000*count/batch_time['median_ms'],'serial_batch_difference':difference})
        for count in ([32] if quick else [32,128]):
            ids=rng.choice(source.shape[1],count,replace=False)
            rhs=np.asfortranarray(source[f.free][:,ids],dtype=dtype)
            for fraction in ([.25,1.] if quick else [0.,.125,.25,.5,1.]):
                mask=np.zeros(count,bool);mask[rng.choice(count,int(round(count*fraction)),replace=False)]=True
                selected=np.flatnonzero(mask);dense_rhs=rhs.copy(order='F');dense_rhs[:,~mask]=0
                def dense():
                    out=np.zeros_like(rhs,order='F')
                    if len(selected):out[:]=factor.solve(dense_rhs)
                    return out
                def compact():
                    out=np.zeros_like(rhs,order='F')
                    if len(selected):
                        packed=np.asfortranarray(rhs[:,selected]);out[:,selected]=factor.solve(packed)
                    return out
                difference=solve_difference(f,peak,dense(),compact())
                if precision=='FP64':assert difference['max_peak_difference_percent']<1e-5,difference
                d=measure(dense,3 if quick else 5);c=measure(compact,3 if quick else 5)
                pack=measure(lambda:np.asfortranarray(rhs[:,selected]),3 if quick else 9)
                result['compaction'].append({'precision':precision,'B':count,'failed_fraction':fraction,
                    'failed_count':len(selected),'dense_zero':d,'compact':c,'pack_only':pack,
                    'speedup':d['median_ms']/c['median_ms'],'dense_compact_difference':difference})
    for capacity in [4,8]:
        old=LegacyHistory(f,peak,capacity);new=CachedHistory(f.k.tocsr(),capacity)
        seed_ids=rng.choice(Ubase.shape[1],32,replace=False)
        directions=[Ubase[:,seed_ids[i]]-Ubase[:,seed_ids[i+1]] for i in range(31)]
        for delta in directions[:capacity]:old.append_direction(delta);new.append_direction(delta)
        residual=Bbase[:,int(seed_ids[-1])]
        def project(history):
            out=np.zeros(f.ndof);history.project(out,residual);return out
        a=project(old);b=project(new)
        difference=float(np.linalg.norm(f.k@(a-b))/max(np.linalg.norm(residual),1e-12))
        assert difference<1e-7,difference
        p_old=measure(lambda:project(old),20 if quick else 100)
        p_new=measure(lambda:project(new),20 if quick else 100)
        t_old=[];t_new=[]
        for delta in directions[capacity:]:
            t=perf_counter();old.append_direction(delta);t_old.append(perf_counter()-t)
            t=perf_counter();new.append_direction(delta);t_new.append(perf_counter()-t)
        a=project(old);b=project(new)
        difference_after=float(np.linalg.norm(f.k@(a-b))/max(np.linalg.norm(residual),1e-12))
        assert difference_after<1e-6,difference_after
        result['history'].append({'capacity':capacity,'project_legacy':p_old,'project_cached':p_new,
            'project_speedup':p_old['median_ms']/p_new['median_ms'],
            'append_legacy_median_ms':float(np.median(t_old)*1000),
            'append_cached_median_ms':float(np.median(t_new)*1000),
            'projection_difference_Knorm_before':difference,'projection_difference_Knorm_after':difference_after,
            'legacy_gram_rebuilds':old.gram_rebuilds,'legacy_shift_copy_bytes':old.copy_shift_bytes,
            'cached_gram_column_updates':new.gram_column_updates,'cached_shift_copy_bytes':new.copy_shift_bytes})
    return result


def trajectory(base,truth,B,steps,abrupt=False):
    rng=np.random.default_rng(700+B)
    phases=rng.choice(base.shape[1],B,replace=False)
    amplitude=rng.uniform(.7,1.3,B)
    for t in range(steps):
        ids=(phases+t)%base.shape[1]
        rhs=np.asfortranarray(base[:,ids]*amplitude)
        expected=truth[ids]*amplitude
        yield rhs,expected


def run_pipeline(f,peak,base,truth,B,steps,name,capacity,rtol):
    if name.startswith('serial'):
        solvers=[tb.TemporalRecovery(f,peak,'direct' if name=='serial_direct' else 'extrapolate',
                    0 if name=='serial_direct' else capacity,rtol,projection_metric='residual') for _ in range(B)]
    else:
        solvers=BatchRecovery(f,peak,B,capacity,rtol,history='legacy' if 'legacy' in name else 'cached',
                              strategy='dense_zero' if 'dense' in name else 'compact',direct=name=='batch_direct')
    times=[];errors=[];residuals=[];stages={s:0. for s in ['predict_project_check','pack','factor','scatter','history','peak']}
    failed=active=solved_columns=factor_calls=0
    for rhs,expected in trajectory(base,truth,B,steps):
        if isinstance(solvers,list):
            u=np.zeros_like(rhs);observed=[];elapsed=0.;stepfailed=stepactive=0
            for env,s in enumerate(solvers):
                u[:,env],row=s.query(rhs[:,env]);observed.append(row['peak_Pa'])
                elapsed+=row['elapsed_ms']/1000;stepfailed+=row['factor_used'];stepactive+=row['full_relative_residual'] is not None
                stages['factor']+=row['factor_ms']/1000;stages['history']+=row['history_update_ms']/1000
                stages['predict_project_check']+=row['predict_check_ms']/1000
            times.append(elapsed);failed+=stepfailed;active+=stepactive;solved_columns+=stepfailed;factor_calls+=stepfailed
            observed=np.asarray(observed)
        else:
            u,observed,row=solvers.step(rhs);times.append(row['seconds'])
            failed+=row['failed'];active+=row['active'];solved_columns+=row['solved_columns'];factor_calls+=row['factor_calls']
            for stage,seconds in row['stage_seconds'].items():stages[stage]+=seconds
        # Independent full K residual includes the six scalar gauge rows.
        norms=np.linalg.norm(rhs,axis=0);relative=np.linalg.norm(f.k@u-rhs,axis=0)/np.maximum(norms,1e-12)
        nonzero=norms>1e-12
        residuals.extend(relative[nonzero].tolist())
        assert np.all(relative[nonzero] <= max(5*rtol,1e-8)),(name,relative.max())
        errors.extend((abs(observed-expected)/np.maximum(expected,1.)).tolist())
    total=float(sum(times));sample_count=B*steps
    return {'method':name,'B':B,'steps':steps,'capacity':capacity,'rtol':rtol,
        'total_seconds':total,'mean_batch_ms':total/steps*1000,'env_recoveries_per_second':sample_count/total,
        'mean_env_ms':total/sample_count*1000,'failed_env_steps':failed,'nonzero_env_steps':active,
        'failed_fraction_nonzero':failed/max(active,1),'actual_solved_RHS_columns':solved_columns,
        'factor_API_calls':factor_calls,'max_peak_error_percent_1Pa_floor':float(max(errors)*100),
        'p95_peak_error_percent_1Pa_floor':float(np.percentile(errors,95)*100),
        'max_full_relative_residual_nonzero':float(max(residuals,default=0.)),
        'stage_mean_ms_per_env':{k:v/sample_count*1000 for k,v in stages.items()},
        'timing_scope':'Already assembled RHS -> recovery, history update and all-corner peak. Reference/validation excluded. Serial sums the original query timers; batch includes gather/scatter and Python orchestration.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');parser.add_argument('--repeats',type=int,default=2);args=parser.parse_args()
    out=Path(__file__).resolve().parent
    with threadpool_limits(limits=1):
        xyz,tet,outer,meta=tb.test.egg.egg_mesh(2,2)
        f=tb.test.conv.FEM(xyz,tet,outer,2,quarter=False,label='batch-compaction-history-validation')
        geometry=tb.test.SurfaceGeometry(f,10);mapper=tb.IndexedContacts(geometry)
        peak=P2Peak(f.glambda,f.elements,f.young/(2*(1+f.poisson)))
        peak.warmup(np.zeros(f.ndof));gravity=f.m@np.tile([0.,0.,-9.81],len(f.xyz))
        cases=tb.smooth_loads.prepare_cases(f,geometry,512)
        data={}
        for case_name in ['smooth_stick_slide_release','abrupt_control']:
            RHS=[];U=[];peaks=[]
            for descriptor in cases[case_name]:
                traction,_,_=tb.smooth_loads.assemble_descriptor(geometry,descriptor)
                ids=np.flatnonzero(np.any(traction!=0.,axis=1))
                b=f.balance(mapper.load_tractions(traction[ids],ids)+gravity)
                u=np.zeros(f.ndof)
                if np.linalg.norm(b)>1e-12:u[f.free]=f.factor.solve(b[f.free])
                assert np.linalg.norm(f.k@u-b)<1e-8*max(np.linalg.norm(b),1.)
                RHS.append(b);U.append(u);peaks.append(peak(u)[0])
            data[case_name]=(np.asfortranarray(np.column_stack(RHS)),np.asfortranarray(np.column_stack(U)),np.asarray(peaks))
        result={'CPU':tb.test.egg.cpu_name(),'platform':platform.platform(),'threads':1,'GPU_tested':False,
            'mesh':{'DOFs':f.ndof,'tets':f.ne,'stress_converged':False},'factor_nnz':f.factor.nnz,
            'scope':'Fixed full-body P2 shell, synthetic friction-admissible prescribed loads. Environments have independent histories, different phases and amplitudes of one moving-contact grasp. Not Genesis/contact dynamics/RL/GPU benchmarking.',
            'accuracy':'Same FP64 factor and same equation tolerance for pipeline ablations. Actual full K residual and global peak against direct reference checked on every env-step. No certified physical peak error bound.',
            'microbenchmarks':{},'pipelines':[]}
        b,u,truth=data['smooth_stick_slide_release']
        result['microbenchmarks']=microbenchmarks(f,peak,b,u,args.quick)
        (out/'batch_history_results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'microbenchmarks_complete':True}),flush=True)
        configurations=[(1,128 if args.quick else 512),(8,64 if args.quick else 512),(32,32 if args.quick else 128)]
        methods=['serial_direct','batch_direct','serial_legacy','batch_legacy_dense','batch_legacy_compact','batch_cached_compact']
        for repetition in range(1 if args.quick else args.repeats):
            for B,steps in configurations:
                order=methods if repetition%2==0 else list(reversed(methods))
                for method in order:
                    if B==1 and method=='batch_legacy_dense':continue
                    record=run_pipeline(f,peak,b,truth,B,steps,method,8,1e-3)
                    record['case']='smooth_friction';record['repetition']=repetition
                    result['pipelines'].append(record)
                    (out/'batch_history_results.json').write_text(json.dumps(result,indent=2))
                    print(json.dumps(record),flush=True)
        # Strict-tolerance and abrupt controls: optimization must not rely on
        # every frame being accepted, or fall back for the entire batch.
        for name,steps,tol in [('abrupt_control',32,1e-3),('smooth_stick_slide_release',64,1e-6)]:
            bb,uu,tt=data[name]
            for method in ['batch_direct','batch_legacy_dense','batch_legacy_compact','batch_cached_compact']:
                record=run_pipeline(f,peak,bb,tt,8,steps,method,8,tol)
                record['case']='abrupt_friction' if name=='abrupt_control' else 'strict_residual'
                result['pipelines'].append(record)
                (out/'batch_history_results.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(record),flush=True)
        print('CPU ablations complete.',flush=True)


if __name__=='__main__':main()
