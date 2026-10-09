#!/usr/bin/env python3
"""Full-body arbitrary-load benchmarks: actual factor solve + all-corner peak."""
import json,argparse,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import numba
from threadpoolctl import threadpool_limits
import arbitrary_contact_test as test
from indexed_contacts import IndexedContacts
import general_peak as gp

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--level',type=int,default=2);p.add_argument('--frames',type=int,default=16);p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
    with threadpool_limits(limits=1):
        xyz,tet,outer,meta=test.egg.egg_mesh(args.level,2)
        f=test.conv.FEM(xyz,tet,outer,order=2,quarter=False,label='fullbody-arbitrary-contact-benchmark')
        geometry=test.SurfaceGeometry(f,10);index=IndexedContacts(geometry)
        cases=test.workloads(f,args.frames);U=[];references=[]
        data={'scope':'Fixed full-body geometry, arbitrary current finite contact patches; known material inputs; coarse/unconverged P2 model',
            'GPU_tested':False,'mesh':{**meta,'DOFs':f.ndof,'nodes':len(f.xyz),'tets':f.ne,'corner_samples':4*f.ne,'stress_converged':False},
            'spatial_index':index.metadata,'cases':{},'postprocessing':None}
        for name,sequence in cases.items():
            fields=[];right=[];peaks=[];oldtimes=[];newtimes=[];errs=[]
            for descriptor in sequence:
                start=perf_counter();a,_=test.load_shared(geometry,descriptor['patches']);oldtimes.append(perf_counter()-start)
                start=perf_counter();b=index.load(descriptor['patches']);newtimes.append(perf_counter()-start)
                diff=np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30) if np.any(a) else np.linalg.norm(b)
                assert diff<1e-11;errs.append(float(diff))
                raw=b-test.centrifugal_inertia(f,np.asarray(descriptor['omega_rad_s']))
                u,rhs,_=f.solve(raw);peak=test.peak(f,u)
                test.validate(f,u,rhs,raw,peak)
                fields.append(u);right.append(rhs);peaks.append(peak)
            U.extend(fields);references.extend(peaks)
            data['cases'][name]={'mapping_shared':test.stats(oldtimes),'mapping_indexed':test.stats(newtimes),
                'mapping_max_relative_load_error':max(errs),'frames':len(sequence)}
            cases[name]={'descriptors':sequence,'u':fields,'rhs':right,'peaks':peaks}
        # Full strain/stress baseline is independent of the direct vm identity.
        mu=f.young/(2*(1+f.poisson));lam=f.young*f.poisson/((1+f.poisson)*(1-2*f.poisson))
        representative=U[::4]+U[-4:]
        data['postprocessing']=gp.benchmark_general(f.glambda,f.elements,mu,representative,lam=lam,repeats=5)
        assert all(r.get('passed_1e-8_scaled_error',True) for r in data['postprocessing']['records'])
        # Compare optimized peaks against the original conv.FEM path as well.
        peak_kernel=gp.P2Peak(f.glambda,f.elements,mu,'onfly_serial');peak_kernel.warmup(U[0])
        values=np.asarray([peak_kernel(u)[0] for u in U]);ref=np.asarray(references)
        data['full_original_peak_error_1Pa_floor']=float(np.max(abs(values-ref)/np.maximum(ref,1.)))
        assert data['full_original_peak_error_1Pa_floor']<1e-8
        numba.set_num_threads(1)
        for name,case in cases.items():
            times={k:[] for k in ['shared_map_solve_numpy_peak','indexed_map_solve_fused_peak','incremental_fused_peak','identical_rhs_cache_peak']}
            solve_times=[];hits=0
            for _ in range(args.repeats):
                last_rhs=np.zeros(f.ndof);last_u=np.zeros(f.ndof);cache_rhs=None;cache_peak=None
                for descriptor,rhs,expected in zip(case['descriptors'],case['rhs'],case['peaks']):
                    start=perf_counter();contact,_=test.load_shared(geometry,descriptor['patches'])
                    raw=contact-test.centrifugal_inertia(f,np.asarray(descriptor['omega_rad_s']));current=f.balance(raw)
                    u=np.zeros(f.ndof);u[f.free]=f.factor.solve(current[f.free]);v=test.peak(f,u)
                    times['shared_map_solve_numpy_peak'].append(perf_counter()-start)
                    assert abs(v-expected)/max(expected,1.)<1e-8
                    start=perf_counter();contact=index.load(descriptor['patches'])
                    raw=contact-test.centrifugal_inertia(f,np.asarray(descriptor['omega_rad_s']));current=f.balance(raw)
                    u=np.zeros(f.ndof);t=perf_counter();u[f.free]=f.factor.solve(current[f.free]);solve_times.append(perf_counter()-t)
                    v=peak_kernel(u)[0];times['indexed_map_solve_fused_peak'].append(perf_counter()-start)
                    assert abs(v-expected)/max(expected,1.)<1e-8
                    start=perf_counter()
                    if not np.any(rhs):incremental=np.zeros(f.ndof)
                    else:
                        du=np.zeros(f.ndof);du[f.free]=f.factor.solve((rhs-last_rhs)[f.free]);incremental=last_u+du
                    v=peak_kernel(incremental)[0];times['incremental_fused_peak'].append(perf_counter()-start)
                    assert abs(v-expected)/max(expected,1.)<1e-8
                    last_rhs=rhs;last_u=incremental
                    start=perf_counter()
                    if cache_rhs is not None and np.array_equal(rhs,cache_rhs):v=cache_peak;hits+=1
                    else:
                        u=np.zeros(f.ndof);u[f.free]=f.factor.solve(rhs[f.free]);v=peak_kernel(u)[0]
                        cache_rhs=rhs.copy();cache_peak=v
                    times['identical_rhs_cache_peak'].append(perf_counter()-start)
                    assert abs(v-expected)/max(expected,1.)<1e-8
            data['cases'][name]['pipeline']={k:test.stats(ts) for k,ts in times.items()}
            data['cases'][name]['factor_solve_only']=test.stats(solve_times)
            data['cases'][name]['identical_rhs_hits']=hits
            data['cases'][name]['scope_notes']={'first_two':'Contact mapping, centrifugal inertia, inertia relief, solve and maximum included',
                'last_two':'Already-assembled RHS supplied; excludes contact mapping and inertia relief; not comparable as full pipeline'}
        (HERE/f'arbitrary_contact_benchmark_L{args.level}.json').write_text(json.dumps(data,indent=2))
        print(json.dumps({'mesh':data['mesh'],'peak_error':data['full_original_peak_error_1Pa_floor'],
            'results':{k:v['pipeline'] for k,v in data['cases'].items()}}),flush=True)


if __name__=='__main__':main()
