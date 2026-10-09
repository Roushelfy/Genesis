#!/usr/bin/env python3
"""Compare temporal step density, force-residual projection and friction effect."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import temporal_benchmark as tb

HERE=Path(__file__).resolve().parent


def main():
    with threadpool_limits(limits=1):
        xyz,tet,outer,meta=tb.test.egg.egg_mesh(2,2)
        f=tb.test.conv.FEM(xyz,tet,outer,2,quarter=False,label='smooth-friction-sampling')
        g=tb.test.SurfaceGeometry(f,10);mapper=tb.IndexedContacts(g)
        kernel=tb.P2Peak(f.glambda,f.elements,f.young/(2*(1+f.poisson)),'onfly_serial')
        kernel.warmup(np.zeros(f.ndof))
        gravity=f.m@np.tile([0.,0.,-9.81],len(f.xyz))
        result={'scope':'Same synthetic smooth stick-slide-release path sampled with 128/512 points. Full-body model and local cone from smooth_loads. Reference, mapping and physical-error checks excluded from solve timing.',
                'GPU_tested':False,'material_fixed':True,'mesh_converged':False,
                'projection':'energy uses Q^T residual; residual uses (KQ)^T residual and minimizes full Euclidean force residual. This is an online changing subspace, not a restricted load basis.',
                'records':[], 'friction_effect':None}
        for frames in [128,512]:
            sequence=tb.smooth_loads.prepare_cases(f,g,frames)['smooth_stick_slide_release']
            right=[];truth=[];normal_peaks=[];checks=[]
            normals=tb.smooth_loads._outward_normals(g)
            for descriptor in sequence:
                traction,check,_=tb.smooth_loads.assemble_descriptor(g,descriptor)
                ids=np.flatnonzero(np.any(traction!=0,axis=1))
                contact=mapper.load_tractions(traction[ids],ids);rhs=f.balance(contact+gravity)
                u=np.zeros(f.ndof)
                if np.linalg.norm(rhs)>1e-12:u[f.free]=f.factor.solve(rhs[f.free])
                right.append(rhs);truth.append(kernel(u)[0]);checks.append(check)
                if frames==128:
                    field=traction.reshape(g.coords.shape)
                    p=-np.sum(field*normals,axis=2)
                    normal=(-p[:,:,None]*normals).reshape(-1,3)
                    normal_contact=mapper.load_tractions(normal[ids],ids)
                    normal_rhs=f.balance(normal_contact+gravity);u=np.zeros(f.ndof)
                    if np.linalg.norm(normal_rhs)>1e-12:u[f.free]=f.factor.solve(normal_rhs[f.free])
                    normal_peaks.append(kernel(u)[0])
            truth=np.asarray(truth)
            if frames==128:
                normal_peaks=np.asarray(normal_peaks)
                active=truth>1.
                differences=(normal_peaks-truth)/np.maximum(truth,1.)
                result['friction_effect']={'scope':'Identical normal pressure/inertia relief with versus without prescribed tangential traction. Not a comparison of two contact dynamics trajectories.',
                    'max_absolute_relative_peak_change_nonzero':float(np.max(np.abs(differences[active]))),
                    'minimum_normal_only_over_full_peak_nonzero':float(np.min(normal_peaks[active]/truth[active])),
                    'maximum_normal_only_over_full_peak_nonzero':float(np.max(normal_peaks[active]/truth[active])),
                    'full_peak_Pa':truth.tolist(),'normal_only_peak_Pa':normal_peaks.tolist(),
                    'phases':[x['phase'] for x in sequence],
                    'max_relative_tangency':max(x['tangency_relative_to_peak_pressure'] for x in checks),
                    'max_relative_cone_violation':max(x['cone_violation_relative_to_peak_pressure'] for x in checks)}
            configurations=[('direct',0,'energy',1e-8),('extrapolate',0,'energy',1e-6),
                ('extrapolate',8,'energy',1e-6),('extrapolate',8,'residual',1e-8),
                ('extrapolate',8,'residual',1e-6),('extrapolate',16,'residual',1e-6)]
            for mode,capacity,metric,tol in configurations:
                times=[];all_errors=[];rows=None
                for _ in range(2):
                    solver=tb.TemporalRecovery(f,kernel,mode,capacity,tol,projection_metric=metric)
                    current=[]
                    for frame,rhs in enumerate(right):
                        _,row=solver.query(rhs)
                        peakerror=abs(row['peak_Pa']-truth[frame])/max(truth[frame],1.)
                        row.update({'frame':frame,'peak_error_1Pa_floor':float(peakerror)})
                        times.append(row['elapsed_ms']/1000);all_errors.append(peakerror);current.append(row)
                    if rows is None:rows=current
                record={'frames':frames,'mode':mode,'history_capacity':capacity,'projection_metric':metric,'rtol':tol,
                    'timing':tb.test.stats(times),'factor_calls_per_run':sum(x['factor_used'] for x in rows),
                    'max_peak_error_1Pa_floor':float(max(all_errors)),
                    'max_full_relative_residual_nonzero':max(x['full_relative_residual'] or 0. for x in rows),
                    'empirical_peak_error_below_0_01_percent':bool(max(all_errors)<1e-4),'rows':rows,
                    **tb.nonzero_factor_counts(rows)}
                result['records'].append(record)
                print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
            (HERE/'sampling_and_friction_results.json').write_text(json.dumps(result,indent=2))
        print('Sampling and friction study complete.',flush=True)


if __name__=='__main__':main()
