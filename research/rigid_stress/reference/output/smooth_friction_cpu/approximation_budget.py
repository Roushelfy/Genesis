#!/usr/bin/env python3
"""Explicitly looser solver tolerances: measure actual full-domain peak errors.

This experiment changes equation precision; it is not an equal-precision
acceleration result or a rigorous output-error bound.
"""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import temporal_benchmark as tb

HERE=Path(__file__).resolve().parent


def main():
    with threadpool_limits(limits=1):
        xyz,tet,outer,_=tb.test.egg.egg_mesh(2,2)
        f=tb.test.conv.FEM(xyz,tet,outer,2,quarter=False,label='smooth-friction-approximation-budget')
        g=tb.test.SurfaceGeometry(f,10);mapper=tb.IndexedContacts(g)
        peak=tb.P2Peak(f.glambda,f.elements,f.young/(2*(1+f.poisson)),'onfly_serial')
        peak.warmup(np.zeros(f.ndof));gravity=f.m@np.tile([0.,0.,-9.81],len(f.xyz))
        workloads=tb.smooth_loads.prepare_cases(f,g,512)
        result={'scope':'Explicitly relaxed equation tolerance; empirical peak accuracy only on synthetic full-body frictional snapshots. No certified physical/error guarantee or GPU test.',
                'timing_scope':'Assembled RHS through recovery + global peak; includes history maintenance, excludes reference, mapping and validation.',
                'cases':{}}
        for name in ['smooth_stick_slide_release','abrupt_control']:
            B=[];truth=[]
            for descriptor in workloads[name]:
                traction,_,_=tb.smooth_loads.assemble_descriptor(g,descriptor)
                ids=np.flatnonzero(np.any(traction!=0,axis=1))
                b=f.balance(mapper.load_tractions(traction[ids],ids)+gravity)
                u=np.zeros(f.ndof)
                if np.linalg.norm(b)>1e-12:u[f.free]=f.factor.solve(b[f.free])
                B.append(b);truth.append(peak(u)[0])
            truth=np.asarray(truth);records=[]
            configs=[('direct',0,1e-8),('extrapolate',4,1e-4),('extrapolate',4,1e-3),
                     ('extrapolate',8,1e-4),('extrapolate',8,1e-3)]
            for mode,capacity,tol in configs:
                times=[];errors=[];first=None
                for _ in range(2):
                    solver=tb.TemporalRecovery(f,peak,mode,capacity,tol,projection_metric='residual')
                    rows=[]
                    for i,b in enumerate(B):
                        _,row=solver.query(b)
                        error=abs(row['peak_Pa']-truth[i])/max(truth[i],1.)
                        row.update({'frame':i,'reference_peak_Pa':float(truth[i]),'peak_error_1Pa_floor':float(error)})
                        rows.append(row);times.append(row['elapsed_ms']/1000);errors.append(error)
                    if first is None:first=rows
                record={'mode':mode,'history_capacity':capacity,'rtol':tol,'timing':tb.test.stats(times),
                    'factor_calls_per_run':sum(x['factor_used'] for x in first),
                    'max_peak_error_1Pa_floor':float(max(errors)),
                    'max_full_relative_residual_nonzero':max(x['full_relative_residual'] or 0. for x in first),
                    'rows':first,**tb.nonzero_factor_counts(first)}
                records.append(record);print(json.dumps({'case':name,**{k:v for k,v in record.items() if k!='rows'}}),flush=True)
            result['cases'][name]={'frames':len(B),'records':records}
            (HERE/'approximation_budget_results.json').write_text(json.dumps(result,indent=2))
        print('Explicit approximation-budget study complete.',flush=True)


if __name__=='__main__':main()
