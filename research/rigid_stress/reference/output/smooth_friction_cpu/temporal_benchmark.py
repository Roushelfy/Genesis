#!/usr/bin/env python3
"""Online temporal predictor/corrector for full-body frictional snapshots.

Checks the current COMPLETE RHS, including residual at the six gauge rows.
The numerical residual is not a certified bound on physical peak stress.
Peak accuracy is compared to a full factor solve on every tested snapshot.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'arbitrary_contact_cpu'))
import arbitrary_contact_test as test
from indexed_contacts import IndexedContacts
from general_peak import P2Peak
import smooth_loads


def nonzero_factor_counts(rows):
    active = [r for r in rows if r['full_relative_residual'] is not None]
    skipped = sum(not r['factor_used'] for r in active)
    return {'nonzero_RHS_frames':len(active), 'nonzero_factor_skipped':skipped,
            'nonzero_factor_skip_fraction':skipped/len(active) if active else 0.}


class TemporalRecovery:
    def __init__(self, f, kernel, mode, capacity, rtol, atol=1e-12, projection_metric='energy'):
        self.f, self.kernel, self.mode = f, kernel, mode
        self.capacity, self.rtol, self.atol = capacity, rtol, atol
        assert projection_metric in ('energy', 'residual')
        self.projection_metric = projection_metric
        self.K = f.k.tocsr()
        self.last = self.before = None
        self.Q = np.empty((f.ndof, 0))
        self.KQ = np.empty_like(self.Q)

    def append_direction(self, delta):
        if not self.capacity:
            return
        if self.Q.shape[1] >= self.capacity:
            self.Q, self.KQ = self.Q[:, 1:].copy(), self.KQ[:, 1:].copy()
        q = delta.copy()
        aq = self.K@q
        original_energy = float(q@aq if self.projection_metric=='energy' else aq@aq)
        if not np.isfinite(original_energy) or original_energy <= 0:
            return
        for _ in range(2):
            if self.Q.shape[1]:
                left = self.Q if self.projection_metric=='energy' else self.KQ
                gram = left.T@self.KQ
                coefficient = linalg.solve(gram, left.T@aq, assume_a='sym')
                q -= self.Q@coefficient
                aq -= self.KQ@coefficient
        energy = float(q@aq if self.projection_metric=='energy' else aq@aq)
        if energy <= 1e-12*original_energy or not np.isfinite(energy):
            return
        scale = np.sqrt(energy)
        self.Q = np.column_stack((self.Q, q/scale))
        self.KQ = np.column_stack((self.KQ, aq/scale))

    def query(self, rhs):
        start = perf_counter()
        norm_rhs = float(np.linalg.norm(rhs))
        tolerance = max(self.atol, self.rtol*norm_rhs)
        stats = {'factor_used': False, 'basis_size_before': self.Q.shape[1],
                 'factor_ms': 0., 'predict_check_ms': 0., 'history_update_ms': 0.}
        if norm_rhs <= self.atol:
            u = np.zeros(self.f.ndof)
        elif self.mode == 'direct':
            u = np.zeros(self.f.ndof)
            t = perf_counter()
            u[self.f.free] = self.f.factor.solve(rhs[self.f.free])
            stats['factor_ms'] = 1e3*(perf_counter()-t)
            stats['factor_used'] = True
        else:
            t = perf_counter()
            if self.last is None:
                u = np.zeros(self.f.ndof)
            elif self.mode == 'hold' or self.before is None:
                u = self.last.copy()
            else:
                # The test samples are uniformly spaced in normalized time.
                u = 2*self.last-self.before
            residual = rhs-self.K@u
            if self.capacity and self.Q.shape[1]:
                left = self.Q if self.projection_metric=='energy' else self.KQ
                gram = left.T@self.KQ
                coefficient = linalg.solve(gram, left.T@residual, assume_a='sym')
                u += self.Q@coefficient
                # Do not rely only on cached KQ for the acceptance test.
                residual = rhs-self.K@u
            stats['predicted_residual_1N_floor'] = float(np.linalg.norm(residual)/max(norm_rhs, 1.))
            stats['predict_check_ms'] = 1e3*(perf_counter()-t)
            if np.linalg.norm(residual) > tolerance:
                correction = np.zeros(self.f.ndof)
                t = perf_counter()
                correction[self.f.free] = self.f.factor.solve(residual[self.f.free])
                stats['factor_ms'] = 1e3*(perf_counter()-t)
                stats['factor_used'] = True
                u += correction
                t = perf_counter()
                self.append_direction(correction)
                stats['history_update_ms'] = 1e3*(perf_counter()-t)
        peak, element, corner = self.kernel(u)
        elapsed = perf_counter()-start
        # Accuracy validation excluded from timing, applied to every result.
        error = rhs-self.K@u
        residual_abs = float(np.linalg.norm(error))
        stats.update({'elapsed_ms': elapsed*1000, 'peak_Pa': peak,
                      'hot_element': element, 'hot_corner': corner,
                      'full_residual_N': residual_abs,
                      'full_relative_residual': residual_abs/norm_rhs if norm_rhs>self.atol else None,
                      'acceptance_tolerance_N': tolerance})
        # FP64 factor roundoff on this thin shell sets a residual floor.
        assert residual_abs <= max(5*tolerance, 1e-8*norm_rhs, 1e-11), stats
        self.before, self.last = self.last, u
        return u, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        xyz, tetra, outer, meta = test.egg.egg_mesh(2, 2)
        f = test.conv.FEM(xyz, tetra, outer, order=2, quarter=False,
                          label='fullbody-smooth-friction')
        g = test.SurfaceGeometry(f, 10)
        mapper = IndexedContacts(g)
        kernel = P2Peak(f.glambda, f.elements, f.young/(2*(1+f.poisson)), 'onfly_serial')
        kernel.warmup(np.zeros(f.ndof))
        K = f.k.tocsr()
        gravity = f.m@np.tile([0., 0., -9.81], len(f.xyz))
        descriptors = smooth_loads.prepare_cases(f, g, frames=args.frames)
        output = {'scope':'Synthetic friction-cone-admissible changing-input snapshots, full P2 shell, known fixed material. No Genesis/contact dynamics, trajectory consistency, mesh convergence or GPU validation.',
                  'CPU':test.egg.cpu_name(), 'threads':1, 'GPU_tested':False,
                  'mesh':{**meta, 'DOFs':f.ndof, 'tets':f.ne, 'quarter':False, 'stress_converged':False},
                  'timing_scope':'Already assembled complete RHS through displacement recovery and full-domain peak scan. Mapping and reference/accuracy checks excluded; history updates included.',
                  'residual_caveat':'Residual thresholds are numerical solver tolerances, not rigorous peak-stress error certificates. Every tested output compared to full FP64 factor reference.',
                  'cases':{}}
        configurations = [('direct', 0, 1e-8)]
        configurations += [(mode, cap, tol) for tol in [1e-8, 1e-6]
                           for mode, cap in [('hold',0), ('extrapolate',0), ('extrapolate',4), ('extrapolate',8), ('extrapolate',16)]]
        for case_name, sequence in descriptors.items():
            right, reference_peaks, load_checks, map_seconds = [], [], [], []
            for descriptor in sequence:
                t = perf_counter()
                traction, check, phase = smooth_loads.assemble_descriptor(g, descriptor)
                check['phase'] = phase
                active = np.flatnonzero(np.any(traction != 0., axis=1))
                contact = mapper.load_tractions(traction[active], active)
                rhs = f.balance(contact+gravity)
                map_seconds.append(perf_counter()-t)
                u = np.zeros(f.ndof)
                if np.linalg.norm(rhs)>1e-12:
                    u[f.free] = f.factor.solve(rhs[f.free])
                relative = np.linalg.norm(K@u-rhs)/max(np.linalg.norm(rhs), 1.)
                assert relative < 1e-7
                reference_peaks.append(kernel(u)[0])
                nodal = contact.reshape(-1,3)
                force_error = np.linalg.norm(nodal.sum(axis=0)-np.asarray(check['integrated_force_N']))
                torque_error = np.linalg.norm(np.cross(f.xyz,nodal).sum(axis=0)-np.asarray(check['integrated_torque_Nm']))
                assert force_error<1e-10 and torque_error<1e-10
                check.update({'nodal_force_error_N':float(force_error),'nodal_torque_error_Nm':float(torque_error)})
                right.append(rhs);load_checks.append(check)
            reference_peaks = np.asarray(reference_peaks)
            records = []
            for mode, capacity, rtol in configurations:
                elapsed, peak_errors, residuals = [], [], []
                summary = None
                for repetition in range(args.repeats):
                    solver = TemporalRecovery(f, kernel, mode, capacity, rtol)
                    rows = []
                    for i, rhs in enumerate(right):
                        _, record = solver.query(rhs)
                        peak_error = abs(record['peak_Pa']-reference_peaks[i])/max(reference_peaks[i], 1.)
                        record.update({'frame':i,'reference_peak_Pa':float(reference_peaks[i]),
                                       'peak_error_1Pa_floor':float(peak_error)})
                        elapsed.append(record['elapsed_ms']/1000)
                        peak_errors.append(peak_error)
                        residuals.append(record['full_relative_residual'] or 0.)
                        rows.append(record)
                    if repetition == 0:
                        summary = rows
                record = {'mode':mode,'history_capacity':capacity,'rtol':rtol,
                          'timing':test.stats(elapsed),
                          'factor_calls_per_run':sum(r['factor_used'] for r in summary),
                          'factor_skip_fraction':float(np.mean([not r['factor_used'] for r in summary])),
                          'max_peak_error_1Pa_floor':float(max(peak_errors)),
                          'max_full_relative_residual_nonzero':float(max(residuals)),
                          'empirical_peak_error_below_0_01_percent':bool(max(peak_errors)<1e-4),
                          'rows':summary, **nonzero_factor_counts(summary)}
                records.append(record)
                print(json.dumps({'case':case_name,**{k:v for k,v in record.items() if k!='rows'}}),flush=True)
            output['cases'][case_name] = {'frames':len(sequence), 'mapping_integration_checks_timing':test.stats(map_seconds),
                                          'load_checks':load_checks,'records':records,'descriptors':sequence}
            (HERE/'smooth_friction_results.json').write_text(json.dumps(output,indent=2))
        print('Smooth friction benchmark completed.',flush=True)


if __name__ == '__main__':
    main()
