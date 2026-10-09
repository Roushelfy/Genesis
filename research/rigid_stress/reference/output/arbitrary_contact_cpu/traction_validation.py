#!/usr/bin/env python3
"""Validate arbitrary vector surface tractions without a contact-profile basis."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import arbitrary_contact_test as test
from indexed_contacts import IndexedContacts
from general_peak import P2Peak, smoke_test

HERE = Path(__file__).resolve().parent


def dense_reference(g, traction):
    weighted = g.integration_weights[:, :, None] * traction.reshape(g.coords.shape)
    local = np.einsum('fqc,qa->fac', weighted, g.shape)
    raw = np.zeros((len(g.f.xyz), 3))
    for a in range(6):
        for axis in range(3):
            np.add.at(raw[:, axis], g.nodes[:, a], local[:, a, axis])
    return raw.ravel()


def main():
    rng = np.random.default_rng(314159)
    with threadpool_limits(limits=1):
        xyz, tet, outer, meta = test.egg.egg_mesh(2, 2)
        f = test.conv.FEM(xyz, tet, outer, order=2, quarter=False,
                          label='arbitrary-vector-traction-validation')
        g = test.SurfaceGeometry(f, 10)
        index = IndexedContacts(g)
        mu = f.young / (2 * (1 + f.poisson))
        kernel = P2Peak(f.glambda, f.elements, mu, 'onfly_serial')
        kernel.warmup(np.zeros(f.ndof))
        # Every component varies with position; neither normal traction nor
        # Gaussian/constant-force patches are assumed by this interface.
        x = index.coords / .03
        dense = 2e4 * np.column_stack((np.sin(3*x[:, 1]) + .4*x[:, 2],
                    x[:, 0]*x[:, 2] + .2*np.cos(5*x[:, 1]),
                    np.cos(4*x[:, 0]) * np.sin(2*x[:, 2])))
        ids = rng.choice(len(x), 257, replace=False)
        sparse = rng.normal(0., 1e5, (len(ids), 3))
        duplicates = np.r_[ids[:50], ids[:50]]
        cases = [('dense_position_varying_vector_traction', dense, None),
                 ('257_random_samples_vector_traction', sparse, ids),
                 ('duplicate_samples_additive', rng.normal(0., 1e5, (100, 3)), duplicates),
                 ('zero_vector_traction', np.zeros_like(dense), None)]
        records = []
        for name, traction, subset in cases:
            sample_ids = np.arange(len(x)) if subset is None else subset
            full = np.zeros_like(dense)
            np.add.at(full, sample_ids, traction)
            expected = dense_reference(g, full)
            actual = index.load_tractions(traction, subset)
            mapping_error = np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1.)
            nodal = actual.reshape(-1, 3)
            forces = index.weights[sample_ids, None]*traction
            expected_force = forces.sum(axis=0)
            expected_torque = np.cross(index.coords[sample_ids], forces).sum(axis=0)
            force_error = np.linalg.norm(nodal.sum(axis=0)-expected_force)
            torque_error = np.linalg.norm(np.cross(f.xyz, nodal).sum(axis=0)-expected_torque)
            assert mapping_error < 1e-12
            assert force_error < 1e-11*max(np.linalg.norm(expected_force), 1.)
            assert torque_error < 1e-11*max(np.linalg.norm(expected_torque), 1.)
            raw = actual-test.centrifugal_inertia(f, np.array([2., -3., 5.])) if np.any(actual) else actual
            u, rhs, _ = f.solve(raw)
            original = test.peak(f, u)
            check = test.validate(f, u, rhs, raw, original)
            peak_error = abs(kernel(u)[0]-original)/max(original, 1.)
            assert peak_error < 1e-8
            records.append({'case': name, 'samples': len(sample_ids),
                'relative_mapping_error_1N_floor': float(mapping_error),
                'force_error_N': float(force_error), 'torque_error_Nm': float(torque_error),
                'fused_peak_error_1Pa_floor': float(peak_error), **check})
        result = {'scope': 'Arbitrary vector traction at any stored outer-surface quadrature samples; no contact profile or symmetry basis. This is discrete integration validation, not continuum convergence.',
                  'mesh': {'DOFs': f.ndof, 'tets': f.ne, 'stress_converged': False},
                  'GPU_tested': False, 'all_passed': True, 'records': records}
        (HERE/'arbitrary_traction_validation.json').write_text(json.dumps(result, indent=2))
        synthetic = smoke_test()
        (HERE/'general_peak_synthetic_validation.json').write_text(json.dumps(synthetic, indent=2))
        print(json.dumps({'tractions': result, 'synthetic_peak': synthetic}, indent=2))


if __name__ == '__main__':
    main()
