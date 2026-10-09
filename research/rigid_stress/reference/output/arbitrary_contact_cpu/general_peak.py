"""General P2 straight-tetrahedron maximum von Mises postprocessing.

All four corners of every element are evaluated for every displacement field.
No contact rank, prescribed contact location, symmetry, loading direction,
preselected stress hotspot, or displacement-solver assumption is used.

For affine tetrahedral geometry the P2 strain and linear elastic stress are
affine in barycentric position. The von Mises norm is convex, so its exact
unaveraged finite-element maximum is attained at one of the four corners.
This statement does not cover curved isoparametric elements, nonlinear
materials, finite-strain stress, averaged nodal stress, or dynamic response.

Connectivity order: vertices0,1,2,3 then edges01,02,03,12,13,23.
P2Peak(...)(u) returns (peak_Pa, element_index, corner_index).
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from threadpoolctl import threadpool_limits

EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], np.int64)
MID = np.array([[0, 4, 5, 6], [4, 1, 7, 8],
                [5, 7, 2, 9], [6, 8, 9, 3]], np.int64)
MODES = ('onfly_serial', 'onfly_parallel', 'cached_serial', 'cached_parallel')


def corner_gradients(glambda):
    """Cached full P2 shape gradients: (ne,4,10,3), 960 FP64 bytes/tet."""
    g = np.asarray(glambda, np.float64)
    bary = np.eye(4)
    vertex = (4 * bary[None, :, :, None] - 1) * g[:, None, :, :]
    edge = 4 * (bary[None, :, EDGES[:, 0], None] * g[:, None, EDGES[:, 1], :]
                + bary[None, :, EDGES[:, 1], None] * g[:, None, EDGES[:, 0], :])
    return np.ascontiguousarray(np.concatenate((vertex, edge), axis=2))


def constitutive(mu, lam=0.0):
    """Isotropic engineering-shear constitutive matrix, Voigt xx yy zz xy yz xz."""
    D = np.diag([2 * mu] * 3 + [mu] * 3)
    D[:3, :3] += lam
    return D


def baseline_numpy(glambda, elements, u, mu, lam=0.0, chunk=1024):
    """Full strain/stress baseline matching conv.FEM.strain + ref.von_mises."""
    G = np.asarray(glambda, np.float64)
    E = np.asarray(elements, np.int64)
    U = np.asarray(u, np.float64).reshape(-1, 3)
    D = constitutive(float(mu), float(lam))
    best, best_e, best_c = -1.0, 0, 0
    for lo in range(0, len(E), chunk):
        hi = min(lo + chunk, len(E))
        grads = corner_gradients(G[lo:hi])
        du = np.einsum('eia,eqib->eqab', U[E[lo:hi]], grads, optimize=True)
        eps = np.stack((du[:, :, 0, 0], du[:, :, 1, 1], du[:, :, 2, 2],
                        du[:, :, 0, 1] + du[:, :, 1, 0],
                        du[:, :, 1, 2] + du[:, :, 2, 1],
                        du[:, :, 0, 2] + du[:, :, 2, 0]), axis=-1)
        stress = eps @ D.T
        squared = (.5 * ((stress[..., 0] - stress[..., 1]) ** 2
                          + (stress[..., 1] - stress[..., 2]) ** 2
                          + (stress[..., 2] - stress[..., 0]) ** 2)
                   + 3 * np.sum(stress[..., 3:] ** 2, axis=-1))
        loc = int(np.argmax(squared))
        value = float(squared.ravel()[loc])
        if value > best:
            e, c = np.unravel_index(loc, squared.shape)
            best, best_e, best_c = value, lo + int(e), int(c)
    return float(np.sqrt(best)), best_e, best_c


@njit(cache=True, inline='always')
def _element_onfly(G, E, U, e):
    # -sum(vertex displacement tensor barycentric gradient), reused at all
    # corners. A vertex at corner c changes coefficient -1 -> +3; incident
    # midpoint nodes contribute 4 times the opposite vertex gradient.
    # Shape-gradient partition of unity makes subtraction of one constant
    # nodal displacement exactly strain-invariant in real arithmetic. This
    # suppresses cancellation from large rigid translations without assuming
    # that u was computed in any particular gauge or coordinate frame.
    n0 = E[e, 0]
    tx, ty, tz = U[n0, 0], U[n0, 1], U[n0, 2]
    b0 = b1 = b2 = b3 = b4 = b5 = 0.0
    for j in range(4):
        n = E[e, j]
        ux, uy, uz = U[n, 0] - tx, U[n, 1] - ty, U[n, 2] - tz
        gx, gy, gz = G[e, j, 0], G[e, j, 1], G[e, j, 2]
        b0 -= ux * gx
        b1 -= uy * gy
        b2 -= uz * gz
        b3 -= ux * gy + uy * gx
        b4 -= uy * gz + uz * gy
        b5 -= ux * gz + uz * gx
    best, best_c = -1.0, 0
    for c in range(4):
        n = E[e, c]
        ux, uy, uz = U[n, 0] - tx, U[n, 1] - ty, U[n, 2] - tz
        gx, gy, gz = G[e, c, 0], G[e, c, 1], G[e, c, 2]
        e0, e1, e2 = b0 + 4 * ux * gx, b1 + 4 * uy * gy, b2 + 4 * uz * gz
        e3 = b3 + 4 * (ux * gy + uy * gx)
        e4 = b4 + 4 * (uy * gz + uz * gy)
        e5 = b5 + 4 * (ux * gz + uz * gx)
        for j in range(4):
            if j == c:
                continue
            n = E[e, MID[c, j]]
            ux, uy, uz = U[n, 0] - tx, U[n, 1] - ty, U[n, 2] - tz
            gx, gy, gz = G[e, j, 0], G[e, j, 1], G[e, j, 2]
            e0 += 4 * ux * gx
            e1 += 4 * uy * gy
            e2 += 4 * uz * gz
            e3 += 4 * (ux * gy + uy * gx)
            e4 += 4 * (uy * gz + uz * gy)
            e5 += 4 * (ux * gz + uz * gx)
        value = (2 * ((e0 - e1) ** 2 + (e1 - e2) ** 2 + (e2 - e0) ** 2)
                 + 3 * (e3 * e3 + e4 * e4 + e5 * e5))
        if value > best:
            best, best_c = value, c
    return best, best_c


@njit(cache=True, inline='always')
def _element_cached(GC, E, U, e):
    n0 = E[e, 0]
    tx, ty, tz = U[n0, 0], U[n0, 1], U[n0, 2]
    best, best_c = -1.0, 0
    for c in range(4):
        e0 = e1 = e2 = e3 = e4 = e5 = 0.0
        for j in range(10):
            n = E[e, j]
            ux, uy, uz = U[n, 0] - tx, U[n, 1] - ty, U[n, 2] - tz
            gx, gy, gz = GC[e, c, j, 0], GC[e, c, j, 1], GC[e, c, j, 2]
            e0 += ux * gx
            e1 += uy * gy
            e2 += uz * gz
            e3 += ux * gy + uy * gx
            e4 += uy * gz + uz * gy
            e5 += ux * gz + uz * gx
        value = (2 * ((e0 - e1) ** 2 + (e1 - e2) ** 2 + (e2 - e0) ** 2)
                 + 3 * (e3 * e3 + e4 * e4 + e5 * e5))
        if value > best:
            best, best_c = value, c
    return best, best_c


@njit(cache=True)
def _scan_onfly(G, E, U):
    best, best_e, best_c = -1.0, 0, 0
    for e in range(len(E)):
        value, c = _element_onfly(G, E, U, e)
        if value > best:
            best, best_e, best_c = value, e, c
    return np.sqrt(best), best_e, best_c


@njit(cache=True)
def _scan_cached(GC, E, U):
    best, best_e, best_c = -1.0, 0, 0
    for e in range(len(E)):
        value, c = _element_cached(GC, E, U, e)
        if value > best:
            best, best_e, best_c = value, e, c
    return np.sqrt(best), best_e, best_c


@njit(cache=True, parallel=True)
def _parallel_onfly(G, E, U, block_size):
    blocks = (len(E) + block_size - 1) // block_size
    values = np.empty(blocks)
    indices = np.empty(blocks, np.int64)
    corners = np.empty(blocks, np.int64)
    for block in prange(blocks):
        best, best_e, best_c = -1.0, block * block_size, 0
        for e in range(block * block_size, min((block + 1) * block_size, len(E))):
            value, c = _element_onfly(G, E, U, e)
            if value > best:
                best, best_e, best_c = value, e, c
        values[block], indices[block], corners[block] = best, best_e, best_c
    block = np.argmax(values)
    return np.sqrt(values[block]), indices[block], corners[block]


@njit(cache=True, parallel=True)
def _parallel_cached(GC, E, U, block_size):
    blocks = (len(E) + block_size - 1) // block_size
    values = np.empty(blocks)
    indices = np.empty(blocks, np.int64)
    corners = np.empty(blocks, np.int64)
    for block in prange(blocks):
        best, best_e, best_c = -1.0, block * block_size, 0
        for e in range(block * block_size, min((block + 1) * block_size, len(E))):
            value, c = _element_cached(GC, E, U, e)
            if value > best:
                best, best_e, best_c = value, e, c
        values[block], indices[block], corners[block] = best, best_e, best_c
    block = np.argmax(values)
    return np.sqrt(values[block]), indices[block], corners[block]


class P2Peak:
    def __init__(self, glambda, elements, mu, mode='onfly_serial', block_size=256):
        start = perf_counter()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        G = np.ascontiguousarray(glambda, np.float64)
        E = np.ascontiguousarray(elements, np.int64)
        if G.shape != (len(E), 4, 3) or E.shape != (len(E), 10) or not len(E):
            raise ValueError('glambda(ne,4,3), elements(ne,10), ne>0 required')
        if np.min(E) < 0 or block_size < 1 or not np.isfinite(mu) or mu <= 0:
            raise ValueError('nonnegative node indices, positive block_size and finite mu>0 required')
        self.G, self.E, self.mu = G, E, float(mu)
        self.max_node = int(E.max())
        self.mode, self.block_size = mode, int(block_size)
        self.cached = mode.startswith('cached')
        self.parallel = mode.endswith('parallel')
        self.GC = corner_gradients(G) if self.cached else None
        self.metadata = {
            'mode': mode, 'element_count': len(E), 'evaluated_corners_per_query': 4 * len(E),
            'gradient_bytes': int(G.nbytes), 'connectivity_bytes': int(E.nbytes),
            'cached_gradient_bytes': int(self.GC.nbytes) if self.cached else 0,
            'block_maximum_workspace_bytes': 24 * ((len(E) + self.block_size - 1) // self.block_size) if self.parallel else 0,
            'block_size': self.block_size, 'construction_s': perf_counter() - start,
            'fastmath': False, 'arithmetic_dtype': 'float64',
            'per_element_translation_centering': True,
            'returns': ['von_Mises_peak_Pa', 'element', 'corner'],
            'scope': 'All elements and all four P2 corners, arbitrary displacement field, isotropic linear elasticity, affine tetrahedral geometry',
            'constitutive_identity': 'sigma_vm = mu*sqrt(2*sum(diagonal_strain_differences**2)+3*sum(engineering_shear**2)); lambda cancels',
        }

    def __call__(self, u):
        U = np.asarray(u, np.float64)
        if U.ndim == 1 and U.size % 3 == 0:
            U = U.reshape(-1, 3)
        if U.ndim != 2 or U.shape[1] != 3 or len(U) <= self.max_node:
            raise ValueError('u must have shape (node_count,3) or flattened 3*node_count')
        U = np.ascontiguousarray(U)
        if self.cached:
            r = (_parallel_cached(self.GC, self.E, U, self.block_size) if self.parallel
                 else _scan_cached(self.GC, self.E, U))
        else:
            r = (_parallel_onfly(self.G, self.E, U, self.block_size) if self.parallel
                 else _scan_onfly(self.G, self.E, U))
        return self.mu * float(r[0]), int(r[1]), int(r[2])

    def warmup(self, u):
        start = perf_counter()
        result = self(u)
        self.metadata['first_query_including_JIT_s'] = perf_counter() - start
        return result


def benchmark_general(glambda, elements, mu, us, lam=0.0, repeats=3,
                      threads=(1, 2, 4, 8), block_size=256):
    """Root-run real-model benchmark; no assumptions about how us were solved.

    Input `us` is an iterable of (node_count,3) displacement arrays. NumPy
    baseline includes complete gradient/strain/stress construction. Every
    optimized query is checked against it; argmax coordinates may differ in
    tied maxima, so numerical peak value is the correctness criterion.
    Construction/JIT costs are separately reported. QPS uses total query time,
    including actual zeros/random loads supplied by the caller, never 1/P50.
    """
    U = [np.ascontiguousarray(u, np.float64).reshape(-1, 3) for u in us]
    if not U or repeats < 1:
        raise ValueError('nonempty displacement list and repeats>=1 required')
    old_threads = get_num_threads()
    out = {'GPU_tested': False, 'result_type': 'general_P2_maximum_postprocessing',
           'fields': len(U), 'repeats': repeats, 'records': [],
           'scope': 'Postprocessing only; load construction and linear solves excluded. All elements/corners evaluated; no contact basis or symmetry.'}
    def timing(values):
        return {'mean_ms': float(np.mean(values) * 1000),
                'p50_ms': float(np.median(values) * 1000),
                'p95_ms': float(np.percentile(values, 95) * 1000),
                'queries_per_second': len(values) / sum(values)}
    try:
        with threadpool_limits(limits=1):
            true = [baseline_numpy(glambda, elements, u, mu, lam) for u in U]
            times = []
            for _ in range(repeats):
                for u in U:
                    start = perf_counter()
                    baseline_numpy(glambda, elements, u, mu, lam)
                    times.append(perf_counter() - start)
            out['records'].append({'mode': 'numpy_full_strain_stress', 'timing': timing(times)})
            for mode in MODES:
                k = P2Peak(glambda, elements, mu, mode, block_size)
                k.warmup(U[0])
                for nt in (threads if k.parallel else (1,)):
                    set_num_threads(nt)
                    result = [k(u) for u in U]
                    difference = np.abs(np.array([r[0] for r in result]) - np.array([r[0] for r in true]))
                    relative = difference / np.maximum([r[0] for r in true], 1.0)
                    times = []
                    for _ in range(repeats):
                        for u in U:
                            start = perf_counter()
                            k(u)
                            times.append(perf_counter() - start)
                    out['records'].append({'mode': mode, 'threads': nt,
                        'timing': timing(times), 'metadata': dict(k.metadata),
                        'max_absolute_error_Pa': float(difference.max()),
                        'max_relative_error_1Pa_floor': float(relative.max()),
                        'all_hotspot_indices_match': all(r[1:] == t[1:] for r, t in zip(result, true)),
                        'passed_1e-8_scaled_error': bool(relative.max() < 1e-8)})
                del k
    finally:
        set_num_threads(old_threads)
    return out


def smoke_test():
    """Small synthetic-only tests plus direct conv.FEM.strain crosscheck."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import eggshell_convergence_cpu as conv
    import rigid_stress_reference as ref
    rng = np.random.default_rng(91827)
    ne = 73
    vertices = rng.normal(size=(ne, 4, 3))
    # Reject poorly conditioned synthetic tetrahedra.
    for e in range(ne):
        while np.linalg.cond(np.c_[np.ones(4), vertices[e]]) > 30:
            vertices[e] = rng.normal(size=(4, 3))
    xyz = np.concatenate((vertices, vertices[:, EDGES].mean(axis=2)), axis=1).reshape(-1, 3)
    E = np.arange(10 * ne, dtype=np.int64).reshape(ne, 10)
    G = np.linalg.inv(np.concatenate((np.ones((ne, 4, 1)), vertices), axis=2))[:, 1:, :].transpose(0, 2, 1)
    mu, lam = 3.7e9, 5.1e9
    arbitrary = rng.normal(size=xyz.shape) * 1e-4
    bend = np.c_[xyz[:, 0] * xyz[:, 2], np.zeros(len(xyz)), -.5 * xyz[:, 0] ** 2] * 1e-4
    affine = xyz @ np.array([[1e-4, 2e-4, -1e-4], [3e-4, 1e-4, 0], [0, 2e-4, -2e-4]]).T
    hydro = .001 * xyz
    rigid = np.cross([.002, -.003, .001], xyz) + np.array([.2, -.1, .3])
    fields = {'arbitrary': arbitrary, 'quadratic_bending': bend, 'affine': affine,
              'hydrostatic': hydro, 'rigid_translation_rotation': rigid, 'zero': np.zeros_like(xyz)}
    fem = object.__new__(conv.FEM)
    fem.ne, fem.glambda, fem.order, fem.nloc = ne, G, 2, 10
    fem.dofs = (3 * E[:, :, None] + np.arange(3)).reshape(ne, 30)
    references = {}
    records = []
    for name, u in fields.items():
        full = []
        for _, _, strain in fem.strain(u.ravel(), np.eye(4)):
            full.append(ref.von_mises(strain @ constitutive(mu, lam).T))
        vm = np.concatenate(full)
        arg = int(np.argmax(vm))
        reference = (float(vm.ravel()[arg]), arg // 4, arg % 4)
        baseline = baseline_numpy(G, E, u, mu, lam)
        assert abs(baseline[0] - reference[0]) / max(reference[0], 1.) < 1e-12
        references[name] = reference
    old_threads = get_num_threads()
    try:
        for mode in MODES:
            kernel = P2Peak(G, E, mu, mode, block_size=11)
            for nt in ((1, 2, 4, 8) if kernel.parallel else (1,)):
                set_num_threads(nt)
                errors = []
                active_errors, null_differences, null_peaks = [], [], []
                for name, u in fields.items():
                    value = kernel(u)
                    error = abs(value[0] - references[name][0]) / max(references[name][0], 1.)
                    null_case = name in ('hydrostatic', 'rigid_translation_rotation', 'zero')
                    # The full stress baseline has FP64 cancellation noise
                    # for large rigid translations at mu=3.7 GPa. Relative
                    # error against that nominally-zero quantity is invalid.
                    tolerance = 1e-4 if null_case else 1e-12
                    assert error < tolerance, (mode, nt, name, value, references[name], error)
                    if name in ('arbitrary', 'quadratic_bending'):
                        assert value[1:] == references[name][1:]
                    if name in ('hydrostatic', 'rigid_translation_rotation', 'zero'):
                        assert value[0] < 1e-5
                        null_differences.append(abs(value[0] - references[name][0]))
                        null_peaks.append(value[0])
                    else:
                        active_errors.append(error)
                    errors.append(error)
                records.append({'mode': mode, 'threads': nt,
                                'max_error_1Pa_floor': max(errors),
                                'max_nonzero_relative_error': max(active_errors),
                                'max_null_case_absolute_difference_Pa': max(null_differences),
                                'max_null_case_computed_peak_Pa': max(null_peaks),
                                'status': 'passed'})
    finally:
        set_num_threads(old_threads)
    return {'element_count': ne, 'synthetic_displacement_fields': list(fields),
            'reference': 'conv.FEM.strain + ref.von_mises', 'records': records}


if __name__ == '__main__':
    import json
    print(json.dumps(smoke_test(), indent=2))
