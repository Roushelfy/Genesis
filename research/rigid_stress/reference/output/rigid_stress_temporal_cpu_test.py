#!/usr/bin/env python3
"""CPU FEM test of temporal reuse for maximum von Mises stress.

Requires NumPy, SciPy, threadpoolctl, and rigid_stress_reference.py next to this
file. Optional --plot requires matplotlib. Run:
    python rigid_stress_temporal_cpu_test.py --output-dir results --plot

Three online paths are timed separately (one environment / one CPU BLAS thread):
  cached_lu: assemble balanced RHS, cached sparse solve, all-tet stress/max;
  response_scan: exact response on a fixed 24-dimensional load space, all tets;
  temporal: previous group hotspots + safe temporal bounds + active group scan.

The load space comprises six finite surface patches x three force directions,
and six rigid-reference rotational inertia coefficients. This is an exact
response for THIS fixed traction model, not a universal approximation for
arbitrary moving contact. There is no Genesis or GPU integration.

Each temporal region holds a reference q, exact region maximum, and reference
argmax. Current reference hotspots establish a lower bound. With A_e mapping q
to a stress vector whose norm is von Mises, the temporal upper bound is
    U_g = m_g(ref) + min(L_g*||dq||, sum_k L_gk*|dq_k|).
L_g=max_e ||A_e||_2; L_gk=max_e ||A_e[:,k]||_2. Conservative float64 margins
are included; this is numerical validation, not interval arithmetic proof.
Regions with upper bound below the current lower bound are skipped. Large
active sets fall back to a vectorized scan. Repeated identical q returns cached
results exactly. No oracle information is fed into the temporal algorithm.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rigid_stress_reference", HERE / "rigid_stress_reference.py")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)


def vm_vector(stress):
    """Six-component, rank-five representation with norm == von Mises."""
    s = np.asarray(stress)
    y = np.empty_like(s)
    c = 1.0 / np.sqrt(2.0)
    y[..., 0] = c * (s[..., 0] - s[..., 1])
    y[..., 1] = c * (s[..., 1] - s[..., 2])
    y[..., 2] = c * (s[..., 2] - s[..., 0])
    y[..., 3:] = np.sqrt(3.0) * s[..., 3:]
    return y


def morton_order(centroids):
    """Offline spatial clustering; normalized axes avoid beam aspect-ratio bias."""
    span = np.ptp(centroids, axis=0)
    x = np.floor((centroids - centroids.min(axis=0)) / np.maximum(span, 1e-12) * 1023).astype(np.uint64)
    code = np.zeros(len(x), dtype=np.uint64)
    for bit in range(10):
        for axis in range(3):
            code |= ((x[:, axis] >> bit) & 1) << (3 * bit + axis)
    return np.argsort(code, kind="stable")


def finite_patch_basis(model, scale=100.0):
    faces = model.surface_faces
    points = model.xyz[faces]
    lo, hi = model.xyz.min(axis=0), model.xyz.max(axis=0)
    normalized = (points - lo) / (hi - lo)
    centroids = normalized.mean(axis=1)
    areas = .5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    # Opposite x patches plus offset y / z patches create different hotspots.
    specs = [(0, 0., [.50, .50]), (0, 1., [.50, .50]),
             (1, 0., [.30, .50]), (1, 1., [.72, .50]),
             (2, 0., [.40, .50]), (2, 1., [.65, .50])]
    result = np.zeros((model.ndof, 18))
    metadata = []
    for p, (axis, side, center) in enumerate(specs):
        surface = np.all(np.abs(normalized[:, :, axis] - side) < 1e-10, axis=1)
        other = [i for i in range(3) if i != axis]
        distance = np.linalg.norm(centroids[:, other] - np.asarray(center), axis=1)
        selected = np.flatnonzero(surface & (distance <= .18))
        if not len(selected):
            candidates = np.flatnonzero(surface)
            selected = candidates[np.argsort(distance[candidates])[:1]]
        nodal = np.zeros(len(model.xyz))
        for a in range(3):
            np.add.at(nodal, faces[selected, a], areas[selected] / 3.0)
        nodal /= nodal.sum()
        for d in range(3):
            result[d::3, 3 * p + d] = scale * nodal
        metadata.append({"patch": p, "axis": axis, "side": side, "triangles": len(selected),
                         "area": float(areas[selected].sum()), "unit_q_force_N": scale})
    return result, metadata


class ResponseModel:
    def __init__(self, subdivisions, group_size=64, sides=(3., .6, .4)):
        self.group_size = group_size
        t = perf_counter()
        xyz, tets = ref.structured_cube(subdivisions)
        xyz *= np.asarray(sides)
        self.fem = ref.CachedElasticRecovery(xyz, tets, young=1e6, poisson=.3, density=2.)
        fem = self.fem
        self.offline_model_seconds = perf_counter() - t
        t = perf_counter()
        loads, self.patch_metadata = finite_patch_basis(fem)
        self.contact_basis = loads
        self.spin_scale = 5.0
        # c(w)=w(w.r)-r(w.w). Six symmetric quadratic coefficients.
        x, y, z = fem.offsets.T
        spin = np.zeros((len(x), 3, 6))
        spin[:, :, 0] = np.column_stack([0*x, -y, -z])
        spin[:, :, 1] = np.column_stack([-x, 0*y, -z])
        spin[:, :, 2] = np.column_stack([-x, -y, 0*z])
        spin[:, :, 3] = np.column_stack([y, x, 0*z])
        spin[:, :, 4] = np.column_stack([z, 0*y, x])
        spin[:, :, 5] = np.column_stack([0*x, z, y])
        spin = spin.reshape(fem.ndof, 6) * self.spin_scale**2
        h = np.column_stack([loads, -(fem.m @ spin)])
        self.rhs_basis = h - fem.mr @ linalg.cho_solve(fem.g_factor, fem.r.T @ h)
        u = np.zeros_like(self.rhs_basis)
        u[fem.free] = fem.factor.solve(np.asfortranarray(self.rhs_basis[fem.free]))
        stress = np.einsum('eij,ejr->eir', fem.stress_operator, u[fem.element_dofs], optimize=False)
        # Component transform along axis 1, before sorting / grouping.
        A = np.empty_like(stress)
        A[:, 0] = (stress[:, 0] - stress[:, 1]) / np.sqrt(2)
        A[:, 1] = (stress[:, 1] - stress[:, 2]) / np.sqrt(2)
        A[:, 2] = (stress[:, 2] - stress[:, 0]) / np.sqrt(2)
        A[:, 3:] = np.sqrt(3) * stress[:, 3:]
        centers = fem.xyz[fem.tetrahedra].mean(axis=1)
        self.order = morton_order(centers)
        self.A = np.ascontiguousarray(A[self.order])
        self.flat = self.A.reshape(-1, 24)
        self.ne = len(self.A)
        self.ng = (self.ne + group_size - 1) // group_size
        self.starts = np.arange(self.ng) * group_size
        self.counts = np.minimum(group_size, self.ne - self.starts)
        spectral = np.empty(self.ne)
        for start in range(0, self.ne, 2048):
            spectral[start:start+2048] = np.linalg.svd(self.A[start:start+2048], compute_uv=False)[:, 0]
        col = np.linalg.norm(self.A, axis=1)
        self.L = np.maximum.reduceat(spectral, self.starts) * (1 + 1e-12)
        self.Lcol = np.maximum.reduceat(col, self.starts, axis=0) * (1 + 1e-12)
        self.offline_response_seconds = perf_counter() - t
        self._u = np.zeros(fem.ndof)
        self._bfree = np.ascontiguousarray(self.rhs_basis[fem.free])
        self.metadata = {"subdivisions": subdivisions, "sides": list(sides),
                         "nodes": len(xyz), "tets": len(tets), "scalar_dofs": fem.ndof,
                         "group_size": group_size, "groups": self.ng, "load_basis_dimension": 24,
                         "LU_nnz_L": fem.factor.L.nnz, "LU_nnz_U": fem.factor.U.nnz,
                         "response_bytes_FP64": self.A.nbytes, "factorization_count": 1,
                         "offline_model_assembly_and_factor_seconds": self.offline_model_seconds,
                         "offline_response_and_bounds_seconds": self.offline_response_seconds,
                         "patches": self.patch_metadata}

    def values(self, q, indices=None):
        if indices is None:
            a = self.flat
        else:
            a = self.A[indices].reshape(-1, 24)
        y = (a @ q).reshape(-1, 6)
        return np.einsum('ij,ij->i', y, y)

    def response_scan(self, q):
        v2 = self.values(q)
        i = int(np.argmax(v2))
        return float(np.sqrt(v2[i])), i

    def cached_lu(self, q):
        # Fast full-recovery reference: no diagnostic K residual in timed path.
        f = self.fem
        self._u[f.free] = f.factor.solve(self._bfree @ q)
        stress = np.einsum('eij,ej->ei', f.stress_operator, self._u[f.element_dofs], optimize=False)
        y = vm_vector(stress)
        v2 = np.einsum('ij,ij->i', y, y)
        e = int(np.argmax(v2))
        # Convert original element id to sorted id only for consistent interface.
        return float(np.sqrt(v2[e])), e

    def check_physical_mapping(self, q, omega):
        gravity = self.fem.consistent_body_force([0., 0., -9.81])
        raw = self.contact_basis @ q[:18] + gravity
        full = self.fem.solve(raw, omega)
        m = float(ref.von_mises(full.stress).max())
        b = self.rhs_basis @ q
        return {"maximum_relative_error": abs(self.response_scan(q)[0] - m)/max(m, 1.),
                "rhs_relative_error": float(np.linalg.norm(b-full.compatible_load)/max(np.linalg.norm(b), 1.)),
                "full_residual": full.residual_relative}


@dataclass
class TrackResult:
    maximum: float
    sorted_argmax: int
    evaluated: int
    active_groups: int
    full_scan: bool
    reused_identical: bool


class TemporalMax:
    def __init__(self, model, fallback_fraction=.45):
        self.model = model
        self.fallback_fraction = fallback_fraction
        self.qref = None
        self.last_q = None
        self.last_m = None
        self.last_i = None

    def _full(self, q):
        m = self.model
        v2 = m.values(q)
        padded = np.zeros(m.ng * m.group_size)
        padded[:m.ne] = v2
        blocks = padded.reshape(m.ng, m.group_size)
        arg = blocks.argmax(axis=1)
        self.hot = m.starts + arg
        self.group_refmax = np.sqrt(blocks[np.arange(m.ng), arg])
        self.qref = np.broadcast_to(q, (m.ng, 24)).copy()
        i = int(v2.argmax())
        value = float(np.sqrt(v2[i]))
        return TrackResult(value, i, m.ne, m.ng, True, False)

    def query(self, q):
        q = np.asarray(q)
        m = self.model
        if self.last_q is not None and np.array_equal(q, self.last_q):
            return TrackResult(self.last_m, self.last_i, 0, 0, False, True)
        if self.qref is None:
            out = self._full(q)
        elif not np.any(q):
            # Exact zero effective load in this response model.
            self.qref.fill(0.)
            self.group_refmax.fill(0.)
            self.hot = m.starts.copy()
            out = TrackResult(0., 0, 0, 0, False, False)
        else:
            seed2 = m.values(q, self.hot)
            seed_group = int(seed2.argmax())
            lower = float(np.sqrt(seed2[seed_group]))
            winner = int(self.hot[seed_group])
            dq = q[None, :] - self.qref
            change = np.minimum(m.L * np.linalg.norm(dq, axis=1),
                                np.sum(m.Lcol * np.abs(dq), axis=1))
            upper = self.group_refmax + change
            margin = 1e-10 + 1e-12*np.maximum(upper, lower)
            active = np.flatnonzero(upper + margin > lower)
            if len(active) >= self.fallback_fraction*m.ng:
                out = self._full(q)
                out.evaluated += m.ng  # Seed work is not hidden.
            elif not len(active):
                out = TrackResult(lower, winner, m.ng, 0, False, False)
            else:
                indices = (m.starts[active, None] + np.arange(m.group_size)[None, :]).ravel()
                valid = indices < m.ne
                indices = indices[valid]
                vals2 = m.values(q, indices)
                local = int(vals2.argmax())
                best = float(np.sqrt(vals2[local]))
                if best > lower:
                    lower, winner = best, int(indices[local])
                # Refresh only active groups; skipped anchors remain unchanged.
                padded = np.zeros(len(active)*m.group_size)
                padded[valid] = vals2
                blocks = padded.reshape(len(active), m.group_size)
                arg = blocks.argmax(axis=1)
                self.hot[active] = m.starts[active] + arg
                self.group_refmax[active] = np.sqrt(blocks[np.arange(len(active)), arg])
                self.qref[active] = q
                out = TrackResult(lower, winner, m.ng+len(indices), len(active), False, False)
        self.last_q = q.copy()
        self.last_m, self.last_i = out.maximum, out.sorted_argmax
        return out


def append_spin(contact_q, omega, spin_scale=5.0):
    w = np.asarray(omega) / spin_scale
    return np.r_[contact_q, w*w, w[0]*w[1], w[0]*w[2], w[1]*w[2]]


def make_sequences(frames, seed=20261008):
    t = np.linspace(0., 1., frames)
    all_cases = {}
    q0 = np.zeros(18); q0[0] = .8; q0[3] = -.8
    all_cases['repeat_hold'] = (np.tile(append_spin(q0, [0., 0., .4]), (frames, 1)),
                               np.tile([0., 0., .4], (frames, 1)))
    q = np.zeros((frames, 18))
    q[:, 0] = .8 + .12*np.sin(2*np.pi*t)
    q[:, 3] = -q[:, 0]
    q[:, 1] = .05*np.sin(2*np.pi*t)
    q[:, 4] = -.04*np.sin(2*np.pi*t+.2)
    q[:, 2] = .03*np.cos(2*np.pi*t)
    w = np.column_stack([.15*np.sin(t*2*np.pi), .1*np.cos(t*np.pi), .5+.1*np.sin(t*np.pi)])
    all_cases['smooth_grasp'] = (np.stack([append_spin(a,b) for a,b in zip(q,w)]),w)
    q = np.zeros((frames, 18))
    q[:, 0] = .25+.9*t; q[:, 3] = -q[:, 0]
    q[:, 7] = .15*t*t; q[:, 10] = -.1*t
    q[:, 2] = .08*np.sin(t*np.pi)
    w = np.column_stack([.8*t, .3*t*t, .2+1.2*t])
    all_cases['slow_drift'] = (np.stack([append_spin(a,b) for a,b in zip(q,w)]),w)
    amplitude = np.maximum(0., 1.-t)
    amplitude[frames//2:] *= .25  # Abrupt unloading, then exactly zero.
    q = np.zeros((frames, 18));q[:,0]=amplitude;q[:,3]=-amplitude
    w = np.zeros((frames,3))
    all_cases['unloading'] = (np.stack([append_spin(a,b) for a,b in zip(q,w)]),w)
    q = np.zeros((frames,18)); w=np.zeros((frames,3))
    for i in range(frames):
        stage=min(2,3*i//frames)
        p0,p1,d=[(0,1,0),(2,3,1),(4,5,2)][stage]
        a=.7+.15*np.sin(i*.04)
        q[i,3*p0+d]=a;q[i,3*p1+d]=-a
        q[i,3*p0+(d+1)%3]=.04*np.cos(i*.03)
        w[i]=[.1*np.sin(i*.02),0.,.3]
    all_cases['contact_switch'] = (np.stack([append_spin(a,b) for a,b in zip(q,w)]),w)
    rng=np.random.default_rng(seed)
    q=rng.normal(0.,.4,size=(frames,18))
    w=rng.normal(0.,1.,size=(frames,3))
    all_cases['random_reload']=(np.stack([append_spin(a,b) for a,b in zip(q,w)]),w)
    return all_cases


def stats(values):
    v=np.asarray(values)
    return {"mean_ms":float(v.mean()*1e3),"p50_ms":float(np.median(v)*1e3),
            "p95_ms":float(np.quantile(v,.95)*1e3)}


def benchmark_case(model, name, qs, omegas, repeats, frame_rows):
    # Independent direct solves establish the full-FEM ground truth for EVERY frame.
    oracle=np.asarray([model.cached_lu(q)[0] for q in qs])
    scans=[model.response_scan(q) for q in qs]
    scan_values=np.array([a[0] for a in scans])
    direct_error=float(np.max(np.abs(scan_values-oracle)/np.maximum(oracle,1.)))
    assert direct_error<1e-9,(name,'response_vs_direct',direct_error)
    physical=[]
    for i in sorted(set([0,len(qs)//3,len(qs)//2,2*len(qs)//3,len(qs)-1])):
        physical.append(model.check_physical_mapping(qs[i],omegas[i]))
    assert max(x['rhs_relative_error'] for x in physical)<1e-10
    assert max(x['maximum_relative_error'] for x in physical)<1e-9
    times={k:[] for k in ['cached_lu','response_scan','temporal']}
    outputs=[]
    # Warm the relevant code paths before measuring trajectories.
    for q in qs[:3]:model.cached_lu(q);model.response_scan(q)
    for repeat in range(repeats):
        for method in ['cached_lu','response_scan','temporal']:
            tracker=TemporalMax(model) if method=='temporal' else None
            for i,q in enumerate(qs):
                start=perf_counter()
                if method=='temporal':
                    o=tracker.query(q);value=o.maximum
                else:
                    value=getattr(model,method)(q)[0]
                elapsed=perf_counter()-start
                times[method].append(elapsed)
                assert abs(value-oracle[i])<=1e-8+1e-9*max(oracle[i],1.),(method,name,i,value,oracle[i])
                if method=='temporal' and repeat==0:
                    outputs.append(o)
                    # Naive previous true argmax is deliberately stronger than a
                    # predictor with an already-wrong history; it is NOT fed to tracker.
                    previous=scans[max(0,i-1)][1]
                    naive=float(np.sqrt(model.values(q,np.asarray([previous]))[0]))
                    frame_rows.append({"subdivisions":model.metadata['subdivisions'],"nodes":model.metadata['nodes'],
                        "tets":model.ne,"scenario":name,"frame":i,
                        "oracle_max_Pa":oracle[i],"temporal_max_Pa":o.maximum,
                        "absolute_error_Pa":abs(o.maximum-oracle[i]),
                        "response_evaluations":o.evaluated,"evaluation_fraction":o.evaluated/model.ne,
                        "active_groups":o.active_groups,"full_scan":int(o.full_scan),
                        "identical_load_reuse":int(o.reused_identical),
                        "temporal_first_repeat_ms":elapsed*1e3,
                        "naive_previous_argmax_Pa":naive,
                        "naive_underestimate_fraction":max(0.,oracle[i]-naive)/max(oracle[i],1.)})
    out={"scenario":name,"frames":len(qs),"repeats":repeats,
         "response_vs_direct_max_relative_error":direct_error,
         "temporal_vs_direct_max_relative_error":max(abs(o.maximum-a)/max(a,1.) for o,a in zip(outputs,oracle)),
         "physical_mapping_checks":physical,
         "timings":{k:stats(v) for k,v in times.items()},
         "temporal_mean_response_evaluation_fraction":float(np.mean([o.evaluated/model.ne for o in outputs])),
         "full_scan_frames_including_cold_start":sum(o.full_scan for o in outputs),
         "identical_load_reuse_frames":sum(o.reused_identical for o in outputs),
         "maximum_tet_id_changes":int(np.sum(np.diff([a[1] for a in scans])!=0))}
    out['temporal_speedup_vs_response_scan_p50']=out['timings']['response_scan']['p50_ms']/out['timings']['temporal']['p50_ms']
    out['temporal_speedup_vs_cached_lu_p50']=out['timings']['cached_lu']['p50_ms']/out['timings']['temporal']['p50_ms']
    current_rows=[r for r in frame_rows if r['subdivisions']==model.metadata['subdivisions'] and r['scenario']==name]
    out['naive_previous_argmax_worst_underestimate_fraction']=max(r['naive_underestimate_fraction'] for r in current_rows)
    return out


def cg_warm_start_test(frames=32):
    """Small FEM check only; do not confuse Krylov reuse with direct-solve reuse."""
    model=ResponseModel(4,group_size=32)
    fem=model.fem
    K=fem.k[fem.free][:,fem.free].tocsr()
    diag=K.diagonal()
    P=LinearOperator(K.shape,matvec=lambda x:x/diag,dtype=np.float64)
    seq=make_sequences(frames)['smooth_grasp'][0]
    previous=None
    results=[]
    for i,q in enumerate(seq):
        b=(model.rhs_basis@q)[fem.free]
        direct=fem.factor.solve(b)
        row={"frame":i}
        for mode in ['cold','warm']:
            counter=[0]
            def callback(x):counter[0]+=1
            start=perf_counter()
            x,info=cg(K,b,x0=previous if mode=='warm' else None,rtol=1e-8,atol=0.,maxiter=4000,M=P,callback=callback)
            elapsed=perf_counter()-start
            assert info==0,(mode,i,info)
            u=np.zeros(fem.ndof);u[fem.free]=x
            stress=np.einsum('eij,ej->ei',fem.stress_operator,u[fem.element_dofs],optimize=False)
            maximum=float(ref.von_mises(stress).max())
            true=model.response_scan(q)[0]
            row[mode+'_iterations']=counter[0]
            row[mode+'_ms']=elapsed*1e3
            row[mode+'_maximum_stress_relative_error']=abs(maximum-true)/max(true,1.)
            row[mode+'_residual_relative']=float(np.linalg.norm(K@x-b)/max(np.linalg.norm(b),1.))
            if mode=='warm':previous=x.copy()
        results.append(row)
    # An exact previous solution plus an exact delta solve gives the same result,
    # but both direct solves still traverse the cached factors.
    b0=(model.rhs_basis@seq[0])[fem.free];b1=(model.rhs_basis@seq[1])[fem.free]
    x0=fem.factor.solve(b0);dx=fem.factor.solve(b1-b0)
    increment_error=float(np.linalg.norm(x0+dx-fem.factor.solve(b1))/np.linalg.norm(fem.factor.solve(b1)))
    return {"model":model.metadata,"preconditioner":"Jacobi","rtol_relative_to_current_rhs":1e-8,
            "frames":frames,"cold_mean_iterations":float(np.mean([r['cold_iterations'] for r in results])),
            "warm_mean_iterations":float(np.mean([r['warm_iterations'] for r in results])),
            "cold_p50_ms":float(np.median([r['cold_ms'] for r in results])),
            "warm_p50_ms":float(np.median([r['warm_ms'] for r in results])),
            "max_stress_relative_error":max(r[m+'_maximum_stress_relative_error'] for r in results for m in ['cold','warm']),
            "incremental_direct_solution_relative_error":increment_error,
            "per_frame":results}


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=max(r['subdivisions'] for r in rows)
    cases=['smooth_grasp','unloading','contact_switch','random_reload']
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
    for ax,name in zip(axes.ravel(),cases):
        selected=[r for r in rows if r['subdivisions']==n and r['scenario']==name]
        frame=[r['frame'] for r in selected]
        ax.plot(frame,[r['evaluation_fraction'] for r in selected],label='Temporal evaluations / full scan',color='#216c93',lw=1.7)
        ax.plot(frame,[1.-r['naive_underestimate_fraction'] for r in selected],label='Previous-hotspot estimate / true max',color='#b64e3a',lw=1.4,ls='--')
        ax.set_title(name.replace('_',' '));ax.set_ylim(-.03,1.12);ax.grid(alpha=.2)
        ax.set_xlabel('Frame');ax.set_ylabel('Fraction / ratio')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    sample=next(r for r in rows if r['subdivisions']==n)
    fig.suptitle(f'CPU FEM temporal maximum search: {sample["nodes"]:,} nodes, {sample["tets"]:,} tetrahedra\nBlue: less is better. Red: 1 is accurate; dips are missed peaks.',fontsize=13)
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subdivisions',type=int,nargs='+',default=[6,12,20])
    parser.add_argument('--frames',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--group-size',type=int,default=64)
    parser.add_argument('--output-dir',type=Path,default=HERE)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    if args.frames<6 or args.repeats<1 or min(args.subdivisions)<2 or args.group_size<1:
        parser.error('Need frames>=6, repeats>=1, subdivisions>=2, group-size>=1.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cpu='unknown'
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
    output={"execution":{"CPU":cpu,"logical_cpus_visible":os.cpu_count(),"Python":platform.python_version(),
                         "NumPy":np.__version__,"SciPy":scipy.__version__,"BLAS_thread_limit":1,
                         "one_environment_per_query":True,"GPU_tested":False,"Genesis_tested":False},
            "method":{"load_basis_dimension":24,"patch_force_scale_N":100.,"spin_scale_rad_s":5.,
                      "exactness":"Numerically exact within fixed finite-patch traction and inertia model; float64 tolerances.",
                      "temporal_oracle_access":False,"fallback_active_fraction":.45,
                      "bounds":"min(spectral norm bound, componentwise triangle bound) + float64 safety margin",
                      "timed_cached_lu_includes":"balanced RHS, cached sparse solve, all-element stress and maximum; excludes diagnostics",
                      "timed_response_scan_includes":"response matvec, all-element von Mises and maximum",
                      "timed_temporal_includes":"equality test, current group hotspots, delta bounds, active scan, cache update"},
            "meshes":[],"limitations":["Shared virtual CPU; single-thread timings are observations, not dedicated-machine guarantees.",
                "Fixed 24-dimensional load space; arbitrary continuously moving contacts are not represented.",
                "CPU NumPy tracker with vectorized regions; no GPU timing or speedup extrapolation.",
                "No mesh-converged physical peak stress validation; tests compare the same discrete linear FEM model.",
                "Spectral bounds include numerical margins, not interval-arithmetic certification.",
                "First frame and large active sets scan all elements; repeated identical loads use exact cache."]}
    frame_rows=[]
    sequences=make_sequences(args.frames)
    json_path=args.output_dir/'rigid_stress_temporal_cpu_results.json'
    with threadpool_limits(limits=1):
        output['execution']['threadpools']=threadpool_info()
        for n in args.subdivisions:
            print(json.dumps({"event":"building_fem","subdivisions":n},ensure_ascii=False),flush=True)
            model=ResponseModel(n,args.group_size)
            mesh={"model":model.metadata,"cases":[]}
            output['meshes'].append(mesh)
            print(json.dumps({"event":"offline_ready","nodes":model.metadata['nodes'],"tets":model.ne,
                              "seconds":model.offline_model_seconds+model.offline_response_seconds}),flush=True)
            for name,(qs,w) in sequences.items():
                case=benchmark_case(model,name,qs,w,args.repeats,frame_rows)
                mesh['cases'].append(case)
                print(json.dumps({"event":"case_done","nodes":model.metadata['nodes'],"scenario":name,
                    "direct_ms":case['timings']['cached_lu']['p50_ms'],"response_ms":case['timings']['response_scan']['p50_ms'],
                    "temporal_ms":case['timings']['temporal']['p50_ms'],
                    "evaluated_fraction":case['temporal_mean_response_evaluation_fraction'],
                    "temporal_vs_scan":case['temporal_speedup_vs_response_scan_p50'],
                    "max_rel_error":case['temporal_vs_direct_max_relative_error']}),flush=True)
                json_path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
            del model
            gc.collect()
        print(json.dumps({"event":"cg_warm_start_check"}),flush=True)
        output['cg_warm_start']=cg_warm_start_test()
    output['all_checks_passed']=True
    output['total_distinct_frames_checked']=len(frame_rows)
    output['maximum_relative_error']=max(r['absolute_error_Pa']/max(r['oracle_max_Pa'],1.) for r in frame_rows)
    json_path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    csv_path=args.output_dir/'rigid_stress_temporal_cpu_frames.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    if args.plot:plot_results(frame_rows,args.output_dir/'rigid_stress_temporal_cpu_plot.png')
    print(json.dumps({"event":"complete","checks_passed":True,"distinct_frames":len(frame_rows),
                      "maximum_relative_error":output['maximum_relative_error'],"json":str(json_path),"csv":str(csv_path)}),flush=True)


if __name__=='__main__':main()
