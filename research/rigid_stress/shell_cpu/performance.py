"""CPU performance experiments for fixed linear-shell DPR recovery.

This is a research harness, not a Genesis or GPU benchmark.  Every timed
recovery method applies the same inertia-relieved linear operator and returns
the same complete top/bottom in-plane stress field.  Contacts move between
faces and their normal and tangential loads change independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from dataclasses import replace
import platform
import statistics
import time

for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import scipy
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits, threadpool_info


def _timer(function, repeats=5, inner=1):
    function()
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        measurements.append((time.perf_counter() - start) / inner)
    return {
        "median_ms": 1000 * statistics.median(measurements),
        "minimum_ms": 1000 * min(measurements),
        "maximum_ms": 1000 * max(measurements),
        "repeats": repeats,
        "inner": inner,
    }


def _local_vm(local):
    """Flattened top/bottom local [xx,yy,xy] -> VM at every sample."""
    values = np.asarray(local)
    if values.ndim == 1:
        values = values[:, None]
    triples = values.reshape((-1, 3, values.shape[-1]))
    x, y, xy = triples[:, 0], triples[:, 1], triples[:, 2]
    return np.sqrt(np.maximum(0.0, x*x - x*y + y*y + 3*xy*xy))


def _global_tensor_basis(model):
    """Fixed local-to-body rotation for six unique symmetric tensor components."""
    first, second = model.axes[:, :, :, 0], model.axes[:, :, :, 1]
    ij = ((0,0),(1,1),(2,2),(0,1),(1,2),(2,0))
    return np.stack([np.stack((first[:,:,i]*first[:,:,j],
                              second[:,:,i]*second[:,:,j],
                              first[:,:,i]*second[:,:,j]+second[:,:,i]*first[:,:,j]),axis=-1)
                     for i,j in ij],axis=2)


def _to_global_shell_field(local, basis):
    b = np.asarray(local).shape[-1]
    values = np.asarray(local).reshape((2,basis.shape[0],4,3,b))
    return np.einsum('egck,segkb->segcb',basis,values,optimize=True)


def _response_peak_tiled(response, forces, samples_per_tile=64):
    """Exact max-only evaluation with bounded stress-output scratch.

    This still stores the immutable response operator.  CPU Python tile-loop
    overhead is intentionally included, so it is not presumed faster than a
    full BLAS multiplication.
    """
    peak = np.zeros(forces.shape[1])
    stride = 3*samples_per_tile
    for start in range(0, response.shape[0], stride):
        values = response[start:start+stride] @ forces
        peak = np.maximum(peak, np.max(_local_vm(values), axis=0))
    return peak


class ExactResponseCache:
    """Exact current-model stress columns, populated on demand.

    A column is a response to a generalized nodal unit force, not a response
    to one fixed contact or symmetric grasp.  Arbitrary contact changes use
    new columns.  It must be invalidated if geometry/material/thickness/gauge
    change.  This deliberately exposes the O(ndof * nstress) memory tradeoff.
    """
    def __init__(self, model, stress_operator):
        self.model = model
        self.operator = stress_operator
        self.columns = np.empty((stress_operator.shape[0], model.ndof))
        self.valid = np.zeros(model.ndof, dtype=bool)
        self.misses = 0

    def apply(self, forces):
        f = np.asarray(forces)
        if f.ndim == 1:
            f = f[:, None]
        active = np.flatnonzero(np.any(f != 0.0, axis=1))
        missing = active[~self.valid[active]]
        if len(missing):
            unit = np.zeros((self.model.ndof, len(missing)))
            unit[missing, np.arange(len(missing))] = 1.0
            self.columns[:, missing] = self.operator @ self.model.solve(unit)
            self.valid[missing] = True
            self.misses += len(missing)
        if not len(active):
            return np.zeros((self.operator.shape[0], f.shape[1]))
        return self.columns[:, active] @ f[active]

    @property
    def allocated_bytes(self):
        return self.columns.nbytes + self.valid.nbytes


def _surface_force_operator(model, side=1):
    """Map 3N surface forces to 6N shell force/couple loads exactly.

    The side is physical inner/outer provenance, not contact position.  Mixed
    sides use the sum of two operators; arbitrary independent nodal couples
    still require the general 6N map.
    """
    n = len(model.vertices)
    operator = np.zeros((model.ndof,3*n))
    for i, director in enumerate(model.directors):
        delta = side*.5*model.h*director
        cross = np.array([[0.,-delta[2],delta[1]],
                          [delta[2],0.,-delta[0]],
                          [-delta[1],delta[0],0.]])
        operator[6*i:6*i+3,3*i:3*i+3] = np.eye(3)
        operator[6*i+3:6*i+6,3*i:3*i+3] = cross
    return operator


def batched_witness_transfer(transfer, contacts):
    """Vectorized complete-disk fast path with the scalar conservative fallback.

    This is an execution experiment for the exact same projected-Hertz load law
    as contacts.py method='fast'.  A checked current material-face witness is
    required; clipped, unresolved, unloaded or unsupported cases go through
    the original scalar implementation, including all its failures.
    """
    from .contacts import _disk_cubature
    batch = len(contacts)
    out = np.zeros((transfer.ndof,batch))
    fallback = np.ones(batch,dtype=bool)
    chosen = np.array([j for j,c in enumerate(contacts)
        if c.face_hint is not None and 0 <= int(c.face_hint) < len(transfer.quads)
        and c.side in (-1,1) and np.linalg.norm(c.force)>0],dtype=int)
    if len(chosen):
        selected = [contacts[j] for j in chosen]
        point = np.stack([c.point for c in selected])
        force = np.stack([c.force for c in selected])
        normal = np.stack([c.normal for c in selected])
        normal /= np.linalg.norm(normal,axis=1)[:,None]
        direction = force/np.linalg.norm(force,axis=1)[:,None]
        fn = -np.einsum('bi,bi->b',force,normal)
        curvature = np.stack([c.curvatures for c in selected])
        modulus = transfer.E/(1-transfer.nu**2) if transfer.contact_modulus is None else transfer.contact_modulus
        with np.errstate(invalid='ignore',divide='ignore'):
            radius = (3*fn/(4*modulus*curvature.mean(axis=1)))**(1/3)
        for j,c in enumerate(selected):
            if c.radius is not None:
                radius[j] = c.radius
        good = (fn>0)&(curvature.min(axis=1)>0)&np.isfinite(radius)&(radius>0)
        side = np.array([c.side for c in selected])
        hint = np.array([int(c.face_hint) for c in selected])
        nodes = transfer.quads[hint]
        directors = transfer.directors[nodes]
        face_points = transfer.vertices[nodes]+side[:,None,None]*.5*transfer.h*directors
        axis = np.eye(3)[np.argmin(np.abs(normal),axis=1)]
        first = np.cross(normal,axis)
        first /= np.linalg.norm(first,axis=1)[:,None]
        plane = np.stack((first,np.cross(normal,first)),axis=-1)
        relative = face_points-point[:,None,:]
        projection = np.einsum('bni,bi->bn',relative,normal)/np.einsum('bi,bi->b',direction,normal)[:,None]
        poly = np.einsum('bni,bik->bnk',relative-projection[:,:,None]*direction[:,None,:],plane)
        area = np.sum(poly[:,:,0]*np.roll(poly[:,:,1],-1,axis=1)-poly[:,:,1]*np.roll(poly[:,:,0],-1,axis=1),axis=1)
        edge = np.roll(poly,-1,axis=1)-poly
        cross_to_origin = edge[:,:,0]*(-poly[:,:,1])-edge[:,:,1]*(-poly[:,:,0])
        good &= np.all(cross_to_origin*np.sign(area)[:,None] > radius[:,None]*np.linalg.norm(edge,axis=2)*(1+2e-12),axis=1)
        p2,w2 = _disk_cubature(2)
        p3,w3 = _disk_cubature(3)
        unit = np.vstack((np.zeros((1,2)),p2,p3))
        target = radius[:,None,None]*unit[None]
        origin = poly.mean(axis=1)
        W = poly-origin[:,None,:]
        target = target-origin[:,None,:]
        x = np.zeros(target.shape[:2]); y = np.zeros_like(x)
        determinant = np.ones_like(x)
        for _ in range(12):
            N = .25*np.stack(((1-x)*(1-y),(1+x)*(1-y),(1+x)*(1+y),(1-x)*(1+y)),axis=-1)
            dx = .25*np.stack((-(1-y),1-y,1+y,-(1+y)),axis=-1)
            dy = .25*np.stack((-(1-x),-(1+x),1+x,1-x),axis=-1)
            residual = np.einsum('bpn,bni->bpi',N,W)-target
            jx = np.einsum('bpn,bni->bpi',dx,W); jy = np.einsum('bpn,bni->bpi',dy,W)
            determinant = jx[:,:,0]*jy[:,:,1]-jx[:,:,1]*jy[:,:,0]
            safe = np.where(np.abs(determinant)>1e-30,determinant,1.)
            sx = (jy[:,:,1]*residual[:,:,0]-jy[:,:,0]*residual[:,:,1])/safe
            sy = (-jx[:,:,1]*residual[:,:,0]+jx[:,:,0]*residual[:,:,1])/safe
            x -= sx; y -= sy
            if max(np.max(np.abs(sx)),np.max(np.abs(sy)))<3e-14:
                break
        N = .25*np.stack(((1-x)*(1-y),(1+x)*(1-y),(1+x)*(1+y),(1-x)*(1+y)),axis=-1)
        residual = np.einsum('bpn,bni->bpi',N,W)-target
        scale = np.linalg.norm(W,axis=2).max(axis=1)
        good &= np.all(np.linalg.norm(residual,axis=2)<2e-11*scale[:,None],axis=1)
        good &= np.all((np.abs(x)<=1+1e-9)&(np.abs(y)<=1+1e-9)&(np.abs(determinant)>1e-30),axis=1)
        hit = np.einsum('bn,bni->bi',N[:,0],face_points)
        surface_normal = side[:,None]*np.einsum('bn,bni->bi',N[:,0],directors)
        local_bound = radius+np.linalg.norm(face_points-face_points.mean(axis=1)[:,None],axis=2).max(axis=1)
        good &= (np.linalg.norm(hit-point,axis=1)<local_bound)&(np.einsum('bi,bi->b',surface_normal,normal)>.25)
        lower = np.einsum('p,bpn->bn',w2,N[:,1:1+len(w2)])
        weights = np.einsum('p,bpn->bn',w3,N[:,1+len(w2):])
        error = np.linalg.norm(weights-lower,axis=1)/np.linalg.norm(weights,axis=1)
        good &= error<=1e-10
        mass = weights.sum(axis=1)
        good &= np.abs(mass-1)<1e-9
        weights /= mass[:,None]
        nodal = weights[:,:,None]*force[:,None,:]
        couples = side[:,None,None]*.5*transfer.h*np.cross(directors,nodal)
        resultant = nodal.sum(axis=1)
        moment = (np.cross(transfer.vertices[nodes],nodal)+couples).sum(axis=1)
        ferr = np.linalg.norm(resultant-force,axis=1)/np.linalg.norm(force,axis=1)
        merr = np.linalg.norm(moment-np.cross(point,force),axis=1)/(np.linalg.norm(force,axis=1)*radius)
        good &= (ferr<1e-10)&(merr<1e-8)
        for j,c in enumerate(selected):
            if c.friction is not None:
                good[j] &= np.linalg.norm(force[j]+fn[j]*normal[j]) <= c.friction*fn[j]*(1+1e-12)
        accepted = np.flatnonzero(good)
        view = out.reshape((-1,6,batch))
        for local in range(4):
            view[nodes[accepted,local],:3,chosen[accepted]] = nodal[accepted,local]
            view[nodes[accepted,local],3:,chosen[accepted]] = couples[accepted,local]
        fallback[chosen[accepted]] = False
    for j in np.flatnonzero(fallback):
        out[:,j] = transfer.transfer(contacts[j],method='fast')[0]
    return out,dict(fast_environments=int((~fallback).sum()),fallback_environments=int(fallback.sum()))


def _contact_bank(model, count=128, seed=20261024):
    from .contacts import Contact
    from .geometry import egg_geometry
    rng = np.random.default_rng(seed)
    bank = []
    for k in range(count):
        # A deterministic, non-symmetric trajectory around all surface faces.
        # This uses no fixed set of grasp positions or opposite paired forces.
        i = (17*k + k//7) % len(model.quads)
        q = model.quads[i]
        center = np.mean(model.vertices[q], axis=0)
        normal, curvatures = egg_geometry(center[None])
        normal = np.asarray(normal)[0]
        curvature = np.asarray(curvatures)[0]
        # Small intra-face motion, independent forces and friction direction.
        edge1 = model.vertices[q[1]] - model.vertices[q[0]]
        edge2 = model.vertices[q[3]] - model.vertices[q[0]]
        point = center + .08*np.sin(.37*k)*edge1 + .08*np.cos(.61*k)*edge2
        tangent = edge1 - normal*np.dot(normal, edge1)
        tangent /= np.linalg.norm(tangent)
        tangent2 = np.cross(normal, tangent)
        fn = .01 + .04*(.5+.5*np.sin(.31*k))
        shear = .25*fn*(np.cos(.29*k)*tangent+np.sin(.29*k)*tangent2)
        force = -fn*normal + shear
        # The contact module convention is outward normal; compression is -F.n.
        # Use the same director-interpolated physical face as the shell backend.
        # The smooth analytic surface supplies a curvature estimate only.
        point = point + .5*model.h*np.mean(model.directors[q], axis=0)
        bank.append(Contact(point=point, force=force, normal=normal,
                            curvatures=curvature, side=1,face_hint=int(i)))
    return bank


def _unpack_transfer(value):
    return value[0] if isinstance(value, tuple) else value


def benchmark_model(resolution=4, batches=(1, 8, 32, 128), repeats=5):
    from .core import ShellModel
    from .geometry import egg_mesh
    from .contacts import ShellDPRTransfer
    vertices, quads = egg_mesh(resolution)
    start = time.perf_counter()
    model = ShellModel(vertices, quads, E=1e10, nu=.3, h=.0005)
    construction_seconds = time.perf_counter() - start
    start = time.perf_counter()
    factor = model.factorize()
    factor_seconds = time.perf_counter() - start
    start = time.perf_counter()
    stress = model.stress_matrix(kind="local").tocsr()
    stress_operator_seconds = time.perf_counter() - start
    tensor_basis = _global_tensor_basis(model)
    transfer = ShellDPRTransfer(model)
    contacts = _contact_bank(model, max(batches))
    start = time.perf_counter()
    loads_list = [_unpack_transfer(transfer.transfer(c)) for c in contacts]
    transfer_bank_seconds = time.perf_counter() - start
    loads = np.stack(loads_list, axis=1)
    contacts_unhinted = [replace(c,face_hint=None) for c in contacts]
    taylor_loads = np.stack([_unpack_transfer(transfer.transfer(c,method='taylor'))
                            for c in contacts],axis=1)
    fast_loads = np.stack([_unpack_transfer(transfer.transfer(c,method='fast'))
                          for c in contacts],axis=1)
    batched_loads, batch_transfer_info = batched_witness_transfer(transfer,contacts)
    cache = ExactResponseCache(model, stress)
    rows = []
    # Full response is optional and measured separately, never excluded silently.
    start = time.perf_counter()
    response = stress @ model.solve(np.eye(model.ndof))
    response_build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    surface_operator = _surface_force_operator(model,side=1)
    surface_response = stress @ model.solve(surface_operator)
    surface_response_build_seconds = time.perf_counter()-start

    for batch in batches:
        f = loads[:, :batch]
        surface_forces = f.reshape((-1,6,batch))[:,:3].reshape((-1,batch))
        direct = stress @ model.solve(f)
        scale = max(np.linalg.norm(direct), 1e-100)
        full = model.stress(model.solve(f))
        canonical = np.stack((full['sigma_top'],full['sigma_bottom']))
        canonical_symmetric = np.stack([canonical[:,:,:,i,j,:]
            for i,j in ((0,0),(1,1),(2,2),(0,1),(1,2),(2,0))],axis=3)
        global_direct = _to_global_shell_field(direct,tensor_basis)
        global_error = np.linalg.norm(global_direct-canonical_symmetric) / max(np.linalg.norm(canonical_symmetric),1e-100)
        cache_start = time.perf_counter()
        cached = cache.apply(f)
        cache_cold_seconds = time.perf_counter() - cache_start
        response_result = response @ f
        scalar_result = np.column_stack([stress @ model.solve(f[:, j]) for j in range(batch)])
        row = {
            "batch": batch,
            "active_dof_union": int(np.count_nonzero(np.any(f != 0, axis=1))),
            "mean_active_dofs_per_environment": float(np.mean(np.count_nonzero(f, axis=0))),
            "cache_cold_ms": cache_cold_seconds*1000,
            "cache_columns_valid": int(cache.valid.sum()),
            "cache_relative_field_error": float(np.linalg.norm(cached-direct)/scale),
            "dense_response_relative_field_error": float(np.linalg.norm(response_result-direct)/scale),
            "scalar_loop_relative_field_error": float(np.linalg.norm(scalar_result-direct)/scale),
            "global_tensor_map_vs_core_relative_error": float(global_error),
            "surface_response_relative_field_error": float(np.linalg.norm(surface_response@surface_forces-direct)/scale),
            "taylor_relative_nodal_load_error": float(np.linalg.norm(taylor_loads[:,:batch]-f)/max(np.linalg.norm(f),1e-100)),
            "taylor_relative_field_error": float(np.linalg.norm(stress@model.solve(taylor_loads[:,:batch])-direct)/scale),
            "fast_relative_nodal_load_error": float(np.linalg.norm(fast_loads[:,:batch]-f)/max(np.linalg.norm(f),1e-100)),
            "fast_relative_field_error": float(np.linalg.norm(stress@model.solve(fast_loads[:,:batch])-direct)/scale),
            "batched_fast_relative_nodal_load_error": float(np.linalg.norm(batched_loads[:,:batch]-fast_loads[:,:batch])/max(np.linalg.norm(fast_loads[:,:batch]),1e-100)),
            "methods": {},
        }
        normals = np.stack([c.normal for c in contacts[:batch]])
        current_forces = np.stack([c.force for c in contacts[:batch]])
        mean_curvature = np.mean(np.stack([c.curvatures for c in contacts[:batch]]),axis=1)
        compressive_force = -np.sum(current_forces*normals,axis=1)
        modulus = model.E/(1-model.nu*model.nu)
        def estimate_radii():
            return (3*compressive_force/(4*modulus*mean_curvature))**(1/3)
        row["radii_m"] = [float(np.min(estimate_radii())),float(np.max(estimate_radii()))]
        row["radius_formula"] = _timer(estimate_radii,repeats=repeats,inner=100)
        def current_fast_loads():
            return np.column_stack([_unpack_transfer(transfer.transfer(c,method='fast')) for c in contacts[:batch]])
        def current_fast_surface_field():
            current = current_fast_loads().reshape((-1,6,batch))[:,:3].reshape((-1,batch))
            return surface_response@current
        def batch_fast_loads():
            return batched_witness_transfer(transfer,contacts[:batch])[0]
        def batch_fast_surface_field():
            current = batch_fast_loads().reshape((-1,6,batch))[:,:3].reshape((-1,batch))
            return surface_response@current
        methods = {
            "prefactorized_multi_rhs_full": lambda: stress @ model.solve(f),
            "prefactorized_column_loop_full": lambda: np.column_stack([
                stress @ model.solve(f[:, j]) for j in range(batch)]),
            "warm_sparse_response_cache_full": lambda: cache.apply(f),
            "dense_precomputed_response_full": lambda: response @ f,
            "outer_surface_response_full": lambda: surface_response @ surface_forces,
            "prefactorized_multi_rhs_peak": lambda: np.max(_local_vm(stress @ model.solve(f)), axis=0),
            "dense_precomputed_response_peak": lambda: np.max(_local_vm(response @ f), axis=0),
            "outer_surface_response_peak": lambda: np.max(_local_vm(surface_response @ surface_forces),axis=0),
            "tiled_precomputed_response_peak": lambda: _response_peak_tiled(response, f),
            "prefactorized_multi_rhs_full_global_tensor": lambda: _to_global_shell_field(stress @ model.solve(f), tensor_basis),
            "dense_precomputed_response_full_global_tensor": lambda: _to_global_shell_field(response @ f, tensor_basis),
            "global_tensor_conversion": lambda: _to_global_shell_field(direct, tensor_basis),
            "current_contact_transfer": lambda: [_unpack_transfer(transfer.transfer(c)) for c in contacts[:batch]],
            "unhinted_contact_transfer": lambda: [_unpack_transfer(transfer.transfer(c)) for c in contacts_unhinted[:batch]],
            "hinted_taylor_contact_transfer": lambda: [_unpack_transfer(transfer.transfer(c,method='taylor')) for c in contacts[:batch]],
            "hinted_fast_contact_transfer": current_fast_loads,
            "fast_transfer_plus_prefactorized_full": lambda: stress@model.solve(current_fast_loads()),
            "fast_transfer_plus_surface_response_full": current_fast_surface_field,
            "fast_transfer_plus_surface_response_peak": lambda: np.max(_local_vm(current_fast_surface_field()),axis=0),
            "batched_fast_contact_transfer": batch_fast_loads,
            "batched_fast_transfer_plus_prefactorized_full": lambda: stress@model.solve(batch_fast_loads()),
            "batched_fast_transfer_plus_surface_response_full": batch_fast_surface_field,
            "batched_fast_transfer_plus_surface_response_peak": lambda: np.max(_local_vm(batch_fast_surface_field()),axis=0),
        }
        for name, fun in methods.items():
            row["methods"][name] = _timer(fun, repeats=repeats, inner=2 if batch <= 8 else 1)
        row["tiled_peak_relative_error"] = float(np.linalg.norm(
            _response_peak_tiled(response, f)-np.max(_local_vm(direct),axis=0)) /
            max(np.linalg.norm(np.max(_local_vm(direct),axis=0)),1e-100))
        row["numerical_checks_passed"] = max(row["cache_relative_field_error"],
            row["dense_response_relative_field_error"],row["scalar_loop_relative_field_error"],
            row["tiled_peak_relative_error"],row["global_tensor_map_vs_core_relative_error"]) < 1e-8
        row["numerical_checks_passed"] &= row["surface_response_relative_field_error"] < 1e-8
        row["fast_numerical_checks_passed"] = max(row["fast_relative_nodal_load_error"],
                                                   row["fast_relative_field_error"],
                                                   row["batched_fast_relative_nodal_load_error"]) < 1e-8
        row["taylor_numerical_checks_passed"] = max(row["taylor_relative_nodal_load_error"],
                                                     row["taylor_relative_field_error"]) < 1e-8
        # Include factorization in one scalar recovery as a clean reference.
        if batch == 1:
            def refactor_one():
                # Same ordering as the cached model factor.  Rebuild numeric
                # factorization for each load, as an explicit cold baseline.
                fresh = splu(model.K[factor.free_dofs][:, factor.free_dofs].tocsc(),
                             permc_spec='MMD_AT_PLUS_A')
                rhs, _ = model.balance_load(f[:, 0])
                q = np.zeros(model.ndof)
                q[factor.free_dofs] = fresh.solve(np.asarray(rhs)[factor.free_dofs])
                return stress @ q
            row["methods"]["refactor_scalar_full"] = _timer(refactor_one, repeats=repeats)
        for name in ("prefactorized_multi_rhs_full", "warm_sparse_response_cache_full",
                     "dense_precomputed_response_full"):
            ms = row["methods"][name]["median_ms"]
            row["methods"][name]["environment_recoveries_per_second"] = batch*1000/ms
        rows.append(row)
    return {
        "resolution": resolution,
        "vertices": len(vertices), "quads": len(quads), "ndof": model.ndof,
        "stress_scalar_outputs": stress.shape[0],
        "global_symmetric_tensor_scalar_outputs": 2*stress.shape[0],
        "construction_seconds": construction_seconds, "factor_seconds": factor_seconds,
        "stress_operator_seconds": stress_operator_seconds,
        "contact_bank_transfer_seconds": transfer_bank_seconds,
        "batched_transfer_info": batch_transfer_info,
        "contact_bank_scope": "Changing/asymmetric force and shear directions across current witnessed face interiors; clipped/vertex/inner-side/unloaded/stale-witness cases checked separately in batched_transfer_checks.json",
        "dense_response_build_seconds": response_build_seconds,
        "dense_response_bytes": response.nbytes,
        "outer_surface_response_build_seconds": surface_response_build_seconds,
        "outer_surface_response_bytes": surface_response.nbytes,
        "on_demand_cache_allocated_bytes": cache.allocated_bytes,
        "stiffness_nnz": model.K.nnz,
        "factor_nnz": factor.lu.L.nnz+factor.lu.U.nnz,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("shell_cpu/results/performance.json"))
    args = parser.parse_args()
    source_files = [Path(__file__).with_name(name) for name in
                    ('core.py','geometry.py','contacts.py','performance.py')]
    source_hashes_before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    with threadpool_limits(limits=1):
        models = [benchmark_model(n, args.batches, args.repeats) for n in args.resolution]
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), platform.processor())
        report = {
            "scope": "CPU research shell recovery only; not Genesis scene steps, RL transitions, GPU throughput or contact-reference qualification",
            "cpu": cpu, "numpy": np.__version__, "scipy": scipy.__version__,
            "threads": 1, "threadpools": threadpool_info(), "models": models,
            "source_sha256": source_hashes_before,
            "source_unchanged_during_run": all(hashlib.sha256(p.read_bytes()).hexdigest()==source_hashes_before[p.name]
                                               for p in source_files),
            "all_numerical_checks_passed": all(row["numerical_checks_passed"]
                for model in models for row in model["rows"]),
            "all_fast_transfer_checks_passed": all(row["fast_numerical_checks_passed"]
                for model in models for row in model["rows"]),
            "semantics": "Same current generalized loads, free-body inertia relief, fixed shell stiffness, full local top/bottom in-plane stress and plane-stress von Mises maximum",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        for model in models:
            print(json.dumps({"ndof": model["ndof"], "rows": model["rows"]}))


if __name__ == "__main__":
    main()
