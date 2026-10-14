"""Audit a continuously legal disk wrench missed by the fixed Q10 circular sample hull."""

import json
import sys
from pathlib import Path

import numpy as np
from circle_disk_candidate_v2 import kernel_build_single_face_disks
from polar_cap_refinement_candidate_v1 import kernel_refine_verified_disk
from scipy.optimize import linprog
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    create_traction_info,
    create_tractions,
    kernel_fit_tractions,
    kernel_scatter_tractions,
)
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import WrenchPatch
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit
from tests.rigid.test_stress_traction import kernel_traction_queries

backend, destination = sys.argv[1], Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=False)
gs.init(backend={"cpu": gs.cpu, "cuda": gs.cuda}[backend], precision="64", logging_level="warning")
asset = Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")
with np.load(asset) as saved:
    vertices, tets, faces = (saved[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
triangle = vertices[faces[0]]
anchor = triangle.mean(axis=0)
owner = np.flatnonzero(np.isin(tets, faces[0]).sum(axis=1) == 3)[0]
opposite = vertices[tets[owner][~np.isin(tets[owner], faces[0])][0]]
normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
normal /= np.linalg.norm(normal)
normal *= np.sign(normal @ (opposite - triangle[0]))
tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
second = np.cross(normal, tangent)
radius = 1e-5
fractions = np.array([0.98, 0.99, 0.98, 0.99, 0.98, 0.99])
angles = np.repeat([0.0, 0.7, 1.8], 2)
frictions = np.repeat([0.5, 0.85, 1.2], 2)
directions = np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * second
forces = normal + 0.4 * directions
points = anchor - (fractions * radius / 0.4)[:, None] * normal
centroids = anchor + fractions[:, None] * radius * directions
patch_radii = 0.25 * radius * (1.0 - fractions)
assert ((np.linalg.norm(centroids - anchor, axis=1) + patch_radii) < radius).all()
# A positive-area uniform vector traction centered at each centroid matches each unchanged original point wrench.
np.testing.assert_allclose(np.cross(centroids - points, forces), 0, rtol=0, atol=1e-18)
assert (frictions > 0.4).all()
options = gs.options.RigidStressOptions(mesh=asset, method="direct")
model = StressModel(options)
surface = StressSurface(10, model.info)
contacts, sources = create_contacts(1, len(fractions), radius), create_sources(1, len(fractions))
contacts.valid.fill(True)
contacts.position.from_numpy(points[None])
contacts.force.from_numpy(forces[None])
contacts.friction.from_numpy(frictions[None])
sources.geom.fill(7)
sources.pair.fill(11)
sources.witness.from_numpy(np.broadcast_to(anchor, (1, len(fractions), 3)).copy())
epsilon = float(np.finfo(gs.np_float).eps)
kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
assert (qd_to_numpy(sources.status, copy=True) == 0).all()
info = create_traction_info(surface.info)
baseline = create_tractions(100, 1, len(fractions), local_samples=True)
kernel_build_single_face_disks(contacts, baseline, surface.info, epsilon)
kernel_fit_tractions(contacts, sources, baseline, info, epsilon, 9e-9, 128)
base_positions = qd_to_numpy(baseline.sample_positions, transpose=True, copy=True)[:, :, 0]
base_status = qd_to_numpy(baseline.status, transpose=True, copy=True)[:, 0]
rows = []
for i_b in range(len(fractions)):
    normalized = (base_positions[i_b] - anchor) / radius
    target = fractions[i_b] * directions[i_b]
    lp = linprog(
        np.zeros(100),
        A_eq=np.vstack((np.ones(100), normalized.T)),
        b_eq=np.concatenate(([1.0], target)),
        bounds=(0, None),
        method="highs",
    )
    gap = float(target @ directions[i_b] - np.max(normalized @ directions[i_b]))
    assert not lp.success and gap > 0
    rows.append(
        {
            "fraction": float(fractions[i_b]),
            "angle": float(angles[i_b]),
            "actual_mu": float(frictions[i_b]),
            "positive_area_witness_radius_m": float(patch_radii[i_b]),
            "fixed_q10_status": int(base_status[i_b]),
            "fixed_q10_hull_gap_m": radius * gap,
            "compression_lp_status": int(lp.status),
        }
    )
refined = create_tractions(900, 1, len(fractions), local_samples=True)
kernel_build_single_face_disks(contacts, refined, surface.info, epsilon)
kernel_refine_verified_disk(contacts, sources, refined, surface.info)
kernel_fit_tractions(contacts, sources, refined, info, epsilon, 9e-9, 128)
status = qd_to_numpy(refined.status, transpose=True, copy=True)[:, 0]
evaluations = qd_to_numpy(refined.evaluations, transpose=True, copy=True)[:, 0]
for i_b, row in enumerate(rows):
    row.update(refined_status=int(status[i_b]), refined_evaluations=int(evaluations[i_b]))
(destination / "preliminary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows), flush=True)
assert (status == 0).all(), "Refined sampling alone did not resolve the native algorithm; preserve this failure"
native_positions = qd_to_numpy(refined.sample_positions, transpose=True, copy=True)[:, :, 0]
profiles = qd_to_numpy(refined.profile, transpose=True, copy=True)[:, :, 0]
sample_shape = qd_to_numpy(refined.sample_shape, transpose=True, copy=True)[:, :, 0]
query = V_VEC(3, dtype=gs.qd_float, shape=refined.profile.shape)
kernel_traction_queries(query, contacts, refined, info)
native_force = qd_to_numpy(query, transpose=True, copy=True)[:, :, 0]
moment = qd_to_numpy(sources.anchor_moment, transpose=True, copy=True)[:, 0]
state, omega, error = (
    model.create_state(len(fractions), "full"),
    V_VEC(3, dtype=gs.qd_float, shape=(len(fractions),)),
    V(dtype=gs.qd_int, shape=(len(fractions),)),
)
omega.fill(0)
kernel_scatter_tractions(contacts, refined, state, model.info, surface.info, info, error, reset_force=True)
assert (qd_to_numpy(error, copy=True) == 0).all()
model.recover(omega, state)
assert qd_to_numpy(state.valid, copy=True).all()
oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, copy=True))
factor = splu(oracle.k[free][:, free].tocsc())
surface_nodes = qd_to_numpy(model.info.surface_nodes, copy=True)[0]
for i_b, row in enumerate(rows):
    positions = native_positions[i_b]
    assert (np.linalg.norm(positions - anchor, axis=1) < radius).all()
    reference = cpu_resolved_wrench_fit(
        positions,
        profiles[i_b],
        np.broadcast_to(normal, positions.shape),
        WrenchPatch(anchor, forces[i_b], radius, frictions[i_b], moment[i_b]),
    )
    np.testing.assert_allclose(native_force[i_b], reference, rtol=1e-4, atol=1e-10)
    force_error = np.linalg.norm(native_force[i_b].sum(axis=0) - forces[i_b]) / np.linalg.norm(forces[i_b])
    moment_error = np.linalg.norm(np.cross(positions - points[i_b], native_force[i_b]).sum(axis=0)) / (
        radius * np.linalg.norm(forces[i_b])
    )
    assert max(force_error, moment_error) <= 1e-8
    compression = native_force[i_b] @ normal
    excess = np.linalg.norm(native_force[i_b] - compression[:, None] * normal, axis=1) - frictions[i_b] * compression
    assert compression.min() >= -1e-12 and excess.max() <= 1e-12
    nodal = np.zeros((oracle.ndof // 3, 3))
    np.add.at(
        nodal,
        np.broadcast_to(surface_nodes, (900, 6)).reshape(-1),
        (sample_shape[i_b, :, :, None] * reference[:, None]).reshape(-1, 3),
    )
    rhs = oracle.balance(nodal.reshape(oracle.ndof, 1))
    displacement = np.zeros_like(rhs)
    displacement[free] = factor.solve(rhs[free])
    actual = qd_to_numpy(state.displacement, transpose=True, copy=True)[i_b].reshape(oracle.ndof, 1)
    residual = float(np.linalg.norm(oracle.k @ actual - rhs))
    assert residual <= max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
    tensor, vm = stress_field(oracle.glambda, oracle.elements, displacement[:, 0], options.young, options.poisson)
    np.testing.assert_allclose(
        qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[i_b], tensor, rtol=1e-4, atol=1e-3
    )
    np.testing.assert_allclose(
        qd_to_numpy(state.von_mises, transpose=True, copy=True)[i_b, :, :, 0], vm, rtol=1e-4, atol=1e-3
    )
    peak = float(qd_to_numpy(state.peak, copy=True)[i_b])
    np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
    row.update(
        force_error=float(force_error),
        original_point_moment_error=float(moment_error),
        full_gauge_residual_N=residual,
        peak_native_Pa=peak,
        peak_cpu_Pa=float(vm.max()),
    )
np.savez(
    destination / "inputs-and-samples.npz",
    anchor=anchor,
    point=points,
    force=forces,
    radius=radius,
    friction=frictions,
    centroid=centroids,
    witness_radius=patch_radii,
    fixed_positions=base_positions,
    refined_positions=native_positions,
    refined_profiles=profiles,
    refined_force=native_force,
)
(destination / "summary.json").write_text(
    json.dumps(
        {
            "backend": backend,
            "cases": rows,
            "scope": "Legal near-capacity disk integration diagnostic. Same continuous area/profile law, unchanged physical inputs and final budgets. Experimental local numerical refinement, not production selection or a physical convergence study.",
        },
        indent=2,
    )
    + "\n"
)
