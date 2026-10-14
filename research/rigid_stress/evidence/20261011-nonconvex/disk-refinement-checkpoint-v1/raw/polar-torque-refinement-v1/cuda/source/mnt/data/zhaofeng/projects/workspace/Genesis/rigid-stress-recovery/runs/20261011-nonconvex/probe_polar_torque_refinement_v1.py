"""Audit a continuously legal disk wrench missed by the fixed Q10 circular sample hull."""

import json
import sys
from pathlib import Path

import numpy as np
from circle_disk_candidate_v2 import kernel_build_single_face_disks
from polar_cap_refinement_candidate_v2 import kernel_refine_verified_disk
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
fractions = np.tile([0.99, 0.999], 3)
angles = np.zeros(6)
frictions = np.repeat([0.5, 0.85, 1.2], 2)
directions = np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * second
forces = np.broadcast_to(normal, (6, 3)).copy()
points = np.broadcast_to(anchor, (6, 3)).copy()
inner, outer = fractions + 0.5 * (1 - fractions), fractions + 0.75 * (1 - fractions)
mean_radius = 2 / 3 * (outer**2 + outer * inner + inner**2) / (outer + inner)
swirl_ratio = frictions * fractions / mean_radius
assert (swirl_ratio < frictions).all() and ((outer < 1) & (inner < outer)).all()
# Finite positive-area annuli with uniform compression and circumferential traction realize these original wrenches.
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
sources.free_moment.from_numpy((radius * fractions * frictions)[None, :, None] * normal)
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
    maximum_radius = np.linalg.norm(base_positions[i_b] - anchor, axis=1).max() / radius
    gap = float(fractions[i_b] - maximum_radius)
    assert gap > 0
    rows.append(
        {
            "fraction": float(fractions[i_b]),
            "angle": float(angles[i_b]),
            "actual_mu": float(frictions[i_b]),
            "positive_area_annulus_inner_R": float(inner[i_b]),
            "positive_area_annulus_outer_R": float(outer[i_b]),
            "swirl_ratio": float(swirl_ratio[i_b]),
            "fixed_q10_status": int(base_status[i_b]),
            "fixed_q10_hull_gap_m": radius * gap,
            "maximum_baseline_sample_radius_R": float(maximum_radius),
        }
    )
refined = create_tractions(900, 1, len(fractions), local_samples=True)
kernel_build_single_face_disks(contacts, refined, surface.info, epsilon)
kernel_refine_verified_disk(contacts, sources, refined, surface.info, epsilon)
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
gauss, weights = np.polynomial.legendre.leggauss(10)
gauss, weights = (gauss + 1.0) / 2.0, weights / 2.0
u, v = np.repeat(gauss, 10), np.tile(gauss, 10)
for i_b, row in enumerate(rows):
    positions = native_positions[i_b]
    assert (np.linalg.norm(positions - anchor, axis=1) < radius).all()
    # Independent CPU geometry uses the analytic manufactured centroid, not native chart data or fitted duals.
    fraction, angle = fractions[i_b], angles[i_b]
    half_width = min(0.5 * np.pi, np.sqrt(8 * (1 - fraction)))
    radial_edges = np.array([0.0, max(0.25 * (1 - fraction), fraction - 8 * (1 - fraction)), 0.5 * (1 + fraction), 1.0])
    angular_edges = np.array([angle - np.pi, angle - half_width, angle + half_width, angle + np.pi])
    r = (radial_edges[:-1, None] + np.diff(radial_edges)[:, None] * u).reshape(3, 1, 100)
    theta = (angular_edges[:-1, None] + np.diff(angular_edges)[:, None] * v).reshape(1, 3, 100)
    reference_positions = anchor + radius * r[..., None] * (
        np.cos(theta)[..., None] * tangent + np.sin(theta)[..., None] * second
    )
    area = radius**2 * r * np.diff(radial_edges)[:, None, None] * np.diff(angular_edges)[None, :, None]
    area *= np.repeat(weights, 10) * np.tile(weights, 10)
    assert (area > 0).all()
    np.testing.assert_allclose(area.sum(), np.pi * radius**2, rtol=2e-14, atol=0)
    reference_profiles = (area * np.exp(-0.5 * (r / 0.45) ** 2) * (1 - r**2) ** 2).reshape(-1)
    reference_positions = reference_positions.reshape(-1, 3)
    local = np.linalg.lstsq((triangle[1:] - triangle[0]).T, (reference_positions - triangle[0]).T, rcond=None)[0].T
    bary = np.column_stack((1 - local.sum(axis=1), local))
    assert (bary > 0).all()
    basis = np.column_stack((bary * (2 * bary - 1), 4 * bary[:, [0, 0, 1]] * bary[:, [1, 2, 2]]))
    np.testing.assert_allclose(positions, reference_positions, rtol=0, atol=128 * epsilon * np.linalg.norm(anchor))
    profile_roundoff = 512 * epsilon * (1 + np.linalg.norm(anchor) / radius) / (1 - fraction)
    np.testing.assert_allclose(profiles[i_b], reference_profiles, rtol=profile_roundoff, atol=0)
    np.testing.assert_allclose(sample_shape[i_b], basis, rtol=2e-12, atol=2e-14)
    reference = cpu_resolved_wrench_fit(
        reference_positions,
        reference_profiles,
        np.broadcast_to(normal, positions.shape),
        WrenchPatch(anchor, forces[i_b], radius, frictions[i_b], moment[i_b]),
    )
    np.testing.assert_allclose(native_force[i_b], reference, rtol=1e-4, atol=1e-10)
    force_error = np.linalg.norm(native_force[i_b].sum(axis=0) - forces[i_b]) / np.linalg.norm(forces[i_b])
    moment_error = np.linalg.norm(
        np.cross(positions - points[i_b], native_force[i_b]).sum(axis=0)
        - radius * fractions[i_b] * frictions[i_b] * normal
    ) / (radius * np.linalg.norm(forces[i_b]))
    assert max(force_error, moment_error) <= 1e-8
    compression = native_force[i_b] @ normal
    excess = np.linalg.norm(native_force[i_b] - compression[:, None] * normal, axis=1) - frictions[i_b] * compression
    assert compression.min() >= -1e-12 and excess.max() <= 1e-12
    nodal = np.zeros((oracle.ndof // 3, 3))
    np.add.at(
        nodal,
        np.broadcast_to(surface_nodes, (900, 6)).reshape(-1),
        (basis[:, :, None] * reference[:, None]).reshape(-1, 3),
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
    annulus_inner_R=inner,
    annulus_outer_R=outer,
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
            "scope": "Legal near-capacity free-moment refinement diagnostic with finite-annulus witnesses, independent CPU geometry, vector fit, signed P2 scatter, full equilibrium, tensor and peak. Continuous footprint law and final budgets unchanged. Not production selection or physical convergence.",
        },
        indent=2,
    )
    + "\n"
)
