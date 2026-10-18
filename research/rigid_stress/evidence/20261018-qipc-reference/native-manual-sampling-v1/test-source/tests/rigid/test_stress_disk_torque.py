from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_refine_failed_disks,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionStatus,
    create_traction_info,
    create_traction_samples,
    kernel_fit_tractions,
    kernel_scatter_tractions,
    kernel_traction_samples,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import WrenchPatch
from tests.rigid.test_stress_disk_refinement import cpu_disk_geometry
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit
from tests.rigid.test_stress_traction import kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("is_geometry_cached", (False, True))
def test_legal_finite_annular_torques_recover_both_signs_and_actual_mu_with_independent_cpu_fields(is_geometry_cached):
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    with np.load(mesh) as saved:
        vertices, tets, faces = (saved[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    triangle = vertices[faces[0]]
    anchor = triangle.mean(axis=0)
    owner = np.flatnonzero(np.isin(tets, faces[0]).sum(axis=1) == 3)[0]
    opposite = vertices[tets[owner][~np.isin(tets[owner], faces[0])][0]]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    normal *= np.sign(normal @ (opposite - triangle[0]))
    radius = 1e-5
    fractions = np.array([0.99, 0.999, 0.99, 0.999, 0.99, 0.999, 0.1])
    frictions = np.array([0.5, 0.5, 0.85, 0.85, 1.2, 1.2, 0.6])
    signs = np.array([1, 1, -1, -1, 1, -1, 1])
    moments = (radius * fractions * frictions * signs)[:, None] * normal
    inner, outer = fractions + 0.5 * (1 - fractions), fractions + 0.75 * (1 - fractions)
    mean_radius = 2 / 3 * (outer**2 + outer * inner + inner**2) / (outer + inner)
    swirl_ratio = frictions * fractions / mean_radius
    assert ((inner < outer) & (outer < 1) & (swirl_ratio < frictions)).all()
    # Uniform positive compression and circumferential traction on a finite annulus realize each signed free moment.
    options, n_envs = RigidStressOptions(mesh=mesh, method="direct"), len(fractions)
    model = StressModel(options)
    surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
    contacts, sources = create_contacts(1, n_envs, radius), create_sources(1, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 1, n_envs)
    tractions, info = (
        create_surface_fan_tractions(surface.info, 1, n_envs, cached_fan_positions=is_geometry_cached),
        create_traction_info(surface.info),
    )
    assert len(tractions.sample_barycentric.shape) == (4 if is_geometry_cached else 0)
    contacts.valid.fill(True)
    contacts.position.from_numpy(np.broadcast_to(anchor, (1, n_envs, 3)).copy())
    contacts.force.from_numpy(np.broadcast_to(normal, (1, n_envs, 3)).copy())
    contacts.friction.from_numpy(frictions[None])
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(np.broadcast_to(anchor, (1, n_envs, 3)).copy())
    sources.free_moment.from_numpy(moments[None])
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    baseline_status = qd_to_numpy(tractions.status, transpose=True, copy=True)[:, 0]
    assert np.isin(baseline_status[:6], [3, 4]).all() and baseline_status[6] == 0
    base_positions, _, _ = cpu_disk_geometry(anchor, triangle, normal, radius, 0, 0, False)
    # Every discrete cone obeys |M_n| <= mu * max(r_q) * total normal compression, regardless of fit iterations.
    assert (fractions[:6] > np.linalg.norm(base_positions - anchor, axis=1).max() / radius).all()
    fast_dual = qd_to_numpy(tractions.dual, transpose=True, copy=True)[6].copy()
    fast_evaluations = qd_to_numpy(tractions.evaluations, transpose=True, copy=True)[6].copy()
    kernel_refine_failed_disks(contacts, sources, footprints, tractions, surface.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=2)
    assert_equal(qd_to_numpy(tractions.status, copy=True), 0)
    assert_equal(qd_to_numpy(tractions.dual, transpose=True, copy=True)[6], fast_dual)
    assert_equal(qd_to_numpy(tractions.evaluations, transpose=True, copy=True)[6], fast_evaluations)
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    query = V_VEC(3, dtype=gs.qd_float, shape=tractions.profile.shape)
    kernel_traction_queries(query, contacts, tractions, info)
    positions_native = qd_to_numpy(samples.positions, transpose=True, copy=True)[:, :, 0]
    basis_native = qd_to_numpy(samples.shape, transpose=True, copy=True)[:, :, 0]
    fitted = qd_to_numpy(query, transpose=True, copy=True)[:, :, 0]
    profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)[:, :, 0]
    indices = qd_to_numpy(tractions.sample_indices, transpose=True, copy=True)[:, :, 0]
    counts = qd_to_numpy(tractions.sample_count, transpose=True, copy=True)[:, 0]
    assert_equal(counts, [900] * 6 + [100])
    state, omega, error = (
        model.create_state(n_envs, "full"),
        V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        V(dtype=gs.qd_int, shape=(n_envs,)),
    )
    omega.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, error, reset_force=True)
    assert_equal(qd_to_numpy(error, copy=True), 0)
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, copy=True).all()
    oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, copy=True))
    factor = splu(oracle.k[free][:, free].tocsc())
    nodes = qd_to_numpy(model.info.surface_nodes, copy=True)[0]
    for i_b, count in enumerate(counts):
        positions, weights, basis = cpu_disk_geometry(anchor, triangle, normal, radius, fractions[i_b], 0, i_b < 6)
        order = indices[i_b, :count] - 3 * surface.info.positions.shape[0]
        assert_equal(np.sort(order), np.arange(count))
        assert_allclose(
            positions_native[i_b, :count], positions[order], rtol=0, atol=128 * epsilon * np.linalg.norm(anchor)
        )
        roundoff = 512 * epsilon * (1 + np.linalg.norm(anchor) / radius) / (1 - fractions[i_b])
        assert_allclose(profiles[i_b, :count], weights[order], rtol=roundoff, atol=0)
        assert_allclose(basis_native[i_b, :count], basis[order], rtol=2e-12, atol=2e-14)
        reference = cpu_resolved_wrench_fit(
            positions,
            weights,
            np.broadcast_to(normal, positions.shape),
            WrenchPatch(anchor, normal, radius, frictions[i_b], moments[i_b]),
        )
        actual = fitted[i_b, :count]
        assert_allclose(actual, reference[order], rtol=1e-4, atol=1e-10)
        assert np.linalg.norm(actual.sum(axis=0) - normal) <= 1e-8
        assert (
            np.linalg.norm(np.cross(positions_native[i_b, :count] - anchor, actual).sum(axis=0) - moments[i_b])
            <= 1e-8 * radius
        )
        compression = actual @ normal
        assert compression.min() >= -1e-12
        assert (
            np.linalg.norm(actual - compression[:, None] * normal, axis=1) <= frictions[i_b] * compression + 1e-12
        ).all()
        nodal = np.zeros((oracle.ndof // 3, 3))
        np.add.at(
            nodal,
            np.broadcast_to(nodes, (count, 6)).reshape(-1),
            (basis[:, :, None] * reference[:, None]).reshape(-1, 3),
        )
        rhs = oracle.balance(nodal.reshape(oracle.ndof, 1))
        displacement = np.zeros_like(rhs)
        displacement[free] = factor.solve(rhs[free])
        actual_displacement = qd_to_numpy(state.displacement, transpose=True, copy=True)[i_b].reshape(oracle.ndof, 1)
        assert np.linalg.norm(oracle.k @ actual_displacement - rhs) <= max(
            options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs)
        )
        tensor, vm = stress_field(oracle.glambda, oracle.elements, displacement[:, 0], options.young, options.poisson)
        assert_allclose(qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[i_b], tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(qd_to_numpy(state.von_mises, transpose=True, copy=True)[i_b, :, :, 0], vm, rtol=1e-4, atol=1e-3)
        assert_allclose(qd_to_numpy(state.peak, copy=True)[i_b], vm.max(), rtol=1e-4, atol=1e-3)
    moments[0] = 1.01 * frictions[0] * radius * normal
    sources.free_moment.from_numpy(moments[None])
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    kernel_refine_failed_disks(contacts, sources, footprints, tractions, surface.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=2)
    status = qd_to_numpy(tractions.status, transpose=True, copy=True)[:, 0]
    assert_equal(status[0], StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE)
    assert_equal(status[1:], 0)
    ray = qd_to_numpy(tractions.certificate, transpose=True, copy=True)[0, 0]
    upper = ray[:3] @ normal + np.linalg.norm(np.cross(normal, ray[3:]))
    upper += frictions[0] * (np.linalg.norm(ray[:3] - (ray[:3] @ normal) * normal) + abs(ray[3:] @ normal))
    assert upper < -1e-4
    assert np.concatenate((normal, moments[0] / radius)) @ ray > 1e-4
    assert_equal(qd_to_numpy(sources.free_moment, transpose=True, copy=True)[:, 0], moments)
    assert_equal(qd_to_numpy(contacts.friction, transpose=True)[:, 0], frictions)
    assert_equal(qd_to_numpy(contacts.radius, transpose=True)[:, 0], radius)
