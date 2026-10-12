from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog
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
from genesis.engine.solvers.rigid.stress.recovery import kernel_reset
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
from genesis.utils.array_class import V_VEC, V, iter_data
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.wrench import WrenchPatch
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit
from tests.rigid.test_stress_traction import kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


def cpu_disk_geometry(anchor, triangle, normal, radius, fraction, angle, refined):
    """Independent FP64 coordinates, positive area/profile weights and signed P2 basis on the same physical disk."""
    tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
    second = np.cross(normal, tangent)
    gauss, weights = np.polynomial.legendre.leggauss(10)
    gauss, weights = (gauss + 1) / 2, weights / 2
    u, v = np.repeat(gauss, 10), np.tile(gauss, 10)
    if refined:
        half_width = min(np.pi / 2, np.sqrt(8 * (1 - fraction)))
        radial = np.array([0.0, max(0.0, fraction - 8 * (1 - fraction)), (1 + fraction) / 2, 1.0])
        angular = np.array([angle - np.pi, angle - half_width, angle + half_width, angle + np.pi])
        r = (radial[:-1, None] + np.diff(radial)[:, None] * u).reshape(3, 1, 100)
        theta = (angular[:-1, None] + np.diff(angular)[:, None] * v).reshape(1, 3, 100)
        area = radius**2 * r * np.diff(radial)[:, None, None] * np.diff(angular)[None, :, None]
    else:
        r, theta = u**2, 2 * np.pi * v
        area = 4 * np.pi * radius**2 * u**3
    positions = anchor + radius * r[..., None] * (
        np.cos(theta)[..., None] * tangent + np.sin(theta)[..., None] * second
    )
    area *= np.repeat(weights, 10) * np.tile(weights, 10)
    assert (area > 0).all()
    assert_allclose(area.sum(), np.pi * radius**2, rtol=2e-14, atol=0)
    profiles = (area * np.exp(-0.5 * (r / 0.45) ** 2) * (1 - r**2) ** 2).reshape(-1)
    positions = positions.reshape(-1, 3)
    local = np.linalg.lstsq((triangle[1:] - triangle[0]).T, (positions - triangle[0]).T, rcond=None)[0].T
    bary = np.column_stack((1 - local.sum(axis=1), local))
    assert (bary > 0).all()
    basis = np.column_stack((bary * (2 * bary - 1), 4 * bary[:, [0, 0, 1]] * bary[:, [1, 2, 2]]))
    return positions, profiles, basis


@pytest.mark.required
@pytest.mark.precision("64")
def test_legal_disk_cap_refinement_matches_cpu_through_release_restore_reset_and_physical_rejection():
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
    tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
    second = np.cross(normal, tangent)
    radius = 1e-5
    fractions = np.array([0.98, 0.99, 0.98, 0.99, 0.98, 0.99, 0.2])
    angles = np.array([0.0, 0.0, 0.7, 0.7, 1.8, 1.8, 0.0])
    frictions = np.array([0.5, 0.5, 0.85, 0.85, 1.2, 1.2, 0.6])
    directions = np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * second
    forces = normal + 0.4 * directions
    points = anchor - (fractions * radius / 0.4)[:, None] * normal
    centroids = anchor + fractions[:, None] * radius * directions
    witness_radius = 0.25 * radius * (1 - fractions)
    assert (np.linalg.norm(centroids - anchor, axis=1) + witness_radius < radius).all()
    assert (frictions > 0.4).all()
    # A positive-area uniform vector traction at these centroids matches the original point wrench and local cone.
    epsilon = float(np.finfo(gs.np_float).eps)
    assert_allclose(np.cross(centroids - points, forces), 0, rtol=0, atol=128 * epsilon * np.linalg.norm(anchor))
    options, n_envs = RigidStressOptions(mesh=mesh, method="direct"), len(fractions)
    model = StressModel(options)
    surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
    contacts, sources = create_contacts(1, n_envs, radius), create_sources(1, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 1, n_envs)
    tractions, info = create_surface_fan_tractions(surface.info, 1, n_envs), create_traction_info(surface.info)
    assert any(item.name == "stress.tractions.circle.focus" for item in iter_data(tractions, "stress.tractions"))
    assert_equal(tractions.sample_positions.shape, ())
    contacts.valid.fill(True)
    contacts.position.from_numpy(points[None])
    contacts.force.from_numpy(forces[None])
    contacts.friction.from_numpy(frictions[None])
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(np.broadcast_to(anchor, (1, n_envs, 3)).copy())
    samples, queries = create_traction_samples(tractions), V_VEC(3, dtype=gs.qd_float, shape=tractions.profile.shape)
    state, omega, error = (
        model.create_state(n_envs, "full"),
        V_VEC(3, dtype=gs.qd_float, shape=(n_envs,)),
        V(dtype=gs.qd_int, shape=(n_envs,)),
    )
    omega.fill(0)
    oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, copy=True))
    factor = splu(oracle.k[free][:, free].tocsc())
    surface_nodes = qd_to_numpy(model.info.surface_nodes, copy=True)[0]
    for phase in range(3):
        active = np.ones((1, n_envs), dtype=bool)
        active[0, 0] = phase != 1
        contacts.valid.from_numpy(active)
        kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
        kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
        kernel_surface_fan_profiles(
            contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon
        )
        assert not qd_to_numpy(tractions.circle.refined, copy=True).any()
        kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
        status = qd_to_numpy(tractions.status, transpose=True, copy=True)[:, 0]
        assert np.isin(status[:6][active[0, :6]], [3, 4]).all()
        assert status[6] == 0
        fast_state = {
            name: qd_to_numpy(value, transpose=True, copy=True)[6]
            for name, value in (
                ("dual", tractions.dual),
                ("profile", tractions.profile),
                ("indices", tractions.sample_indices),
                ("evaluations", tractions.evaluations),
                ("weight", tractions.weight_sum),
                ("scale", tractions.scale),
            )
        }
        if phase == 0:
            for i_b in range(6):
                positions, _, _ = cpu_disk_geometry(
                    anchor, triangle, normal, radius, fractions[i_b], angles[i_b], False
                )
                normalized = (positions - anchor) / radius
                result = linprog(
                    np.zeros(100),
                    A_eq=np.vstack((np.ones(100), normalized.T)),
                    b_eq=np.concatenate(([1.0], fractions[i_b] * directions[i_b])),
                    bounds=(0, None),
                    method="highs",
                )
                assert result.status == 2
        kernel_refine_failed_disks(contacts, sources, footprints, tractions, surface.info, info, epsilon)
        kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=2)
        assert_equal(qd_to_numpy(tractions.status, copy=True), 0)
        for name, value in (
            ("dual", tractions.dual),
            ("profile", tractions.profile),
            ("indices", tractions.sample_indices),
            ("evaluations", tractions.evaluations),
            ("weight", tractions.weight_sum),
            ("scale", tractions.scale),
        ):
            assert_equal(qd_to_numpy(value, transpose=True, copy=True)[6], fast_state[name])
        counts = qd_to_numpy(tractions.sample_count, transpose=True, copy=True)[:, 0]
        assert_equal(counts, np.where(active[0], np.where(np.arange(n_envs) < 6, 900, 100), 0))
        assert_equal(qd_to_numpy(footprints.samples, transpose=True, copy=True)[:, 0], counts)
        assert_equal(qd_to_numpy(contacts.radius, transpose=True)[:, 0], radius)
        assert_equal(qd_to_numpy(contacts.friction, transpose=True)[:, 0], frictions)
        if phase == 1:
            assert_equal(qd_to_numpy(tractions.circle.face, transpose=True)[0, 0], -1)
            assert_equal(qd_to_numpy(tractions.circle.focus, transpose=True)[0, 0], 0)
        kernel_traction_samples(samples, tractions, surface.info, info)
        kernel_traction_queries(queries, contacts, tractions, info)
        sample_positions = qd_to_numpy(samples.positions, transpose=True, copy=True)[:, :, 0]
        sample_basis = qd_to_numpy(samples.shape, transpose=True, copy=True)[:, :, 0]
        fitted = qd_to_numpy(queries, transpose=True, copy=True)[:, :, 0]
        profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)[:, :, 0]
        indices = qd_to_numpy(tractions.sample_indices, transpose=True, copy=True)[:, :, 0]
        nodal = np.zeros((n_envs, oracle.ndof // 3, 3))
        for i_b, count in enumerate(counts):
            if count > 0:
                positions, weights, basis = cpu_disk_geometry(
                    anchor, triangle, normal, radius, fractions[i_b], angles[i_b], i_b < 6
                )
                order = indices[i_b, :count] - 3 * surface.info.positions.shape[0]
                assert_equal(np.sort(order), np.arange(count))
                assert_allclose(
                    sample_positions[i_b, :count], positions[order], rtol=0, atol=128 * epsilon * np.linalg.norm(anchor)
                )
                roundoff = 512 * epsilon * (1 + np.linalg.norm(anchor) / radius) / (1 - fractions[i_b])
                assert_allclose(profiles[i_b, :count], weights[order], rtol=roundoff, atol=0)
                assert_allclose(sample_basis[i_b, :count], basis[order], rtol=2e-12, atol=2e-14)
                reference = cpu_resolved_wrench_fit(
                    positions,
                    weights,
                    np.broadcast_to(normal, positions.shape),
                    WrenchPatch(
                        anchor, forces[i_b], radius, frictions[i_b], np.cross(points[i_b] - anchor, forces[i_b])
                    ),
                )
                assert_allclose(fitted[i_b, :count], reference[order], rtol=1e-4, atol=1e-10)
                actual = fitted[i_b, :count]
                assert np.linalg.norm(actual.sum(axis=0) - forces[i_b]) <= 1e-8 * np.linalg.norm(forces[i_b])
                assert np.linalg.norm(
                    np.cross(sample_positions[i_b, :count] - points[i_b], actual).sum(axis=0)
                ) <= 1e-8 * radius * np.linalg.norm(forces[i_b])
                compression = actual @ normal
                assert compression.min() >= -1e-12
                assert (
                    np.linalg.norm(actual - compression[:, None] * normal, axis=1)
                    <= frictions[i_b] * compression + 1e-12
                ).all()
                np.add.at(
                    nodal[i_b],
                    np.broadcast_to(surface_nodes, (count, 6)).reshape(-1),
                    (basis[:, :, None] * reference[:, None]).reshape(-1, 3),
                )
        error.fill(0)
        kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, error, reset_force=True)
        assert_equal(qd_to_numpy(error, copy=True), 0)
        model.recover(omega, state)
        assert qd_to_numpy(state.valid, copy=True).all()
        for i_b in range(n_envs):
            rhs = oracle.balance(nodal[i_b].reshape(oracle.ndof, 1))
            reference = np.zeros_like(rhs)
            reference[free] = factor.solve(rhs[free])
            actual = qd_to_numpy(state.displacement, transpose=True, copy=True)[i_b].reshape(oracle.ndof, 1)
            assert np.linalg.norm(oracle.k @ actual - rhs) <= max(
                options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs)
            )
            tensor, vm = stress_field(oracle.glambda, oracle.elements, reference[:, 0], options.young, options.poisson)
            assert_allclose(
                qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[i_b], tensor, rtol=1e-4, atol=1e-3
            )
            assert_allclose(
                qd_to_numpy(state.von_mises, transpose=True, copy=True)[i_b, :, :, 0], vm, rtol=1e-4, atol=1e-3
            )
            assert_allclose(qd_to_numpy(state.peak, copy=True)[i_b], vm.max(), rtol=1e-4, atol=1e-3)
        if phase == 0:
            before = [
                qd_to_numpy(value, transpose=True, copy=True)
                for value in (state.peak, state.displacement, state.stress_tensor, state.von_mises)
            ]
            kernel_reset(np.array([2], dtype=np.int32), state)
            for value, saved in zip((state.peak, state.displacement, state.stress_tensor, state.von_mises), before):
                assert_equal(
                    qd_to_numpy(value, transpose=True, copy=True)[[0, 1, 3, 4, 5, 6]], saved[[0, 1, 3, 4, 5, 6]]
                )
            assert_equal(qd_to_numpy(state.peak, copy=True)[2], 0)
    incompatible = points.copy()
    incompatible[0] = anchor - (1.01 * radius / 0.4) * normal
    contacts.position.from_numpy(incompatible[None])
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    kernel_refine_failed_disks(contacts, sources, footprints, tractions, surface.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128, retry_mode=2)
    status = qd_to_numpy(tractions.status, transpose=True, copy=True)[:, 0]
    assert_equal(status[0], StressTractionStatus.CONTINUOUS_WRENCH_INFEASIBLE)
    assert_equal(status[1:], 0)
    assert not qd_to_numpy(tractions.circle.refined, transpose=True)[0, 0]
    ray = qd_to_numpy(tractions.certificate, transpose=True, copy=True)[0, 0]
    # The exact entire-disk polar bound uses normal compression and the analytic support function of a circle.
    upper = ray[:3] @ normal + np.linalg.norm(np.cross(normal, ray[3:]))
    upper += frictions[0] * (np.linalg.norm(ray[:3] - (ray[:3] @ normal) * normal) + abs(ray[3:] @ normal))
    assert upper < -1e-4
    target = np.concatenate((forces[0], np.cross(incompatible[0] - anchor, forces[0]) / radius))
    assert target @ ray > 1e-4
    assert_equal(qd_to_numpy(contacts.position, transpose=True, copy=True)[0, 0], incompatible[0])
    assert_equal(qd_to_numpy(contacts.radius, transpose=True)[:, 0], radius)
    assert_equal(qd_to_numpy(contacts.friction, transpose=True)[:, 0], frictions)
