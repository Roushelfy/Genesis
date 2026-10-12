from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize, root
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.recovery import kernel_reset
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
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
from research.rigid_stress.wrench import WrenchFit, WrenchPatch, project_coulomb
from tests.rigid.test_stress_footprint import cpu_intrinsic_distances
from tests.rigid.test_stress_traction import kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


def cpu_surface_fan_geometry(anchor, vertices, faces, quadrature, coordinate_power=2):
    """Independent nearest points, finite triangle partitions and signed P2 bases."""
    triangles = vertices[faces]
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    uv = np.einsum("fij,fj->fi", np.linalg.pinv(edges), anchor - triangles[:, 0])
    inside = (uv >= 0).all(axis=1) & (uv.sum(axis=1) <= 1)
    center_bary = np.column_stack((1 - uv.sum(axis=1), uv))
    first, second = triangles[:, [0, 1, 2]], triangles[:, [1, 2, 0]]
    boundary_edges = second - first
    fraction = np.einsum("fei,fei->fe", anchor - first, boundary_edges)
    fraction /= np.einsum("fei,fei->fe", boundary_edges, boundary_edges)
    fraction = np.clip(fraction, 0, 1)
    boundary_points = first + fraction[..., None] * boundary_edges
    closest = np.argmin(np.linalg.norm(boundary_points - anchor, axis=2), axis=1)
    boundary_bary = np.zeros((len(faces), 3))
    boundary_bary[np.arange(len(faces)), closest] = 1 - fraction[np.arange(len(faces)), closest]
    boundary_bary[np.arange(len(faces)), (closest + 1) % 3] = fraction[np.arange(len(faces)), closest]
    center_bary[~inside] = boundary_bary[~inside]
    gauss, weights = np.polynomial.legendre.leggauss(quadrature)
    gauss, weights = (gauss + 1) / 2, weights / 2
    weights *= coordinate_power * gauss ** (coordinate_power - 1)
    gauss **= coordinate_power
    u, v = np.meshgrid(gauss, gauss, indexing="ij")
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)
    bary = (1 - u) * center_bary[:, None, None] + u * (1 - v) * np.eye(3)[None, :, None]
    bary += u * v * np.eye(3)[[1, 2, 0]][None, :, None]
    points = np.einsum("fkqa,fai->fkqi", bary, triangles)
    area = np.linalg.norm(np.cross(boundary_edges[:, 0], -boundary_edges[:, 2]), axis=1)
    area = area[:, None] * center_bary[:, [2, 0, 1]]
    area_weights = area[..., None] * (u[:, 0] * (weights[:, None] * weights[None]).reshape(-1))
    points = points.reshape(-1, 3)
    indices = np.repeat(np.arange(len(faces)), 3 * quadrature**2)
    bary = bary.reshape(-1, 3)
    basis = np.column_stack((bary * (2 * bary - 1), 4 * bary[:, [0, 0, 1]] * bary[:, [1, 2, 2]]))
    return points, area_weights.reshape(-1), indices, basis


def cpu_dual_energy_gradient(multiplier, fitter):
    vectors = fitter.patch.force_n - np.einsum("qij,i->qj", fitter.operator, multiplier)
    projected, _ = project_coulomb(vectors, fitter.normals, fitter.patch.friction)
    energy = 0.5 * np.einsum("q,qi,qi->", fitter.weights, projected, projected)
    energy += fitter.target @ multiplier
    return energy, -fitter.scale * fitter.residual(multiplier)


def cpu_dual_hessian(multiplier, fitter):
    return -fitter.scale * fitter.jacobian(multiplier)


def cpu_resolved_wrench_fit(positions, profiles, normals, patch):
    """FP64 convex-dual trust-region oracle with strict independent wrench checks.

    Residual least-squares can have a stationary point with nonzero wrench
    error when local cones remove curvature. The convex dual has the actual
    wrench mismatch as its gradient; optimize that objective instead.
    No native dual, Hessian regularization or fitted forces enter this solve.
    """
    scale = np.linalg.norm(patch.force_n)
    normalized = WrenchPatch(
        patch.center_m, patch.force_n / scale, patch.radius_m, patch.friction, patch.moment_nm / scale
    )
    fitter = WrenchFit(positions, profiles, normals, normalized)
    gram = np.einsum("qij,qkj,q->ik", fitter.operator, fitter.operator, fitter.weights)
    initial = np.einsum("qij,j,q->i", fitter.operator, normalized.force_n, fitter.weights)
    initial = np.linalg.solve(gram, initial - fitter.target)
    result = minimize(
        cpu_dual_energy_gradient,
        initial,
        args=(fitter,),
        jac=True,
        hess=cpu_dual_hessian,
        method="trust-exact",
        options={"gtol": 1e-11, "maxiter": 1024, "initial_trust_radius": 1.0, "max_trust_radius": 1e9},
    )
    multiplier = result.x
    if np.linalg.norm(fitter.residual(multiplier)) * fitter.scale > 1e-11:
        polished = root(
            fitter.residual,
            multiplier,
            jac=fitter.jacobian,
            method="hybr",
            options={"xtol": 1e-12, "maxfev": 4096},
        )
        if np.linalg.norm(fitter.residual(polished.x)) < np.linalg.norm(fitter.residual(multiplier)):
            multiplier = polished.x
    vectors = normalized.force_n - np.einsum("qij,i->qj", fitter.operator, multiplier)
    projected, _ = project_coulomb(vectors, normals, patch.friction)
    force = scale * fitter.weights[:, None] * projected
    mismatch = np.einsum("qij,qj->i", fitter.operator, force) - scale * fitter.target
    assert np.linalg.norm(mismatch[:3]) / scale <= 1e-8
    assert np.linalg.norm(mismatch[3:]) / scale <= 1e-8
    return force


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize(
    "tick,coordinate_power",
    ((147, 1), (427, 1), (454, 1), (147, 2), (427, 2), (454, 2), (517, 2), (594, 2), (1076, 2)),
)
def test_saved_edge_contact_uses_neighbor_finite_area_and_cpu_oracle(tick, coordinate_power):
    with np.load(Path(__file__).with_name("assets") / f"nonconvex_mesh64_tick{tick}.npz") as saved:
        point, anchor, force = (saved[key] for key in ("position", "anchor", "force"))
        radius, friction = float(saved["radius"]), float(saved["friction"])
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    options = RigidStressOptions(mesh=mesh, method="direct")
    model, n_envs = StressModel(options), 3
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 8)
    info = create_traction_info(surface.info)
    contacts, sources = create_contacts(1, n_envs, radius), create_sources(1, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 1, n_envs)
    tractions = create_surface_fan_tractions(surface.info, 1, n_envs, coordinate_power=coordinate_power)
    contacts.position.from_numpy(np.broadcast_to(point, (1, n_envs, 3)).copy())
    contacts.force.from_numpy(np.broadcast_to(force, (1, n_envs, 3)).copy())
    contacts.friction.from_numpy(np.array([[friction, 0.85, 1.2]]))
    contacts.valid.fill(True)
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(np.broadcast_to(anchor, (1, n_envs, 3)).copy())
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    quad_force = V_VEC(3, dtype=gs.qd_float, shape=tractions.profile.shape)
    kernel_traction_queries(quad_force, contacts, tractions, info)
    count = qd_to_numpy(tractions.sample_count, transpose=True)[:, 0]
    indices = qd_to_numpy(tractions.sample_indices, transpose=True)[:, :, 0]
    sample_positions = qd_to_numpy(samples.positions, transpose=True)[:, :, 0]
    sample_faces = qd_to_numpy(samples.faces, transpose=True)[:, :, 0]
    sample_normals = qd_to_numpy(samples.normals, transpose=True)[:, :, 0]
    sample_basis = qd_to_numpy(samples.shape, transpose=True)[:, :, 0]
    actual_force = qd_to_numpy(quad_force, transpose=True)[:, :, 0]
    profiles = qd_to_numpy(tractions.profile, transpose=True)[:, :, 0]
    anchor_face = qd_to_numpy(contacts.anchor_face, transpose=True)[:, 0]
    with np.load(mesh) as asset:
        vertices, tets, faces = (asset[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    positions, weights, cpu_faces, basis = cpu_surface_fan_geometry(anchor, vertices, faces, 10, coordinate_power)
    candidates = (weights > 0) & (np.linalg.norm(positions - anchor, axis=1) < radius)
    cpu_distances = cpu_intrinsic_distances(
        anchor[None],
        anchor_face[:1],
        positions[candidates],
        cpu_faces[candidates],
        vertices,
        faces,
        qd_to_numpy(paths.info.positions, transpose=True),
        qd_to_numpy(paths.info.face_ports, transpose=True).T,
        qd_to_numpy(paths.info.starts, transpose=True),
        qd_to_numpy(paths.info.neighbors, transpose=True),
    )[0]
    cpu_profiles = np.zeros(len(weights))
    cpu_profiles[candidates] = weights[candidates] * np.exp(-0.5 * (cpu_distances / (0.45 * radius)) ** 2)
    cpu_profiles[candidates] *= np.maximum(1 - (cpu_distances / radius) ** 2, 0) ** 2
    # Independent FP64 fitting uses the CPU profile and material normals;
    # it receives neither the native dual nor its fitted sample forces.
    normals = -qd_to_numpy(surface.info.face_normal, transpose=True)[cpu_faces]
    selected = cpu_profiles > 0
    nodal_reference = np.zeros((n_envs, model.info.vertices.shape[0], 3))
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
    moment = np.cross(point - anchor, force)
    for i_b, mu in enumerate([friction, 0.85, 1.2]):
        native_indices = indices[i_b, : count[i_b]]
        assert_allclose(sample_positions[i_b, : count[i_b]], positions[native_indices], rtol=2e-12, atol=2e-17)
        assert_equal(sample_faces[i_b, : count[i_b]], cpu_faces[native_indices])
        assert_allclose(sample_basis[i_b, : count[i_b]], basis[native_indices], rtol=2e-12, atol=2e-14)
        assert_allclose(profiles[i_b, : count[i_b]], cpu_profiles[native_indices], rtol=2e-11, atol=2e-20)
        patch = WrenchPatch(anchor, force, radius, mu, moment)
        cpu_force = cpu_resolved_wrench_fit(positions[selected], cpu_profiles[selected], normals[selected], patch)
        np.add.at(
            nodal_reference[i_b],
            surface_nodes[cpu_faces[selected]].reshape(-1),
            (basis[selected, :, None] * cpu_force[:, None]).reshape(-1, 3),
        )
        native_force = actual_force[i_b, : count[i_b]]
        assert np.linalg.norm(native_force.sum(axis=0) - force) / np.linalg.norm(force) <= 1e-8
        native_moment = np.cross(sample_positions[i_b, : count[i_b]] - point, native_force).sum(axis=0)
        assert np.linalg.norm(native_moment) / (radius * np.linalg.norm(force)) <= 1e-8
        compression = np.einsum("qi,qi->q", native_force, sample_normals[i_b, : count[i_b]])
        assert (compression >= -1e-12).all()
        assert (
            np.linalg.norm(native_force - compression[:, None] * sample_normals[i_b, : count[i_b]], axis=1)
            <= mu * compression + 1e-12
        ).all()
    state = model.create_state(n_envs, "full")
    errno, omega = V(dtype=gs.qd_int, shape=(n_envs,)), V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    nodal = qd_to_numpy(state.force, transpose=True)
    assert_allclose(nodal, nodal_reference, rtol=1e-4, atol=1e-10)
    xyz = qd_to_numpy(model.info.vertices, transpose=True)
    assert (np.linalg.norm(nodal.sum(axis=1) - force, axis=1) / np.linalg.norm(force) <= 1e-8).all()
    assert (
        np.linalg.norm(np.cross(xyz[None] - point, nodal).sum(axis=1), axis=1) / (radius * np.linalg.norm(force))
        <= 1e-8
    ).all()
    oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    rhs = oracle.balance(np.moveaxis(nodal_reference, 0, -1).reshape(oracle.ndof, n_envs))
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    reference = np.zeros_like(rhs)
    reference[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    assert (residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))).all()
    actual_tensor = qd_to_numpy(state.stress_tensor, transpose=True)
    actual_vm = qd_to_numpy(state.von_mises, transpose=True)
    peaks = []
    for i_b in range(n_envs):
        expected_tensor, expected_vm = stress_field(
            oracle.glambda, oracle.elements, reference[:, i_b], options.young, options.poisson
        )
        assert_allclose(actual_tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(actual_vm[i_b, :, :, 0], expected_vm, rtol=1e-4, atol=1e-3)
        peaks.append(expected_vm.max())
    assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)


@pytest.mark.required
@pytest.mark.precision("64")
def test_legal_small_intrinsic_disks_recover_original_wrench_and_full_field_through_release_reset():
    with np.load(Path(__file__).with_name("assets") / "nonconvex_mesh64_small_disk.npz") as saved:
        anchor, point, force = (saved[key] for key in ("anchor", "point", "force"))
        original_radius, original_samples = float(saved["radius"]), saved["samples"]
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    with np.load(mesh) as asset:
        vertices, tets, faces = (asset[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    triangle = vertices[faces[0]]
    owners = np.flatnonzero(np.isin(tets, faces[0]).sum(axis=1) == 3)
    opposite = vertices[tets[owners[0]][~np.isin(tets[owners[0]], faces[0])][0]]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    normal *= np.sign(normal @ (opposite - triangle[0]))
    tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
    second = np.cross(normal, tangent)
    ratio = np.linalg.norm(force - normal * (force @ normal)) / (force @ normal)
    centroid_coordinate = -ratio * ((point - anchor) @ normal)
    patch_radius = 0.25 * (original_radius - abs(centroid_coordinate))
    # A uniform vector traction on this positive-area subdisk realizes the
    # original point wrench. Its pressure centroid lies outside the old
    # triangle-fan sample hull, so iterations alone cannot recover this input.
    assert patch_radius > 0 and abs(centroid_coordinate) + patch_radius < original_radius
    assert centroid_coordinate < ((original_samples - anchor) @ tangent).min() - 4e-6
    assert_allclose(np.cross(anchor + centroid_coordinate * tangent - point, force), 0, rtol=0, atol=1e-18)
    edge_vectors = np.roll(triangle, -1, axis=0) - triangle
    edge_clearance = np.linalg.norm(np.cross(anchor - triangle, edge_vectors), axis=1)
    edge_clearance /= np.linalg.norm(edge_vectors, axis=1)
    radii, frictions = original_radius * np.array([1.0, 2.0, 0.5, 0.1]), np.array([0.5, 0.6, 0.85, 1.2])
    assert (edge_clearance.min() > radii).all() and (ratio < frictions).all()
    points = anchor + (radii / original_radius)[:, None] * (point - anchor)
    options, n_envs = RigidStressOptions(mesh=mesh, method="direct"), len(radii)
    model = StressModel(options)
    surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
    contacts, sources = create_contacts(1, n_envs, original_radius), create_sources(1, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 1, n_envs)
    tractions, info = create_surface_fan_tractions(surface.info, 1, n_envs), create_traction_info(surface.info)
    assert any(item.name == "stress.tractions.circle.face" for item in iter_data(tractions, "stress.tractions"))
    contacts.valid.fill(True)
    contacts.position.from_numpy(points[None])
    contacts.force.from_numpy(np.broadcast_to(force, (1, n_envs, 3)).copy())
    contacts.radius.from_numpy(radii[None])
    contacts.friction.from_numpy(frictions[None])
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(np.broadcast_to(anchor, (1, n_envs, 3)).copy())
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    assert_equal(qd_to_numpy(sources.status, transpose=True), 0)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    assert_equal(qd_to_numpy(tractions.circle.face, transpose=True), 0)
    assert_equal(qd_to_numpy(tractions.sample_count, transpose=True), 100)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    assert_equal(qd_to_numpy(contacts.radius, transpose=True)[:, 0], radii)
    assert_equal(qd_to_numpy(contacts.friction, transpose=True)[:, 0], frictions)
    samples, quad_force = create_traction_samples(tractions), V_VEC(3, dtype=gs.qd_float, shape=tractions.profile.shape)
    kernel_traction_samples(samples, tractions, surface.info, info)
    kernel_traction_queries(quad_force, contacts, tractions, info)
    sample_positions = qd_to_numpy(samples.positions, transpose=True, copy=True)[:, :100, 0]
    sample_normals = qd_to_numpy(samples.normals, transpose=True, copy=True)[:, :100, 0]
    sample_basis = qd_to_numpy(samples.shape, transpose=True, copy=True)[:, :100, 0]
    sample_indices = qd_to_numpy(tractions.sample_indices, transpose=True, copy=True)[:, :100, 0]
    profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)[:, :100, 0]
    actual_force = qd_to_numpy(quad_force, transpose=True, copy=True)[:, :100, 0]
    gauss, weights = np.polynomial.legendre.leggauss(10)
    gauss, weights = (gauss + 1) / 2, weights / 2
    u, v = np.repeat(gauss, 10), np.tile(gauss, 10)
    reference_positions = anchor + radii[:, None, None] * u[None, :, None] ** 2 * (
        np.cos(2 * np.pi * v)[:, None] * tangent + np.sin(2 * np.pi * v)[:, None] * second
    )
    reference_profiles = 4 * np.pi * radii[:, None] ** 2 * u**3 * np.repeat(weights, 10) * np.tile(weights, 10)
    reference_profiles *= np.exp(-0.5 * (u**2 / 0.45) ** 2) * (1 - u**4) ** 2
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T[0]
    nodal_reference = np.zeros((n_envs, model.info.vertices.shape[0], 3))
    for i_b, mu in enumerate(frictions):
        order = sample_indices[i_b] - 3 * surface.info.positions.shape[0]
        assert_equal(np.sort(order), np.arange(100))
        positions = reference_positions[i_b]
        local = np.linalg.lstsq((triangle[1:] - triangle[0]).T, (positions - triangle[0]).T, rcond=None)[0].T
        bary = np.column_stack((1 - local.sum(axis=1), local))
        basis = np.column_stack((bary * (2 * bary - 1), 4 * bary[:, [0, 0, 1]] * bary[:, [1, 2, 2]]))
        assert (bary > 0).all()
        assert_allclose(sample_positions[i_b], positions[order], rtol=0, atol=128 * epsilon * np.linalg.norm(anchor))
        assert_allclose(sample_basis[i_b], basis[order], rtol=2e-12, atol=2e-14)
        assert_allclose(profiles[i_b], reference_profiles[i_b, order], rtol=2e-12, atol=0)
        assert_allclose(sample_normals[i_b], np.broadcast_to(normal, (100, 3)), rtol=0, atol=64 * epsilon)
        patch = WrenchPatch(anchor, force, radii[i_b], mu, np.cross(points[i_b] - anchor, force))
        cpu_force = cpu_resolved_wrench_fit(
            positions, reference_profiles[i_b], np.broadcast_to(normal, (100, 3)), patch
        )
        assert_allclose(actual_force[i_b], cpu_force[order], rtol=1e-4, atol=1e-10)
        np.add.at(
            nodal_reference[i_b],
            np.broadcast_to(surface_nodes, (100, 6)).reshape(-1),
            (basis[:, :, None] * cpu_force[:, None]).reshape(-1, 3),
        )
        assert np.linalg.norm(actual_force[i_b].sum(axis=0) - force) / np.linalg.norm(force) <= 1e-8
        moment = np.cross(sample_positions[i_b] - points[i_b], actual_force[i_b]).sum(axis=0)
        assert np.linalg.norm(moment) / (radii[i_b] * np.linalg.norm(force)) <= 1e-8
        compression = np.einsum("qi,qi->q", actual_force[i_b], sample_normals[i_b])
        assert (compression >= -1e-12).all()
        assert (
            np.linalg.norm(actual_force[i_b] - compression[:, None] * sample_normals[i_b], axis=1)
            <= mu * compression + 1e-12
        ).all()
    state = model.create_state(n_envs, "full")
    errno, omega = V(dtype=gs.qd_int, shape=(n_envs,)), V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    assert_allclose(qd_to_numpy(state.force, transpose=True), nodal_reference, rtol=1e-4, atol=1e-10)
    xyz = qd_to_numpy(model.info.vertices, transpose=True)
    moments = np.cross(xyz[None] - points[:, None], qd_to_numpy(state.force, transpose=True)).sum(axis=1)
    assert (np.linalg.norm(moments, axis=1) / (radii * np.linalg.norm(force)) <= 1e-8).all()
    oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    rhs = oracle.balance(np.moveaxis(nodal_reference, 0, -1).reshape(oracle.ndof, n_envs))
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    reference = np.zeros_like(rhs)
    reference[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    assert (residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))).all()
    tensors, vms = qd_to_numpy(state.stress_tensor, transpose=True), qd_to_numpy(state.von_mises, transpose=True)
    peaks = []
    for i_b in range(n_envs):
        tensor, vm = stress_field(oracle.glambda, oracle.elements, reference[:, i_b], options.young, options.poisson)
        assert_allclose(tensors[i_b], tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(vms[i_b, :, :, 0], vm, rtol=1e-4, atol=1e-3)
        peaks.append(vm.max())
    assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)
    saved_tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
    kernel_reset(np.array([0], dtype=gs.np_int), state)
    assert_equal(qd_to_numpy(state.peak, transpose=True)[0], 0)
    assert_equal(qd_to_numpy(state.stress_tensor, transpose=True)[1:], saved_tensor[1:])
    contacts.valid.from_numpy(np.array([[False, True, True, True]]))
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    assert_equal(qd_to_numpy(tractions.circle.face, transpose=True)[:, 0], [-1, 0, 0, 0])
    assert_equal(qd_to_numpy(tractions.sample_count, transpose=True)[:, 0], [0, 100, 100, 100])
    errno.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    model.recover(omega, state)
    assert_equal(qd_to_numpy(state.peak, transpose=True)[0], 0)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    released_rhs = rhs.copy()
    released_rhs[:, 0] = 0
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - released_rhs, axis=0)
    assert (
        residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(released_rhs, axis=0))
    ).all()
    tensors, vms = qd_to_numpy(state.stress_tensor, transpose=True), qd_to_numpy(state.von_mises, transpose=True)
    assert_equal(tensors[0], 0)
    for i_b in range(1, n_envs):
        tensor, vm = stress_field(oracle.glambda, oracle.elements, reference[:, i_b], options.young, options.poisson)
        assert_allclose(tensors[i_b], tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(vms[i_b, :, :, 0], vm, rtol=1e-4, atol=1e-3)
    assert_allclose(qd_to_numpy(state.peak, transpose=True)[1:], peaks[1:], rtol=1e-4, atol=1e-3)
    # A boundary crossing must invalidate the whole-circle chart, even on the
    # same face; restoring the contact rebuilds it from current geometry.
    moved = np.broadcast_to(anchor, (1, n_envs, 3)).copy()
    moved[0, 0] = triangle[:2].mean(axis=0)
    contacts.valid.fill(True)
    contacts.center.from_numpy(moved)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    assert_equal(qd_to_numpy(tractions.circle.face, transpose=True)[:, 0], [-1, 0, 0, 0])
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    assert_equal(qd_to_numpy(tractions.circle.face, transpose=True), 0)
    errno.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    model.recover(omega, state)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)
