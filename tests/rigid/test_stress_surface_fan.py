from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
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
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
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
from research.rigid_stress.wrench import WrenchFit, WrenchPatch, project_coulomb
from tests.rigid.test_stress_footprint import cpu_intrinsic_distances
from tests.rigid.test_stress_traction import kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


def cpu_surface_fan_geometry(anchor, vertices, faces, quadrature):
    """Independent nearest points, finite triangle partitions and signed P2 bases."""
    triangles = vertices[faces]
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    uv = np.einsum("fij,fj->fi", np.linalg.pinv(edges), anchor - triangles[:, 0])
    inside = (uv >= 0).all(axis=1) & (uv.sum(axis=1) <= 1)
    centers = triangles[:, 0] + np.einsum("fij,fj->fi", edges, uv)
    first, second = triangles[:, [0, 1, 2]], triangles[:, [1, 2, 0]]
    boundary_edges = second - first
    fraction = np.einsum("fei,fei->fe", anchor - first, boundary_edges)
    fraction /= np.einsum("fei,fei->fe", boundary_edges, boundary_edges)
    boundary_points = first + np.clip(fraction, 0, 1)[..., None] * boundary_edges
    closest = np.argmin(np.linalg.norm(boundary_points - anchor, axis=2), axis=1)
    centers[~inside] = boundary_points[np.arange(len(faces)), closest][~inside]
    gauss, weights = np.polynomial.legendre.leggauss(quadrature)
    gauss, weights = (gauss + 1) / 2, weights / 2
    u, v = np.meshgrid(gauss, gauss, indexing="ij")
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)
    points = (1 - u) * centers[:, None, None] + u * (1 - v) * first[:, :, None] + u * v * second[:, :, None]
    area = np.linalg.norm(np.cross(first - centers[:, None], second - centers[:, None]), axis=2)
    area_weights = area[..., None] * (u[:, 0] * (weights[:, None] * weights[None]).reshape(-1))
    points = points.reshape(-1, 3)
    indices = np.repeat(np.arange(len(faces)), 3 * quadrature**2)
    uv = np.einsum("qij,qj->qi", np.linalg.pinv(edges)[indices], points - triangles[indices, 0])
    bary = np.column_stack((1 - uv.sum(axis=1), uv))
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
    vectors = normalized.force_n - np.einsum("qij,i->qj", fitter.operator, result.x)
    projected, _ = project_coulomb(vectors, normals, patch.friction)
    force = scale * fitter.weights[:, None] * projected
    mismatch = np.einsum("qij,qj->i", fitter.operator, force) - scale * fitter.target
    assert np.linalg.norm(mismatch[:3]) / scale <= 1e-8
    assert np.linalg.norm(mismatch[3:]) / scale <= 1e-8
    return force


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("tick", (147, 427, 454))
def test_saved_edge_contact_uses_neighbor_finite_area_and_cpu_oracle(tick):
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
    tractions = create_surface_fan_tractions(surface.info, 1, n_envs)
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
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info)
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
    positions, weights, cpu_faces, basis = cpu_surface_fan_geometry(anchor, vertices, faces, 10)
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
