import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    StressFootprintStatus,
    create_anchor_fan_tractions,
    create_footprints,
    kernel_anchor_fan_profiles,
    kernel_footprint_ports,
    kernel_footprint_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionStatus,
    create_traction_info,
    create_tractions,
    kernel_fit_tractions,
    kernel_scatter_tractions,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh
from tests.rigid.test_stress_traction import cpu_cone_projection, kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


def cpu_intrinsic_distances(
    anchors, faces, points, target_faces, vertices, triangles, ports, face_ports, starts, neighbors
):
    """Independent graph distances and interval-union coverage on these manifold voxel fixtures."""
    edges = np.stack(
        (vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]]),
        axis=-1,
    )
    dual = np.linalg.pinv(edges)
    raw_normals = np.cross(edges[:, :, 0], edges[:, :, 1])
    normals = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]
    origin = vertices[triangles[:, 0]]
    rows = np.repeat(np.arange(len(ports)), np.diff(starts))
    lengths = np.linalg.norm(ports[neighbors] - ports[rows], axis=1)
    graph = sparse.csr_matrix((lengths, neighbors, starts), shape=(len(ports), len(ports)))
    shared = dijkstra(graph, directed=True)
    first = np.linalg.norm(ports[face_ports[faces]] - anchors[:, None], axis=2)
    to_ports = np.min(first[:, :, None] + shared[face_ports[faces]], axis=1)
    last = np.linalg.norm(points[:, None] - ports[face_ports[target_faces]], axis=2)
    distances = np.min(to_ports[:, face_ports[target_faces]] + last[None], axis=2)
    for i_b, anchor in enumerate(anchors):
        normal = normals[faces[i_b]]
        coplanar = np.abs(normals @ normal) >= 1.0 - 1e-12
        coplanar &= np.abs((origin - anchor) @ normal) <= 1e-14
        local = np.einsum("fji,fi->fj", dual[coplanar], anchor - origin[coplanar])
        start = np.column_stack((1.0 - local.sum(axis=1), local))
        delta = np.einsum("fji,qi->qfj", dual[coplanar], points - anchor)
        delta = np.concatenate((-delta.sum(axis=2, keepdims=True), delta), axis=2)
        boundary = np.divide(-start[None], delta, out=np.zeros_like(delta), where=np.abs(delta) > 1e-14)
        lower = np.maximum(0.0, np.max(np.where(delta > 1e-14, boundary, -np.inf), axis=2))
        upper = np.minimum(1.0, np.min(np.where(delta < -1e-14, boundary, np.inf), axis=2))
        is_parallel_valid = ((np.abs(delta) > 1e-14) | (start[None] >= -1e-12)).all(axis=2)
        lower = np.where(is_parallel_valid & (lower <= upper), lower, np.inf)
        order = np.argsort(lower, axis=1)
        lower = np.take_along_axis(lower, order, axis=1)
        upper = np.take_along_axis(upper, order, axis=1)
        covered = np.zeros(len(points))
        for i_face in range(lower.shape[1]):
            covered = np.where(lower[:, i_face] <= covered + 1e-12, np.maximum(covered, upper[:, i_face]), covered)
        is_straight = coplanar[target_faces] & (covered >= 1.0 - 1e-12)
        distances[i_b, is_straight] = np.linalg.norm(points[is_straight] - anchor, axis=1)
    return distances


def cpu_anchor_fan_geometry(anchors, faces, vertices, triangles, positions, weights, quadrature):
    """Independent positive quadrature of the three exact anchor/edge triangles."""
    n_quadrature = quadrature**2
    n_shared = len(positions)
    n_samples = n_shared + 2 * n_quadrature
    sample_positions = np.broadcast_to(
        np.concatenate((positions, np.zeros((2 * n_quadrature, 3)))), (len(anchors), n_samples, 3)
    ).copy()
    sample_weights = np.broadcast_to(np.r_[weights, np.zeros(2 * n_quadrature)], (len(anchors), n_samples)).copy()
    sample_faces = np.broadcast_to(np.arange(n_samples) // n_quadrature, (len(anchors), n_samples)).copy()
    gauss, gauss_weights = np.polynomial.legendre.leggauss(quadrature)
    gauss, gauss_weights = (1 + gauss) / 2, gauss_weights / 2
    u, v = np.meshgrid(gauss, gauss, indexing="ij")
    base_weights = u * gauss_weights[:, None] * gauss_weights[None]
    bary = np.stack((1 - u, u * (1 - v), u * v), axis=-1).reshape(n_quadrature, 3)
    for i_b, (anchor, face) in enumerate(zip(anchors, faces)):
        sample_faces[i_b, n_shared:] = face
        xyz = vertices[triangles[face]]
        for i_fan in range(3):
            start = face * n_quadrature if i_fan == 0 else n_shared + (i_fan - 1) * n_quadrature
            fan = np.stack((anchor, xyz[i_fan], xyz[(i_fan + 1) % 3]))
            sample_positions[i_b, start : start + n_quadrature] = bary @ fan
            area = np.linalg.norm(np.cross(fan[1] - fan[0], fan[2] - fan[0]))
            sample_weights[i_b, start : start + n_quadrature] = base_weights.reshape(n_quadrature) * area
    xyz = vertices[triangles[sample_faces]]
    edges = np.stack((xyz[:, :, 1] - xyz[:, :, 0], xyz[:, :, 2] - xyz[:, :, 0]), axis=-1)
    local = np.einsum("bqji,bqi->bqj", np.linalg.pinv(edges), sample_positions - xyz[:, :, 0])
    bary = np.concatenate((1 - local.sum(axis=2, keepdims=True), local), axis=2)
    basis = np.concatenate((bary * (2 * bary - 1), 4 * bary[:, :, [0, 0, 1]] * bary[:, :, [1, 2, 2]]), axis=2)
    return sample_positions, sample_weights, sample_faces, basis


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("anchor_fan", (False, True))
def test_native_intrinsic_profiles_and_vector_recovery(nonconvex_material_mesh, asset_tmp_path, anchor_fan):  # noqa: F811
    vertices, tetrahedra, triangles, shape = nonconvex_material_mesh
    mesh = asset_tmp_path / f"footprint-{shape}.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=triangles)
    options = RigidStressOptions(mesh=mesh, method="direct")
    model = StressModel(options)
    surface = StressSurface(4, model.info)
    paths = StressSurfacePaths(model.info, 2)
    info = create_traction_info(surface.info)
    n_envs = 3
    radius = 0.0014 if shape == "u" else 0.0012
    anchors = (
        np.array([[0.00075, 0.00251, 0.001], [0.00072, 0.00239, 0.001], [0.00083, 0.00248, 0.001]])
        if shape == "u"
        else np.array([[0.001, 0.00151, 0.00149], [0.001, 0.00140, 0.00157], [0.001, 0.00160, 0.00138]])
    )
    positions = qd_to_numpy(surface.info.positions, transpose=True, copy=True)
    weights = qd_to_numpy(surface.info.weights, transpose=True, copy=True)
    normals = qd_to_numpy(info.inward_normals, transpose=True, copy=True)
    ports = qd_to_numpy(paths.info.positions, transpose=True, copy=True)
    face_ports = qd_to_numpy(paths.info.face_ports, transpose=True, copy=True).T
    starts = qd_to_numpy(paths.info.starts, transpose=True, copy=True)
    neighbors = qd_to_numpy(paths.info.neighbors, transpose=True, copy=True)
    target_faces = np.arange(len(positions)) // surface.info.shape.shape[0]
    contacts = create_contacts(1, n_envs, radius)
    sources = create_sources(1, n_envs)
    footprints = create_footprints(len(ports), 1, n_envs)
    tractions = (
        create_anchor_fan_tractions(surface.info, 1, n_envs)
        if anchor_fan
        else create_tractions(len(positions), 1, n_envs)
    )
    kernel_profiles = kernel_anchor_fan_profiles if anchor_fan else kernel_footprint_profiles
    points = anchors + [0.00003, -0.00001, 0.00002]
    contacts.position.from_numpy(points[None])
    contacts.force.fill(0.0)
    contacts.valid.fill(True)
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(anchors[None])
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    faces = qd_to_numpy(contacts.anchor_face, transpose=True, copy=True)[:, 0]
    if anchor_fan:
        sample_positions, sample_weights, sample_faces, sample_basis = cpu_anchor_fan_geometry(
            anchors, faces, vertices, triangles, positions, weights, 4
        )
    else:
        sample_positions = np.broadcast_to(positions, (n_envs, len(positions), 3))
        sample_weights = np.broadcast_to(weights, (n_envs, len(positions)))
        sample_faces = np.broadcast_to(target_faces, (n_envs, len(positions)))
        basis = qd_to_numpy(surface.info.shape, transpose=True, copy=True)[
            np.arange(len(positions)) % surface.info.shape.shape[0]
        ]
        sample_basis = np.broadcast_to(basis, (n_envs, len(positions), 6))
    sample_normals = normals[sample_faces * surface.info.shape.shape[0]]
    distances = np.concatenate(
        [
            cpu_intrinsic_distances(
                anchors[i_b : i_b + 1],
                faces[i_b : i_b + 1],
                sample_positions[i_b],
                sample_faces[i_b],
                vertices,
                triangles,
                ports,
                face_ports,
                starts,
                neighbors,
            )
            for i_b in range(n_envs)
        ]
    )
    expected = sample_weights * np.exp(-0.5 * (distances / (0.45 * radius)) ** 2)
    expected *= np.maximum(1.0 - (distances / radius) ** 2, 0.0) ** 2
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info)
    actual_profile = qd_to_numpy(tractions.profile, transpose=True, copy=True)[:, :, 0]
    assert_equal(qd_to_numpy(footprints.status, transpose=True), StressFootprintStatus.SUCCESS)
    assert_allclose(actual_profile, expected, rtol=2e-12, atol=2e-20)
    if anchor_fan:
        assert_allclose(
            qd_to_numpy(tractions.sample_positions, transpose=True)[:, :, 0], sample_positions, rtol=0, atol=1e-15
        )
        assert_allclose(
            qd_to_numpy(tractions.sample_normals, transpose=True)[:, :, 0], sample_normals, rtol=0, atol=1e-15
        )
        assert_equal(qd_to_numpy(tractions.sample_faces, transpose=True)[:, :, 0], sample_faces)
        assert_allclose(
            qd_to_numpy(tractions.sample_shape, transpose=True)[:, :, 0], sample_basis, rtol=2e-12, atol=2e-14
        )
    euclidean = np.linalg.norm(sample_positions - anchors[:, None], axis=2)
    other_branch = (
        (sample_positions[:, :, 0] >= 0.002) & (sample_positions[:, :, 1] >= 0.002)
        if shape == "u"
        else np.isclose(sample_positions[:, :, 0], 0.002)
    )
    excluded = other_branch & (euclidean < radius) & (distances >= radius)
    if shape == "u":
        # Same material normal and close Euclidean position still cannot cross the gap.
        excluded &= np.isclose(sample_positions[:, :, 2], 0.001)
    assert excluded.any(axis=1).all()
    assert_equal(actual_profile[excluded], 0.0)
    assert (qd_to_numpy(footprints.maximum_distance, transpose=True) < radius).all()
    assert_equal(qd_to_numpy(footprints.samples, transpose=True)[:, 0], (expected > 0.0).sum(axis=1))
    # Manufacture an independent primal/dual optimum from the declared intrinsic weights.
    rng = np.random.default_rng(20261012 + (shape == "cavity"))
    friction = np.array([0.25, 0.55, 0.85])
    initial_force = np.array([0.0, 0.0, -0.001] if shape == "u" else [-0.001, 0.0, 0.0])
    dual_force = rng.uniform(-8e-5, 8e-5, size=(n_envs, 3))
    dual_moment = rng.uniform(-1e-4, 1e-4, size=(n_envs, 3))
    relative = ((sample_positions - anchors[:, None]) / radius).transpose(1, 0, 2)
    values = initial_force - dual_force[None] - np.cross(dual_moment[None], relative)
    reference = (expected / expected.sum(axis=1)[:, None]).T[..., None] * cpu_cone_projection(
        values, sample_normals.transpose(1, 0, 2), friction[None]
    )
    force = reference.sum(axis=0)
    moment = np.cross((sample_positions - anchors[:, None]).transpose(1, 0, 2), reference).sum(axis=0)
    contacts.force.from_numpy(force[None])
    contacts.friction.from_numpy(friction[None])
    sources.free_moment.from_numpy((moment - np.cross(points - anchors, force))[None])
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 1e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), StressTractionStatus.SUCCESS)
    quadrature_force = V_VEC(3, dtype=gs.qd_float, shape=(sample_positions.shape[1], 1, n_envs))
    kernel_traction_queries(quadrature_force, contacts, tractions, info)
    actual = qd_to_numpy(quadrature_force, transpose=True, copy=True)[:, :, 0].transpose((1, 0, 2))
    assert_allclose(actual, reference, rtol=1e-4, atol=1e-12)
    state = model.create_state(n_envs, "full")
    errno = V(dtype=gs.qd_int, shape=(n_envs,))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    nodal_force = qd_to_numpy(state.force, transpose=True, copy=True)
    node_positions = qd_to_numpy(model.info.vertices, transpose=True, copy=True)
    assert (np.linalg.norm(nodal_force.sum(axis=1) - force, axis=1) / np.linalg.norm(force, axis=1) <= 1e-8).all()
    recovered_moment = np.cross(node_positions[None] - points[:, None], nodal_force).sum(axis=1)
    free_moment = moment - np.cross(points - anchors, force)
    assert (
        np.linalg.norm(recovered_moment - free_moment, axis=1) / (radius * np.linalg.norm(force, axis=1)) <= 1e-8
    ).all()
    oracle = P2Shell(
        vertices, tetrahedra, triangles, options.young, options.poisson, options.density, 1, factor_backend="none"
    )
    nodes = qd_to_numpy(model.info.surface_nodes, transpose=True, copy=True).T[sample_faces]
    expected_force = np.zeros((model.info.vertices.shape[0], n_envs, 3))
    for i_b in range(n_envs):
        np.add.at(
            expected_force[:, i_b],
            nodes[i_b].reshape(-1),
            (sample_basis[i_b, :, :, None] * reference[:, i_b, None]).reshape(-1, 3),
        )
    rhs = oracle.balance(np.moveaxis(expected_force, 1, -1).reshape(oracle.ndof, n_envs))
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    displacement = np.zeros_like(rhs)
    displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    native_displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ native_displacement - rhs, axis=0)
    assert (residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))).all()
    actual_tensor = qd_to_numpy(state.stress_tensor, transpose=True)
    actual_vm = qd_to_numpy(state.von_mises, transpose=True)[..., 0]
    expected_peaks = np.empty(n_envs)
    for i_b in range(n_envs):
        tensor, vm = stress_field(oracle.glambda, oracle.elements, displacement[:, i_b], options.young, options.poisson)
        assert_allclose(actual_tensor[i_b], tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(actual_vm[i_b], vm, rtol=1e-4, atol=1e-3)
        expected_peaks[i_b] = vm.max()
    assert_allclose(qd_to_numpy(state.peak, transpose=True), expected_peaks, rtol=1e-4, atol=1e-3)
    # Only env 0 moves and changes radius; env 1 releases and env 2 stays intact.
    moved = anchors.copy()
    moved[0, 1] -= 0.00005
    sources.witness.from_numpy(moved[None])
    contacts.radius.from_numpy(np.array([[radius * 0.9, radius, radius]]))
    contacts.valid.from_numpy(np.array([[True, False, True]], dtype=gs.np_bool))
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info)
    updated = qd_to_numpy(tractions.profile, transpose=True, copy=True)[:, :, 0]
    assert_equal(updated[1], 0.0)
    assert_equal(updated[2], actual_profile[2])
    assert not np.array_equal(updated[0], actual_profile[0])
    contacts.radius.fill(radius)
    contacts.valid.fill(True)
    sources.witness.from_numpy(anchors[None])
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info)
    assert_equal(qd_to_numpy(tractions.profile, transpose=True)[:, :, 0], actual_profile)
