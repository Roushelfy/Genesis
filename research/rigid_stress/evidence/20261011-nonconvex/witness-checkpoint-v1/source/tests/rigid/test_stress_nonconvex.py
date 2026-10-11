from itertools import permutations

import numpy as np
import pytest
import quadrants as qd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo, StressSurfacePaths, func_surface_distance
from genesis.engine.solvers.rigid.stress.provenance import StressSourceStatus, create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.topology import build_surface_topology
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from tests.utils.assertions import assert_allclose, assert_equal


@qd.kernel
def kernel_surface_queries(
    faces: qd.types.ndarray(),
    points: qd.types.ndarray(),
    distances: qd.Tensor,
    straight: qd.Tensor,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    for i in range(distances.shape[0]):
        anchor = qd.Vector([points[i, 0, a] for a in qd.static(range(3))])
        target = qd.Vector([points[i, 1, a] for a in qd.static(range(3))])
        distance, shortcut = func_surface_distance(
            faces[i, 0], faces[i, 1], anchor, target, stress_info, surface_info, path_info
        )
        distances[i], straight[i] = distance, shortcut


@pytest.fixture(params=("u", "cavity"))
def nonconvex_material_mesh(request):
    shape = (3, 3, 1) if request.param == "u" else (3, 3, 3)
    cells = np.indices(shape).reshape((3, -1)).T
    if request.param == "u":
        cells = cells[(cells[:, 0] != 1) | (cells[:, 1] == 0)]
    else:
        cells = cells[(cells != 1).any(axis=1)]
    steps = np.eye(3, dtype=gs.np_int)[np.array(list(permutations(range(3))))]
    offsets = np.concatenate((np.zeros((6, 1, 3), dtype=gs.np_int), steps.cumsum(axis=1)), axis=1)
    coords = cells[:, None, None, :] + offsets
    grid_shape = tuple(size + 1 for size in shape)
    tetrahedra = np.ravel_multi_index(np.moveaxis(coords, -1, 0), grid_shape).reshape((-1, 4))
    used, inverse = np.unique(tetrahedra, return_inverse=True)
    vertices = np.indices(grid_shape).reshape((3, -1)).T[used] * 0.001
    tetrahedra = inverse.reshape((-1, 4))
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    unique, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1), axis=0, return_counts=True)
    surface = unique[counts == 1]
    surface[::2] = surface[::2, ::-1]
    return vertices, tetrahedra, surface, request.param


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_nonconvex_material_normals(nonconvex_material_mesh, tmp_path):
    vertices, tetrahedra, surface, shape = nonconvex_material_mesh
    mesh = tmp_path / "nonconvex.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    sampled = StressSurface(4, model.info)
    normals = qd_to_numpy(sampled.info.face_normal, transpose=True)
    centers = vertices[surface].mean(axis=1)
    # Axis-aligned voxel boundaries give an independent material-side oracle.
    expected = np.zeros_like(normals)
    upper = np.array([0.003, 0.003, 0.001 if shape == "u" else 0.003])
    for axis in range(3):
        expected[np.isclose(centers[:, axis], 0.0), axis] = -1
        expected[np.isclose(centers[:, axis], upper[axis]), axis] = 1
    if shape == "u":
        expected[np.isclose(centers[:, 0], 0.001) & (centers[:, 1] > 0.001), 0] = 1
        expected[np.isclose(centers[:, 0], 0.002) & (centers[:, 1] > 0.001), 0] = -1
        expected[np.isclose(centers[:, 1], 0.001) & (centers[:, 0] > 0.001) & (centers[:, 0] < 0.002), 1] = 1
    else:
        interior = (centers > 0).all(axis=1) & (centers < upper).all(axis=1)
        for axis in range(3):
            expected[interior & np.isclose(centers[:, axis], 0.001), axis] = 1
            expected[interior & np.isclose(centers[:, axis], 0.002), axis] = -1
    assert_equal(np.linalg.norm(expected, axis=1), 1.0)
    assert_allclose(normals, expected, atol=1e-14)
    # Both fixtures contain inward-facing surfaces that the COM rule reverses.
    assert (np.einsum("fi,fi->f", expected, centers - vertices.mean(axis=0)) < 0).any()


@pytest.mark.required
@pytest.mark.precision("64")
def test_material_boundary_adjacency_uses_vertex_identity(nonconvex_material_mesh):
    _, tetrahedra, surface, _ = nonconvex_material_mesh
    topology = build_surface_topology(tetrahedra, surface)
    assert (topology.neighbors >= 0).all()
    neighbors = surface[topology.neighbors]
    edges = surface[:, [[0, 1], [1, 2], [2, 0]]]
    assert_equal((neighbors[:, :, :, None] == edges[:, :, None, :]).any(axis=2).all(axis=2), True)
    count = tetrahedra.max() + 1
    separate = build_surface_topology(
        np.concatenate((tetrahedra, tetrahedra + count)), np.concatenate((surface, surface + count))
    )
    assert_equal(separate.neighbors[: len(surface)], topology.neighbors)
    assert_equal(separate.neighbors[len(surface) :], topology.neighbors + len(surface))


@pytest.mark.required
@pytest.mark.precision("64")
def test_material_boundary_rejects_interior_and_duplicate_faces(nonconvex_material_mesh):
    _, tetrahedra, surface, _ = nonconvex_material_mesh
    with pytest.raises(gs.GenesisException, match="duplicated"):
        build_surface_topology(tetrahedra, np.concatenate((surface, surface[:1, ::-1])))
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1), axis=0, return_counts=True)
    with pytest.raises(gs.GenesisException, match="exactly one owning"):
        build_surface_topology(tetrahedra, faces[counts == 2][:1])


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_surface_paths_do_not_cross_gaps_or_holes(nonconvex_material_mesh, tmp_path):
    vertices, tetrahedra, surface, shape = nonconvex_material_mesh
    mesh = tmp_path / "nonconvex.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    sampled = StressSurface(4, model.info)
    if shape == "u":
        points = np.array(
            [
                [[0.00021, 0.00138, 0.001], [0.00078, 0.00262, 0.001]],
                [[0.00047, 0.00253, 0.001], [0.00246, 0.00249, 0.001]],
                [[0.001, 0.00251, 0.00049], [0.002, 0.00248, 0.00051]],
            ]
        )
    else:
        points = np.array(
            [
                [[0.00021, 0.00138, 0.003], [0.00278, 0.00262, 0.003]],
                [[0.001, 0.00151, 0.00149], [0.002, 0.00148, 0.00151]],
                [[0.00021, 0.00138, 0.003], [0.001, 0.00151, 0.00149]],
            ]
        )
    origins = vertices[surface[:, 0]]
    edges = vertices[surface[:, 1:]] - origins[:, None, :]
    dual = np.linalg.solve(edges @ np.swapaxes(edges, 1, 2), edges)
    relative = points.reshape((-1, 3))[:, None, :] - origins
    local = np.einsum("fai,pfi->pfa", dual, relative)
    normal = np.cross(edges[:, 0], edges[:, 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    on_face = (local >= -1e-12).all(axis=2) & (local.sum(axis=2) <= 1.0 + 1e-12)
    on_face &= np.abs(np.einsum("pfi,fi->pf", relative, normal)) < 1e-14
    assert on_face.any(axis=1).all()
    faces = on_face.argmax(axis=1).reshape((-1, 2)).astype(gs.np_int)
    result = V(dtype=gs.qd_float, shape=(3,))
    straight = V(dtype=gs.qd_bool, shape=(3,))
    previous = None
    for subdivisions in (2, 4):
        paths = StressSurfacePaths(model.info, subdivisions)
        lengths = qd_to_numpy(paths.info.lengths, transpose=True)
        columns = qd_to_numpy(paths.info.neighbors, transpose=True)
        starts = qd_to_numpy(paths.info.starts, transpose=True)
        reference = dijkstra(sparse.csr_matrix((lengths, columns, starts)), directed=True)
        assert_allclose(qd_to_numpy(paths.info.distances, transpose=True), reference, atol=1e-14)
        kernel_surface_queries(faces, points, result, straight, model.info, sampled.info, paths.info)
        distances = qd_to_numpy(result, transpose=True).copy()
        assert_equal(qd_to_numpy(straight, transpose=True), [True, False, False])
        assert_allclose(distances[0], np.linalg.norm(points[0, 1] - points[0, 0]), atol=1e-14)
        if shape == "u":
            assert (distances[1:] > 0.003).all()
            assert np.linalg.norm(points[2, 1] - points[2, 0]) < 0.0011
        else:
            assert distances[1] > 0.0018
            assert np.isinf(distances[2])
        if previous is not None:
            assert (distances <= previous + 1e-14).all()
        previous = distances


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_nonconvex_full_equilibrium_and_field(nonconvex_material_mesh, tmp_path):
    vertices, tetrahedra, surface, _ = nonconvex_material_mesh
    mesh = tmp_path / "nonconvex.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    options = RigidStressOptions(mesh=mesh, method="direct")
    model = StressModel(options)
    maximum, complete = model.create_state(2), model.create_state(2, "full")
    force = np.random.default_rng(20261011).normal(size=(model.info.vertices.shape[0], 2, 3)) * 1e-4
    omega = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    omega.fill(0)
    for state in (maximum, complete):
        state.force.from_numpy(force)
        model.recover(omega, state)
        assert qd_to_numpy(state.valid, transpose=True).all()

    oracle = P2Shell(
        vertices, tetrahedra, surface, options.young, options.poisson, options.density, 1, factor_backend="none"
    )
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    raw = np.moveaxis(force, 1, -1).reshape((oracle.ndof, 2))
    rhs = oracle.balance(raw)
    displacement = qd_to_numpy(complete.displacement, transpose=True).reshape((2, oracle.ndof)).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    budget = np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))
    # Include every row, particularly the six gauge rows omitted by the pinned solve.
    assert (residual <= budget).all()
    reference = np.zeros_like(rhs)
    reference[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    tensor = qd_to_numpy(complete.stress_tensor, transpose=True)
    vm = qd_to_numpy(complete.von_mises, transpose=True)[..., 0]
    for i_b in range(2):
        expected_tensor, expected_vm = stress_field(
            oracle.glambda, oracle.elements, reference[:, i_b], options.young, options.poisson
        )
        assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(vm[i_b], expected_vm, rtol=1e-4, atol=1e-3)
    assert_equal(vm.max(axis=(1, 2)), qd_to_numpy(complete.peak, transpose=True))
    assert_allclose(qd_to_numpy(maximum.peak, transpose=True), vm.max(axis=(1, 2)), rtol=1e-4, atol=1e-3)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_source_provenance_ambiguity_and_offset_wrench(nonconvex_material_mesh, tmp_path):
    vertices, tetrahedra, surface, shape = nonconvex_material_mesh
    mesh = tmp_path / "nonconvex.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    sampled = StressSurface(4, model.info)
    anchor = np.array([0.001, 0.00237 if shape == "u" else 0.00137, 0.00042 if shape == "u" else 0.00142])
    origins = vertices[surface[:, 0]]
    edges = vertices[surface[:, 1:]] - origins[:, None, :]
    dual = np.linalg.solve(edges @ np.swapaxes(edges, 1, 2), edges)
    local = np.einsum("fai,fi->fa", dual, anchor - origins)
    plane = np.einsum("fi,fi->f", np.cross(edges[:, 0], edges[:, 1]), anchor - origins)
    on_face = (local >= 0).all(axis=1) & (local.sum(axis=1) <= 1.0) & (np.abs(plane) < 1e-20)
    assert on_face.sum() == 1
    face = on_face.argmax()
    bary = np.r_[1.0 - local[face].sum(), local[face]]
    contacts = create_contacts(1, 6, 0.0015)
    sources = create_sources(1, 6)
    point = np.broadcast_to(anchor + [0.00003, 0.00001, -0.00002], (1, 6, 3)).copy()
    force = np.broadcast_to([-0.001, 0.0002, -0.0001], (1, 6, 3)).copy()
    moment = np.broadcast_to([1e-8, -2e-8, 3e-8], (1, 6, 3)).copy()
    witnesses = np.broadcast_to(anchor, (1, 6, 3)).copy()
    witnesses[0, 1] += [0.00005, 0, 0]
    witnesses[0, 2] = [0.0015, anchor[1], anchor[2]] if shape == "u" else [0.0015] * 3
    witnesses[0, 4] = 0.02
    witnesses[0, 5] = witnesses[0, 2]
    contacts.position.from_numpy(point)
    contacts.force.from_numpy(force)
    contacts.valid.fill(True)
    contacts.center.fill(float("nan"))
    sources.geom.from_numpy(np.array([[7, 7, 7, -1, 7, 7]], dtype=gs.np_int))
    sources.pair.fill(11)
    sources.witness.from_numpy(witnesses)
    sources.free_moment.from_numpy(moment)
    sources.witness_face.from_numpy(np.array([[face, -1, -1, -1, -1, face]], dtype=gs.np_int))
    sources.witness_barycentric.from_numpy(np.broadcast_to(bary, (1, 6, 3)).copy())
    kernel_resolve_sources(0.0006, float(np.finfo(gs.np_float).eps), contacts, sources, sampled.info)
    assert_equal(
        qd_to_numpy(sources.status, transpose=True)[:, 0],
        [
            StressSourceStatus.SUCCESS,
            StressSourceStatus.SUCCESS,
            StressSourceStatus.AMBIGUOUS,
            StressSourceStatus.INVALID_INPUT,
            StressSourceStatus.UNRESOLVED,
            StressSourceStatus.SUCCESS,
        ],
    )
    centers = qd_to_numpy(contacts.center, transpose=True)[:, 0]
    assert_allclose(centers[[0, 1, 5]], anchor, atol=1e-15)
    assert np.isnan(centers[[2, 3, 4]]).all()
    # Projection retains the original point and all six wrench components.
    assert_equal(qd_to_numpy(contacts.position, transpose=True)[:, 0], point[0])
    assert_equal(qd_to_numpy(contacts.force, transpose=True)[:, 0], force[0])
    mapped_moment = qd_to_numpy(sources.anchor_moment, transpose=True)[[0, 1, 5], 0]
    original_moment = mapped_moment + np.cross(centers[[0, 1, 5]] - point[0, [0, 1, 5]], force[0, [0, 1, 5]])
    assert_allclose(original_moment, moment[0, [0, 1, 5]], atol=1e-22)
    witnesses[0, 0] = witnesses[0, 2]
    sources.witness.from_numpy(witnesses)
    sources.witness_face.from_numpy(np.array([[-1, -1, -1, -1, -1, face]], dtype=gs.np_int))
    kernel_resolve_sources(0.0006, float(np.finfo(gs.np_float).eps), contacts, sources, sampled.info)
    assert_equal(qd_to_numpy(sources.status, transpose=True)[0, 0], StressSourceStatus.AMBIGUOUS)
    assert np.isnan(qd_to_numpy(contacts.center, transpose=True)[0, 0]).all()
    assert np.isnan(qd_to_numpy(sources.anchor_moment, transpose=True)[0, 0]).all()
    assert_allclose(qd_to_numpy(contacts.center, transpose=True)[[1, 5], 0], anchor, atol=1e-15)
