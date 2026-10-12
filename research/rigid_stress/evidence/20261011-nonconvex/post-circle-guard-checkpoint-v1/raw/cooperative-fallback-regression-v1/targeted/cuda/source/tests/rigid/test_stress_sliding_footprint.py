from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_anchor_fan_tractions,
    create_footprints,
    kernel_anchor_fan_profiles,
    kernel_footprint_ports,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionStatus,
    create_traction_info,
    kernel_fit_tractions,
    kernel_scatter_tractions,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from tests.rigid.test_stress_footprint import cpu_anchor_fan_geometry, cpu_intrinsic_distances
from tests.rigid.test_stress_traction import cpu_cone_projection, expand_traction_samples, kernel_traction_queries
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("tick", (110, 400, 460))
@pytest.mark.parametrize("packed", (False, True))
def test_native_saved_sliding_contact_with_finite_anchor_integration(tick, packed):
    """Actual Franka contacts whose original Q10 sampled cone systems are infeasible."""
    with np.load(Path(__file__).with_name("assets") / f"nonconvex_sliding_tick{tick}.npz") as saved:
        point, anchor, force = (saved[key] for key in ("position", "anchor", "force"))
        radius, friction = float(saved["radius"]), float(saved["friction"])
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    options = RigidStressOptions(mesh=mesh, method="direct")
    model = StressModel(options)
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 2)
    info = create_traction_info(surface.info)
    n_envs = 3
    contacts = create_contacts(1, n_envs, radius)
    sources = create_sources(1, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 1, n_envs)
    tractions = create_anchor_fan_tractions(surface.info, 1, n_envs, compact=packed, packed=packed)
    forces = np.array([1.0, 0.7, 1.3])[:, None] * force
    frictions = np.array([friction, 0.85, 1.2])
    points = np.broadcast_to(point, (n_envs, 3)).copy()
    witnesses = np.broadcast_to(anchor, (n_envs, 3)).copy()
    contacts.valid.fill(True)
    contacts.position.from_numpy(points[None])
    contacts.force.from_numpy(forces[None])
    contacts.friction.from_numpy(frictions[None])
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(witnesses[None])
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    assert_equal(qd_to_numpy(sources.status, transpose=True), 0)
    anchors = qd_to_numpy(contacts.center, transpose=True)[:, 0]
    faces = qd_to_numpy(contacts.anchor_face, transpose=True)[:, 0]
    assert_equal(faces, 49 if tick == 110 else 51)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_anchor_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 8e-9 if tick == 110 else 1e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), StressTractionStatus.SUCCESS)
    with np.load(mesh) as asset:
        vertices, tetrahedra, triangles = (asset[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    shared_positions = qd_to_numpy(surface.info.positions, transpose=True)
    shared_weights = qd_to_numpy(surface.info.weights, transpose=True)
    cpu_positions, cpu_weights, cpu_faces, cpu_basis = cpu_anchor_fan_geometry(
        anchors, faces, vertices, triangles, shared_positions, shared_weights, 10
    )
    ports = qd_to_numpy(paths.info.positions, transpose=True)
    face_ports = qd_to_numpy(paths.info.face_ports, transpose=True).T
    starts = qd_to_numpy(paths.info.starts, transpose=True)
    neighbors = qd_to_numpy(paths.info.neighbors, transpose=True)
    distances = np.concatenate(
        [
            cpu_intrinsic_distances(
                anchors[i_b : i_b + 1],
                faces[i_b : i_b + 1],
                cpu_positions[i_b],
                cpu_faces[i_b],
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
    profiles = cpu_weights * np.exp(-0.5 * (distances / (0.45 * radius)) ** 2)
    profiles *= np.maximum(1 - (distances / radius) ** 2, 0) ** 2
    indices = qd_to_numpy(tractions.sample_indices, transpose=True) if packed else None
    counts = qd_to_numpy(tractions.sample_count, transpose=True)
    assert_allclose(
        expand_traction_samples(qd_to_numpy(tractions.profile, transpose=True)[:, :, 0], indices, counts),
        profiles,
        rtol=2e-12,
        atol=2e-20,
    )
    normals = -qd_to_numpy(surface.info.face_normal, transpose=True)[cpu_faces]
    relative = (cpu_positions - anchors[:, None]) / radius
    dual = qd_to_numpy(tractions.dual, transpose=True)[:, 0]
    scales = qd_to_numpy(tractions.scale, transpose=True)[:, 0]
    values = forces[:, None] / scales[:, None, None] - dual[:, None, :3] - np.cross(dual[:, None, 3:], relative)
    expected = profiles[..., None] / profiles.sum(axis=1)[:, None, None] * scales[:, None, None]
    expected = expected * cpu_cone_projection(values, normals, frictions[:, None])
    quadrature_force = V_VEC(3, dtype=gs.qd_float, shape=(tractions.profile.shape[0], 1, n_envs))
    kernel_traction_queries(quadrature_force, contacts, tractions, info)
    actual = expand_traction_samples(qd_to_numpy(quadrature_force, transpose=True)[:, :, 0], indices, counts)
    assert_allclose(actual, expected, rtol=1e-4, atol=1e-12)
    compression = np.einsum("bqi,bqi->bq", actual, normals)
    tangent = actual - compression[..., None] * normals
    assert (compression >= -1e-12).all()
    assert (np.linalg.norm(tangent, axis=2) <= frictions[:, None] * compression + 1e-12).all()
    norm_force = np.linalg.norm(forces, axis=1)
    assert (np.linalg.norm(actual.sum(axis=1) - forces, axis=1) / norm_force <= 1e-8).all()
    moment = np.cross(cpu_positions - points[:, None], actual).sum(axis=1)
    assert (np.linalg.norm(moment, axis=1) / (norm_force * radius) <= 1e-8).all()
    state = model.create_state(n_envs, "full")
    errno = V(dtype=gs.qd_int, shape=(n_envs,))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T[cpu_faces]
    expected_nodal = np.zeros((n_envs, model.info.vertices.shape[0], 3))
    for i_b in range(n_envs):
        np.add.at(
            expected_nodal[i_b],
            surface_nodes[i_b].reshape(-1),
            (cpu_basis[i_b, :, :, None] * expected[i_b, :, None]).reshape(-1, 3),
        )
    nodal = qd_to_numpy(state.force, transpose=True)
    assert_allclose(nodal, expected_nodal, rtol=1e-4, atol=1e-12)
    assert (np.linalg.norm(nodal.sum(axis=1) - forces, axis=1) / norm_force <= 1e-8).all()
    xyz = qd_to_numpy(model.info.vertices, transpose=True)
    nodal_moment = np.cross(xyz[None] - points[:, None], nodal).sum(axis=1)
    assert (np.linalg.norm(nodal_moment, axis=1) / (norm_force * radius) <= 1e-8).all()
    oracle = P2Shell(
        vertices, tetrahedra, triangles, options.young, options.poisson, options.density, 2, factor_backend="none"
    )
    rhs = oracle.balance(np.moveaxis(expected_nodal, 0, -1).reshape(oracle.ndof, n_envs))
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    reference = np.zeros_like(rhs)
    reference[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    assert (residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))).all()
    tensor = qd_to_numpy(state.stress_tensor, transpose=True)
    vm = qd_to_numpy(state.von_mises, transpose=True)[..., 0]
    peaks = []
    for i_b in range(n_envs):
        expected_tensor, expected_vm = stress_field(
            oracle.glambda, oracle.elements, reference[:, i_b], options.young, options.poisson
        )
        assert_allclose(tensor[i_b], expected_tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(vm[i_b], expected_vm, rtol=1e-4, atol=1e-3)
        peaks.append(expected_vm.max())
    assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)
