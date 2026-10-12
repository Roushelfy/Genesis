from pathlib import Path

import numpy as np
import pytest
import quadrants as qd
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.intrinsic import kernel_bind_traction_support
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionInfo,
    StressTractionState,
    create_traction_info,
    create_traction_samples,
    func_source_traction,
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
from tests.rigid.test_stress_footprint import cpu_intrinsic_distances
from tests.rigid.test_stress_surface_fan import cpu_resolved_wrench_fit, cpu_surface_fan_geometry
from tests.utils.assertions import assert_allclose, assert_equal


@qd.kernel
def kernel_shared_traction_queries(
    forces: qd.Tensor,
    contact_state: StressContactState,
    traction_state: StressTractionState,
    traction_info: StressTractionInfo,
):
    for i_q, i_s, i_c, i_b in qd.ndrange(forces.shape[0], forces.shape[1], forces.shape[2], forces.shape[3]):
        forces[i_q, i_s, i_c, i_b] = func_source_traction(
            i_q, i_s, i_c, i_b, contact_state, traction_state, traction_info
        )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("tick", (710, 1160))
def test_plane_pair_union_preserves_each_wrench_with_destination_friction(tick):
    with np.load(Path(__file__).with_name("assets") / f"nonconvex_plane_pair_tick{tick}.npz") as saved:
        points, anchors, forces = (saved[name] for name in ("position", "anchor", "force"))
        radii, mu = saved["radius"], saved["friction"]
        source_geoms, source_pairs = saved["geom"], saved["pair"]
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    options = RigidStressOptions(mesh=mesh, method="direct", intrinsic_support="pair_union")
    model, n_envs = StressModel(options), 3
    surface = StressSurface(10, model.info)
    paths = StressSurfacePaths(model.info, 8)
    info = create_traction_info(surface.info)
    contacts, sources = create_contacts(2, n_envs, 0.006), create_sources(2, n_envs)
    footprints = create_footprints(paths.info.positions.shape[0], 2, n_envs)
    tractions = create_surface_fan_tractions(surface.info, 2, n_envs, shared_support=True)
    frictions = np.array([mu, mu * [1.15, 1.4], [0.85, 1.2]])
    contacts.position.from_numpy(np.broadcast_to(points[:, None], (2, n_envs, 3)).copy())
    contacts.force.from_numpy(np.broadcast_to(forces[:, None], (2, n_envs, 3)).copy())
    contacts.radius.from_numpy(np.broadcast_to(radii[:, None], (2, n_envs)).copy())
    contacts.friction.from_numpy(frictions.T.copy())
    contacts.valid.fill(True)
    sources.geom.from_numpy(np.broadcast_to(source_geoms[:, None], (2, n_envs)).copy())
    sources.pair.from_numpy(np.broadcast_to(source_pairs[:, None], (2, n_envs)).copy())
    sources.witness.from_numpy(np.broadcast_to(anchors[:, None], (2, n_envs, 3)).copy())
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-8, epsilon, contacts, sources, model.info, surface.info)
    assert_equal(qd_to_numpy(sources.status, transpose=True), 0)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(contacts, footprints, tractions, model.info, surface.info, paths.info, info, epsilon)
    kernel_bind_traction_support(contacts, sources, footprints, tractions)
    assert_equal(qd_to_numpy(tractions.support.start, transpose=True), 0)
    assert_equal(qd_to_numpy(tractions.support.end, transpose=True), 2)
    kernel_fit_tractions(contacts, sources, tractions, info, epsilon, 9e-9, 128)
    assert_equal(qd_to_numpy(tractions.status, transpose=True), 0)
    samples = create_traction_samples(tractions)
    kernel_traction_samples(samples, tractions, surface.info, info)
    queried = V_VEC(3, dtype=gs.qd_float, shape=(tractions.profile.shape[0], 2, 2, n_envs))
    kernel_shared_traction_queries(queried, contacts, tractions, info)
    actual_force = qd_to_numpy(queried, transpose=True, copy=True)
    sample_positions = qd_to_numpy(samples.positions, transpose=True, copy=True)
    sample_normals = qd_to_numpy(samples.normals, transpose=True, copy=True)
    sample_basis = qd_to_numpy(samples.shape, transpose=True, copy=True)
    counts = qd_to_numpy(tractions.sample_count, transpose=True, copy=True)
    indices = qd_to_numpy(tractions.sample_indices, transpose=True, copy=True)
    profiles = qd_to_numpy(tractions.profile, transpose=True, copy=True)
    anchor_faces = qd_to_numpy(contacts.anchor_face, transpose=True, copy=True)
    with np.load(mesh) as asset:
        vertices, tets, faces = (asset[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    owners = np.all(np.any(faces[:, None, :, None] == tets[None, :, None, :], axis=-1), axis=-1).argmax(axis=1)
    inward = np.einsum("fi,fi->f", normals, vertices[tets[owners]].mean(axis=1) - triangles.mean(axis=1))
    normals[inward < 0] *= -1
    cloud_positions, cloud_profiles, cloud_normals, cloud_faces, cloud_basis, destinations = [], [], [], [], [], []
    for i_s in range(2):
        positions, weights, cpu_faces, basis = cpu_surface_fan_geometry(anchors[i_s], vertices, faces, 10)
        candidates = (weights > 0) & (np.linalg.norm(positions - anchors[i_s], axis=1) < radii[i_s])
        distances = cpu_intrinsic_distances(
            anchors[i_s : i_s + 1],
            anchor_faces[0, i_s : i_s + 1],
            positions[candidates],
            cpu_faces[candidates],
            vertices,
            faces,
            qd_to_numpy(paths.info.positions, transpose=True),
            qd_to_numpy(paths.info.face_ports, transpose=True).T,
            qd_to_numpy(paths.info.starts, transpose=True),
            qd_to_numpy(paths.info.neighbors, transpose=True),
        )[0]
        cpu_profile = np.zeros(len(weights))
        cpu_profile[candidates] = weights[candidates] * np.exp(-0.5 * (distances / (0.45 * radii[i_s])) ** 2)
        cpu_profile[candidates] *= np.maximum(1 - (distances / radii[i_s]) ** 2, 0) ** 2
        for i_b in range(n_envs):
            count = counts[i_b, i_s]
            selected = indices[i_b, :count, i_s]
            assert_allclose(sample_positions[i_b, :count, i_s], positions[selected], rtol=2e-12, atol=2e-17)
            assert_allclose(sample_basis[i_b, :count, i_s], basis[selected], rtol=2e-12, atol=2e-14)
            assert_allclose(profiles[i_b, :count, i_s], cpu_profile[selected], rtol=2e-11, atol=2e-20)
        selected = cpu_profile > 0
        cloud_positions.append(positions[selected])
        cloud_profiles.append(cpu_profile[selected])
        cloud_normals.append(normals[cpu_faces[selected]])
        cloud_faces.append(cpu_faces[selected])
        cloud_basis.append(basis[selected])
        destinations.append(np.full(selected.sum(), i_s))
    positions, weights, normals, cpu_faces, basis, destination = (
        np.concatenate(values)
        for values in (cloud_positions, cloud_profiles, cloud_normals, cloud_faces, cloud_basis, destinations)
    )
    xyz = qd_to_numpy(model.info.vertices, transpose=True)
    surface_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
    nodal_reference = np.zeros((n_envs, len(xyz), 3))
    for i_b in range(n_envs):
        for i_c in range(2):
            patch = WrenchPatch(
                anchors[i_c],
                forces[i_c],
                radii[i_c],
                frictions[i_b, destination],
                np.cross(points[i_c] - anchors[i_c], forces[i_c]),
            )
            cpu_force = cpu_resolved_wrench_fit(positions, weights, normals, patch)
            np.add.at(
                nodal_reference[i_b],
                surface_nodes[cpu_faces].reshape(-1),
                (basis[:, :, None] * cpu_force[:, None]).reshape(-1, 3),
            )
            total, moment = np.zeros(3), np.zeros(3)
            for i_s in range(2):
                count = counts[i_b, i_s]
                force = actual_force[i_b, :count, i_s, i_c]
                normal = sample_normals[i_b, :count, i_s]
                compression = np.einsum("qi,qi->q", force, normal)
                assert (compression >= -1e-12).all()
                assert (
                    np.linalg.norm(force - compression[:, None] * normal, axis=1)
                    <= frictions[i_b, i_s] * compression + 1e-12
                ).all()
                total += force.sum(axis=0)
                moment += np.cross(sample_positions[i_b, :count, i_s] - points[i_c], force).sum(axis=0)
            assert np.linalg.norm(total - forces[i_c]) / np.linalg.norm(forces[i_c]) <= 1e-8
            assert np.linalg.norm(moment) / (radii[i_c] * np.linalg.norm(forces[i_c])) <= 1e-8
    state = model.create_state(n_envs, "full")
    errno, omega = V(dtype=gs.qd_int, shape=(n_envs,)), V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0)
    kernel_scatter_tractions(contacts, tractions, state, model.info, surface.info, info, errno, reset_force=True)
    assert_equal(qd_to_numpy(errno, transpose=True), 0)
    assert_allclose(qd_to_numpy(state.force, transpose=True), nodal_reference, rtol=1e-4, atol=1e-10)
    oracle = P2Shell(vertices, tets, faces, options.young, options.poisson, options.density, 2, factor_backend="none")
    rhs = oracle.balance(np.moveaxis(nodal_reference, 0, -1).reshape(oracle.ndof, n_envs))
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins, transpose=True))
    expected = np.zeros_like(rhs)
    expected[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    model.recover(omega, state)
    assert qd_to_numpy(state.valid, transpose=True).all()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(n_envs, oracle.ndof).T
    residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
    assert (residual <= np.maximum(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs, axis=0))).all()
    actual_tensor = qd_to_numpy(state.stress_tensor, transpose=True)
    actual_vm = qd_to_numpy(state.von_mises, transpose=True)
    peaks = []
    for i_b in range(n_envs):
        tensor, vm = stress_field(oracle.glambda, oracle.elements, expected[:, i_b], options.young, options.poisson)
        assert_allclose(actual_tensor[i_b], tensor, rtol=1e-4, atol=1e-3)
        assert_allclose(actual_vm[i_b, :, :, 0], vm, rtol=1e-4, atol=1e-3)
        peaks.append(vm.max())
    assert_allclose(qd_to_numpy(state.peak, transpose=True), peaks, rtol=1e-4, atol=1e-3)
