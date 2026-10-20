import xml.etree.ElementTree as ET

import numpy as np
import pytest
import quadrants as qd
import trimesh
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.pair_geometry import (
    PairGeometryStatus,
    StressPairGeometry,
    StressPairState,
    StressPairSamples,
    func_opponent_boundary,
    func_opponent_occluded,
    kernel_pair_samples,
)
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_capture_rigid_sources
from genesis.engine.solvers.rigid.stress.topology import build_surface_topology
from genesis.utils import array_class
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh  # noqa: F401
from tests.utils.assertions import assert_allclose, assert_equal


@qd.kernel
def kernel_boundary_queries(
    geoms: qd.types.ndarray(),
    points: qd.types.ndarray(),
    result: qd.types.ndarray(),
    status: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    epsilon: float,
):
    for i in range(points.shape[0]):
        point = qd.Vector([points[i, a] for a in qd.static(range(3))])
        closest, normal, gap, face, flag = func_opponent_boundary(geoms[i], 0, point, dyn_state, dyn_info, epsilon)
        for a in qd.static(range(3)):
            result[i, a], result[i, a + 3] = closest[a], normal[a]
        result[i, 6] = gap
        status[i] = flag


@qd.kernel
def kernel_opponent_occlusion(
    links: qd.types.ndarray(),
    points: qd.types.ndarray(),
    targets: qd.types.ndarray(),
    result: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    epsilon: float,
):
    for i in range(points.shape[0]):
        link = links[i]
        point = qd.Vector([points[i, a] for a in qd.static(range(3))])
        target = qd.Vector([targets[i, a] for a in qd.static(range(3))])
        result[i] = func_opponent_occluded(link, 0, point, target, dyn_state, dyn_info, epsilon)


def associate(scene, entry, sources):
    solver = scene.rigid_solver
    kernel_associate(
        entry.link.idx,
        solver._links_offset_pos,
        solver._links_offset_quat,
        entry.omega,
        solver.dyn_state,
        entry.contacts,
        solver.collider.collider_state,
        solver._links_offset_quat.ndim == 3,
        True,
        entry.state.step_valid,
        solver._errno,
    )
    kernel_capture_rigid_sources(entry.link.idx, entry.contacts, sources, solver.collider.collider_state)


def mesh_urdf(path):
    root = ET.Element("robot", name="query_material")
    link = ET.SubElement(root, "link", name="material")
    collision = ET.SubElement(link, "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=str(path))
    return ET.tostring(root, encoding="unicode")


@pytest.mark.required
@pytest.mark.precision("64")
def test_pair_boundary_primitives_and_material_frame(asset_tmp_path):
    mesh = trimesh.creation.box(extents=(0.004, 0.006, 0.008))
    path = asset_tmp_path / "query-box.obj"
    mesh.export(path)
    xml = ET.Element("mujoco")
    body = ET.SubElement(ET.SubElement(xml, "worldbody"), "body", pos="0.16 0 0")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", type="capsule", size="0.001 0.002")
    scene = gs.Scene(
        show_viewer=False,
    )
    entities = [
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.004, 0.006, 0.008),
            ),
        ),
        scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.04, 0.0, 0.0),
                radius=0.002,
            ),
        ),
        scene.add_entity(
            morph=gs.morphs.Cylinder(
                pos=(0.08, 0.0, 0.0),
                radius=0.002,
                height=0.004,
            ),
        ),
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=mesh_urdf(path),
                pos=(0.12, 0.0, 0.0),
                align=False,
                convexify=False,
                decimate=False,
                preserve_mesh_precision=True,
            ),
        ),
        scene.add_entity(
            morph=gs.morphs.MJCF(
                file=ET.tostring(xml, encoding="unicode"),
            ),
        ),
        scene.add_entity(
            morph=gs.morphs.Plane(
                pos=(0.2, 0.0, 0.0),
            ),
        ),
    ]
    scene.build()
    geoms = np.array(
        [next(geom.idx for link in entity.links for geom in link.geoms) for entity in entities], dtype=gs.np_int
    )
    points = np.array(
        [[0.003, 0.001, 0.001], [0, 0, 0.003], [0.003, 0, 0], [0.003, 0.001, 0.001], [0, 0, 0.004], [0, 0, 0.001]]
    )
    normals = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    expected = points - 0.001 * normals
    result = np.empty((len(points), 7))
    status = np.empty(len(points), dtype=gs.np_int)
    solver = scene.rigid_solver
    offsets = tensor_to_array(solver.get_geoms_pos())[geoms]
    quaternion = tensor_to_array(solver.get_geoms_quat())[geoms]
    initial_rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    kernel_boundary_queries(
        geoms,
        initial_rotation.apply(points) + offsets,
        result,
        status,
        solver.dyn_state,
        solver.dyn_info,
        float(np.finfo(gs.np_float).eps),
    )
    np.testing.assert_array_equal(status, 0)
    np.testing.assert_allclose(result[:, :3], initial_rotation.apply(expected) + offsets, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[:, 3:6], initial_rotation.apply(normals), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[:, 6], 0.001, rtol=0, atol=1e-12)
    ray_points = np.tile([[0.004, 0, 0], [0.004, 0.01, 0.01], [0.004, 0, 0]], (len(entities), 1, 1))
    ray_targets = ray_points.copy()
    ray_targets[:, 0, 0] = -0.004
    ray_targets[:, 1, 0] = 0.005
    ray_targets[:, 2, 0] = [0.002, 0.002, 0.002, 0.002, 0.001, 0.0]
    ray_points[-1] = [[0, 0, 0.001], [0, 0, 0.002], [0, 0, 0.001]]
    ray_targets[-1] = [[0, 0, -0.001], [0, 0, 0.003], [0, 0, 0.0]]
    ray_rotation = Rotation.from_quat(np.repeat(quaternion[:, [1, 2, 3, 0]], 3, axis=0))
    ray_offsets = np.repeat(offsets, 3, axis=0)
    links = np.repeat(np.array([entity.base_link.idx for entity in entities], dtype=gs.np_int), 3)
    blocked = np.empty(len(links), dtype=gs.np_int)
    kernel_opponent_occlusion(
        links,
        ray_rotation.apply(ray_points.reshape((-1, 3))) + ray_offsets,
        ray_rotation.apply(ray_targets.reshape((-1, 3))) + ray_offsets,
        blocked,
        solver.dyn_state,
        solver.dyn_info,
        float(np.finfo(gs.np_float).eps),
    )
    assert_equal(blocked.reshape((-1, 3)), [True, False, False])
    # A live pose change must be read in the next query; no world-space cache.
    rotation = Rotation.from_euler("xyz", (17, -23, 31), degrees=True)
    translation = np.array([0.02, -0.03, 0.01])
    entities[0].set_pos(translation)
    entities[0].set_quat(rotation.as_quat()[[3, 0, 1, 2]])
    query = rotation.apply(points[:1]) + translation
    kernel_boundary_queries(
        geoms[:1], query, result[:1], status[:1], solver.dyn_state, solver.dyn_info, float(np.finfo(gs.np_float).eps)
    )
    np.testing.assert_array_equal(status[:1], 0)
    np.testing.assert_allclose(result[0, :3], rotation.apply(expected[0]) + translation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[0, 3:6], rotation.apply(normals[0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[0, 6], 0.001, rtol=0, atol=1e-12)
    kernel_boundary_queries(
        geoms[:1],
        translation[None],
        result[:1],
        status[:1],
        solver.dyn_state,
        solver.dyn_info,
        float(np.finfo(gs.np_float).eps),
    )
    np.testing.assert_array_equal(status[:1], PairGeometryStatus.AMBIGUOUS)


@pytest.mark.required
@pytest.mark.precision("64")
def test_pair_material_visibility_and_current_velocity(nonconvex_material_mesh, asset_tmp_path):  # noqa: F811
    vertices, tets, faces, shape = nonconvex_material_mesh
    elastic = asset_tmp_path / "pair-elastic.npz"
    collision = asset_tmp_path / "pair-material.obj"
    np.savez(elastic, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
    topology = build_surface_topology(tets, faces)
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flipped = np.einsum("ij,ij->i", normals, vertices[topology.opposite_vertices] - corners[:, 0]) > 0
    oriented = faces.copy()
    oriented[flipped] = oriented[flipped, ::-1]
    trimesh.Trimesh(vertices, oriented, process=False).export(collision)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=False,
    )
    observed = scene.add_entity(
        morph=gs.morphs.URDF(
            file=mesh_urdf(collision),
            align=False,
            convexify=False,
            decimate=False,
            preserve_mesh_precision=True,
        ),
    )
    observed.base_link.configure_stress_recovery(
        gs.options.RigidStressOptions(
            mesh=elastic,
            method="pcg",
            update_mode="manual",
            quadrature=2,
        )
    )
    pad = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0015, 0.0015, 0.0065),
            size=(0.02, 0.02, 0.001),
        ),
    )
    scene.build(n_envs=2)
    solver = scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    sources = create_sources(*entry.contacts.valid.shape)
    associate(scene, entry, sources)
    query = np.array([[0.0015, 0.0015, 0.0016], [0.0015, 0.0015, 0.0015]])
    closest, direction = np.array([0.0015, 0.0015, 0.002]), np.array([0, 0, -1])
    if shape == "u":
        query = np.array([[0.0016, 0.002, 0.0005], [0.0015, 0.002, 0.0005]])
        closest, direction = np.array([0.002, 0.002, 0.0005]), np.array([-1, 0, 0])
    queried = np.empty((2, 7))
    flags = np.empty(2, dtype=gs.np_int)
    geom_indices = np.full(2, observed.base_link.geoms[0].idx, dtype=gs.np_int)
    kernel_boundary_queries(
        geom_indices, query, queried, flags, solver.dyn_state, solver.dyn_info, float(np.finfo(gs.np_float).eps)
    )
    np.testing.assert_array_equal(flags, [PairGeometryStatus.SUCCESS, PairGeometryStatus.AMBIGUOUS])
    np.testing.assert_allclose(queried[0, :3], closest, rtol=0, atol=1e-12)
    np.testing.assert_allclose(queried[0, 3:6], direction, rtol=0, atol=1e-12)
    np.testing.assert_allclose(queried[0, 6], 0.0004, rtol=0, atol=1e-12)
    pairs = StressPairState(
        environment=V(dtype=gs.qd_int, shape=(2,)),
        opponent_link=V(dtype=gs.qd_int, shape=(2,)),
        source_count=V(dtype=gs.qd_int, shape=(2,)),
        force=V_VEC(3, dtype=gs.qd_float, shape=(2,)),
        moment=V_VEC(3, dtype=gs.qd_float, shape=(2,)),
    )
    pairs.environment.from_numpy(np.arange(2, dtype=gs.np_int))
    pairs.opponent_link.fill(pad.base_link.idx)
    samples_shape = (entry.surface.info.positions.shape[0], 2)
    samples = StressPairSamples(
        opponent_position=V_VEC(3, dtype=gs.qd_float, shape=samples_shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=samples_shape),
        relative_velocity=V_VEC(3, dtype=gs.qd_float, shape=samples_shape),
        gap=V(dtype=gs.qd_float, shape=samples_shape),
        projected_area=V(dtype=gs.qd_float, shape=samples_shape),
        opponent_geom=V(dtype=gs.qd_int, shape=samples_shape),
        opponent_face=V(dtype=gs.qd_int, shape=samples_shape),
        status=V(dtype=gs.qd_int, shape=samples_shape),
    )
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_pair_samples(
        entry.link.idx, solver.dyn_state, entry.contacts, pairs, samples, solver.dyn_info, entry.surface.info, epsilon
    )
    status = qd_to_numpy(samples.status, transpose=True, copy=True)
    np.testing.assert_array_equal(qd_to_numpy(samples.opponent_geom, transpose=True), pad.base_link.geoms[0].idx)
    gap = qd_to_numpy(samples.gap, transpose=True, copy=True)
    position = qd_to_numpy(entry.surface.info.positions, transpose=True, copy=True)
    n_quad = entry.surface.info.shape.shape[0]
    normal = np.repeat(qd_to_numpy(entry.surface.info.face_normal, transpose=True), n_quad, axis=0)
    facing = normal[:, 2] > 0.5
    visible = facing & np.isclose(position[:, 2], vertices[:, 2].max(), rtol=0, atol=1e-12)
    expected = np.full(len(position), PairGeometryStatus.BACK_FACING)
    expected[facing & ~visible] = PairGeometryStatus.OCCLUDED
    expected[visible] = PairGeometryStatus.SUCCESS
    np.testing.assert_array_equal(status, np.tile(expected, (2, 1)))
    np.testing.assert_allclose(gap, np.tile(0.006 - position[:, 2], (2, 1)), rtol=0, atol=1e-12)
    if shape == "cavity":
        assert (status == PairGeometryStatus.OCCLUDED).any()
    area = qd_to_numpy(samples.projected_area, transpose=True, copy=True)
    weights = qd_to_numpy(entry.surface.info.weights, transpose=True)
    np.testing.assert_allclose(area[:, visible], np.tile(weights[visible], (2, 1)), rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(area[:, ~visible], 0)
    baseline = scene.get_state()
    pad.set_pos((0.0015, 0.0015, 0.0055), envs_idx=[0])
    pad.set_dofs_velocity((0.02, 0.0, 0.0), dofs_idx_local=[0, 1, 2], envs_idx=[0])
    kernel_pair_samples(
        entry.link.idx, solver.dyn_state, entry.contacts, pairs, samples, solver.dyn_info, entry.surface.info, epsilon
    )
    current = qd_to_numpy(samples.gap, transpose=True, copy=True)
    np.testing.assert_allclose(current[0], gap[0] - 0.001, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(current[1], gap[1])
    velocity = qd_to_numpy(samples.relative_velocity, transpose=True, copy=True)
    np.testing.assert_allclose(velocity[0], np.tile([0.02, 0, 0], (len(position), 1)), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(velocity[1], 0)
    scene.reset(state=baseline, envs_idx=[0])
    associate(scene, entry, sources)
    kernel_pair_samples(
        entry.link.idx, solver.dyn_state, entry.contacts, pairs, samples, solver.dyn_info, entry.surface.info, epsilon
    )
    np.testing.assert_array_equal(qd_to_numpy(samples.gap, transpose=True), gap)
    np.testing.assert_array_equal(qd_to_numpy(samples.relative_velocity, transpose=True), 0)


@pytest.mark.required
@pytest.mark.precision("64")
def test_pair_grouping_preserves_partitioned_wrench_and_isolation(asset_tmp_path):
    vertices = np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0], [0, 0, 3]]) * 0.001
    tets = np.array([[0, 1, 2, 3]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    elastic = asset_tmp_path / "pair-group-elastic.npz"
    collision = asset_tmp_path / "pair-group.obj"
    np.savez(elastic, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
    trimesh.Trimesh(vertices, faces, process=False).export(collision)
    root = ET.Element("robot", name="two_pad_components")
    link = ET.SubElement(root, "link", name="pad")
    for x in (-0.005, 0.005):
        child = ET.SubElement(link, "collision")
        ET.SubElement(child, "origin", xyz=f"{x} 0 -0.001")
        ET.SubElement(ET.SubElement(child, "geometry"), "box", size="0.01 0.02 0.002")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            enable_contact_patch=True,
        ),
        show_viewer=False,
    )
    observed = scene.add_entity(
        morph=gs.morphs.URDF(
            file=mesh_urdf(collision),
            pos=(0.0, 0.0, -0.0001),
            align=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=True,
        ),
    )
    observed.base_link.configure_stress_recovery(
        gs.options.RigidStressOptions(
            mesh=elastic,
            method="pcg",
            update_mode="manual",
            quadrature=2,
        )
    )
    pad = scene.add_entity(
        morph=gs.morphs.URDF(
            file=ET.tostring(root, encoding="unicode"),
            fixed=True,
        ),
    )
    scene.build(n_envs=2)
    solver = scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    collider = solver.collider.collider_state
    solver.collider.detection()
    counts = qd_to_numpy(collider.n_contacts, transpose=True, copy=True)
    order = qd_to_numpy(collider.contact_sort_idx, transpose=True, copy=True)
    normal = qd_to_numpy(collider.contact_data.normal, transpose=True, copy=True)
    force = qd_to_numpy(collider.contact_data.force, transpose=True, copy=True)
    assert (counts >= 2).all() and 2 * counts.max() <= order.shape[1]
    for env, count in enumerate(counts):
        slots = order[env, :count]
        force[env, slots] = -0.1 * normal[env, slots] + [0.005, 0, 0]
    collider.contact_data.force.from_numpy(force.swapaxes(0, 1))
    sources = create_sources(*entry.contacts.valid.shape)
    geometry = StressPairGeometry(entry.contacts)
    epsilon = float(np.finfo(gs.np_float).eps)
    associate(scene, entry, sources)
    geometry.update(
        entry.link.idx,
        entry.contacts,
        sources,
        entry.surface.info,
        solver.dyn_info,
        solver.dyn_state,
        collider,
        epsilon,
    )
    assert geometry.pairs.environment.shape == (2,)
    np.testing.assert_array_equal(qd_to_numpy(geometry.pairs.opponent_link, transpose=True), pad.base_link.idx)
    np.testing.assert_array_equal(qd_to_numpy(geometry.pairs.source_count, transpose=True), counts)
    original_force = qd_to_numpy(geometry.pairs.force, transpose=True, copy=True)
    original_moment = qd_to_numpy(geometry.pairs.moment, transpose=True, copy=True)
    original_status = qd_to_numpy(geometry.samples.status, transpose=True, copy=True)
    assert np.linalg.norm(original_force) > 0.1
    points = qd_to_numpy(entry.contacts.position, transpose=True, copy=True)
    loads = qd_to_numpy(entry.contacts.force, transpose=True, copy=True)
    valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)
    np.testing.assert_allclose(original_force, np.sum(np.where(valid[..., None], loads, 0), axis=1), rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        original_moment, np.sum(np.where(valid[..., None], np.cross(points, loads), 0), axis=1), rtol=0, atol=1e-16
    )
    # Split each source into two coincident half loads, without adding a physical pair.
    for env, count in enumerate(counts):
        order[env, : 2 * count] = np.repeat(order[env, :count], 2)
    collider.contact_sort_idx.from_numpy(order.T)
    collider.n_contacts.from_numpy(2 * counts)
    collider.contact_data.force.from_numpy(0.5 * force.swapaxes(0, 1))
    associate(scene, entry, sources)
    geometry.update(
        entry.link.idx,
        entry.contacts,
        sources,
        entry.surface.info,
        solver.dyn_info,
        solver.dyn_state,
        collider,
        epsilon,
    )
    assert geometry.pairs.environment.shape == (2,)
    np.testing.assert_allclose(qd_to_numpy(geometry.pairs.force, transpose=True), original_force, rtol=0, atol=1e-14)
    np.testing.assert_allclose(qd_to_numpy(geometry.pairs.moment, transpose=True), original_moment, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(qd_to_numpy(geometry.samples.status, transpose=True), original_status)
    provenance = qd_to_numpy(sources.geom, transpose=True, copy=True)
    provenance[0, 0] = -1
    sources.geom.from_numpy(provenance.T)
    geometry.update(
        entry.link.idx,
        entry.contacts,
        sources,
        entry.surface.info,
        solver.dyn_info,
        solver.dyn_state,
        collider,
        epsilon,
    )
    np.testing.assert_array_equal(qd_to_numpy(geometry.index.invalid, transpose=True), [True, False])
    np.testing.assert_array_equal(qd_to_numpy(geometry.pairs.environment, transpose=True), [1])
    np.testing.assert_allclose(
        qd_to_numpy(geometry.pairs.force, transpose=True), original_force[1:], rtol=0, atol=1e-14
    )
    associate(scene, entry, sources)
    geometry.update(
        entry.link.idx,
        entry.contacts,
        sources,
        entry.surface.info,
        solver.dyn_info,
        solver.dyn_state,
        collider,
        epsilon,
    )
    np.testing.assert_array_equal(qd_to_numpy(geometry.index.invalid, transpose=True), False)
    np.testing.assert_allclose(qd_to_numpy(geometry.pairs.force, transpose=True), original_force, rtol=0, atol=1e-14)
    entry.contacts.valid.fill(False)
    geometry.update(
        entry.link.idx,
        entry.contacts,
        sources,
        entry.surface.info,
        solver.dyn_info,
        solver.dyn_state,
        collider,
        epsilon,
    )
    assert geometry.pairs is None and geometry.samples is None


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("nonconvex_material_mesh", ["u"], indirect=True)
def test_pair_accessible_boundary_selection(nonconvex_material_mesh, asset_tmp_path):  # noqa: F811
    vertices, tets, faces, _ = nonconvex_material_mesh
    elastic = asset_tmp_path / "accessible-elastic.npz"
    collision = asset_tmp_path / "accessible-material.obj"
    np.savez(elastic, vertices=vertices, tetrahedra=tets, surface_triangles=faces)
    topology = build_surface_topology(tets, faces)
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flipped = np.einsum("ij,ij->i", normals, vertices[topology.opposite_vertices] - corners[:, 0]) > 0
    oriented = faces.copy()
    oriented[flipped] = oriented[flipped, ::-1]
    trimesh.Trimesh(vertices, oriented, process=False).export(collision)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=False,
    )
    observed = scene.add_entity(
        morph=gs.morphs.URDF(
            file=mesh_urdf(collision),
            align=False,
            convexify=False,
            decimate=False,
            preserve_mesh_precision=True,
        ),
    )
    observed.base_link.configure_stress_recovery(
        gs.options.RigidStressOptions(
            mesh=elastic,
            method="pcg",
            update_mode="manual",
            quadrature=2,
        )
    )
    # The near pad lies behind the U's opposite wall. The farther pad is
    # accessible through its open end. A transverse pad blocks that route.
    centers = ((0.0033, 0.002, 0.0005), (0.0016, 0.005, 0.0005), (0.0015, 0.0035, 0.0005))
    sizes = ((0.0002, 0.004, 0.001), (0.0002, 0.0002, 0.001), (0.003, 0.0002, 0.001))
    opponents = []
    routes_blocked = []
    for use_mesh, blocked, reverse in (
        (False, False, False),
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ):
        indices = list(range(3 if blocked else 2))
        if reverse:
            indices.reverse()
        root = ET.Element("robot", name="candidate_pads")
        link = ET.SubElement(root, "link", name="pad")
        if use_mesh:
            meshes = []
            for index in indices:
                mesh = trimesh.creation.box(extents=sizes[index])
                mesh.apply_translation(centers[index])
                meshes.append(mesh)
            path = asset_tmp_path / f"accessible-pad-{len(opponents)}.obj"
            trimesh.util.concatenate(meshes).export(path)
            child = ET.SubElement(link, "collision")
            ET.SubElement(ET.SubElement(child, "geometry"), "mesh", filename=str(path))
        else:
            for index in indices:
                child = ET.SubElement(link, "collision")
                ET.SubElement(child, "origin", xyz=" ".join(map(str, centers[index])))
                ET.SubElement(ET.SubElement(child, "geometry"), "box", size=" ".join(map(str, sizes[index])))
        opponents.append(
            scene.add_entity(
                morph=gs.morphs.URDF(
                    file=ET.tostring(root, encoding="unicode"),
                    align=False,
                    convexify=False,
                    decimate=False,
                    preserve_mesh_precision=True,
                ),
            )
        )
        routes_blocked.append(blocked)
    scene.build(n_envs=2)
    solver = scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    sources = create_sources(*entry.contacts.valid.shape)
    associate(scene, entry, sources)
    n_pairs = 2 * len(opponents)
    pairs = StressPairState(
        environment=V(dtype=gs.qd_int, shape=(n_pairs,)),
        opponent_link=V(dtype=gs.qd_int, shape=(n_pairs,)),
        source_count=V(dtype=gs.qd_int, shape=(n_pairs,)),
        force=V_VEC(3, dtype=gs.qd_float, shape=(n_pairs,)),
        moment=V_VEC(3, dtype=gs.qd_float, shape=(n_pairs,)),
    )
    pairs.environment.from_numpy(np.tile(np.arange(2, dtype=gs.np_int), len(opponents)))
    pairs.opponent_link.from_numpy(
        np.repeat(np.array([entity.base_link.idx for entity in opponents], dtype=gs.np_int), 2)
    )
    shape = (entry.surface.info.positions.shape[0], n_pairs)
    samples = StressPairSamples(
        opponent_position=V_VEC(3, dtype=gs.qd_float, shape=shape),
        normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
        relative_velocity=V_VEC(3, dtype=gs.qd_float, shape=shape),
        gap=V(dtype=gs.qd_float, shape=shape),
        projected_area=V(dtype=gs.qd_float, shape=shape),
        opponent_geom=V(dtype=gs.qd_int, shape=shape),
        opponent_face=V(dtype=gs.qd_int, shape=shape),
        status=V(dtype=gs.qd_int, shape=shape),
    )
    epsilon = float(np.finfo(gs.np_float).eps)
    points = qd_to_numpy(entry.surface.info.positions, transpose=True, copy=True)
    selected = np.isclose(points[:, 0], 0.001, rtol=0, atol=1e-12) & (points[:, 1] > 0.001) & (points[:, 1] < 0.0022)
    assert selected.any()
    expected = points[selected].copy()
    expected[:, 0], expected[:, 1] = 0.0015, 0.0049
    delta = points[selected] - expected
    distance = np.linalg.norm(delta, axis=1)
    normal = delta / distance[:, None]
    weights = qd_to_numpy(entry.surface.info.weights, transpose=True)[selected]
    baseline = scene.get_state()
    rotation = Rotation.from_euler("xyz", (19, -27, 33), degrees=True)
    translation = np.array([0.03, -0.02, 0.04])
    for moved in (False, True):
        if moved:
            for entity in (observed, *opponents):
                entity.set_pos(translation, envs_idx=[0])
                entity.set_quat(rotation.as_quat()[[3, 0, 1, 2]], envs_idx=[0])
            associate(scene, entry, sources)
        kernel_pair_samples(
            entry.link.idx,
            solver.dyn_state,
            entry.contacts,
            pairs,
            samples,
            solver.dyn_info,
            entry.surface.info,
            epsilon,
        )
        status = qd_to_numpy(samples.status, transpose=True)
        targets = qd_to_numpy(samples.opponent_position, transpose=True)
        gaps = qd_to_numpy(samples.gap, transpose=True)
        directions = qd_to_numpy(samples.normal, transpose=True)
        areas = qd_to_numpy(samples.projected_area, transpose=True)
        for index, blocked in enumerate(routes_blocked):
            group = slice(2 * index, 2 * index + 2)
            if blocked:
                assert np.isin(
                    status[group, selected], (PairGeometryStatus.BACK_FACING, PairGeometryStatus.OCCLUDED)
                ).all(), (index, moved, targets[group, selected], gaps[group, selected])
                assert_equal(areas[group, selected], 0)
            else:
                assert_equal(status[group, selected], PairGeometryStatus.SUCCESS)
                assert_allclose(targets[group, selected], expected, atol=1e-12)
                assert_allclose(gaps[group, selected], distance, atol=1e-12)
                assert_allclose(directions[group, selected], normal, atol=1e-12)
                assert_allclose(areas[group, selected], -normal[:, 0] * weights, rtol=1e-12, atol=1e-18)
        if not moved:
            original_gaps = gaps.copy()
            original_status = status.copy()
        else:
            assert_equal(gaps[1::2], original_gaps[1::2])
            assert_equal(status[1::2], original_status[1::2])
    scene.reset(state=baseline, envs_idx=[0])
    associate(scene, entry, sources)
    kernel_pair_samples(
        entry.link.idx, solver.dyn_state, entry.contacts, pairs, samples, solver.dyn_info, entry.surface.info, epsilon
    )
    assert_equal(qd_to_numpy(samples.gap, transpose=True), original_gaps)
    assert_equal(qd_to_numpy(samples.status, transpose=True), original_status)
