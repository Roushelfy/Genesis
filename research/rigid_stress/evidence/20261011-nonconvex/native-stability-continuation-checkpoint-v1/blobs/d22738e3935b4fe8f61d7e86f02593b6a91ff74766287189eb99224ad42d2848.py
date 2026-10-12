import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.footprint import (
    create_footprints,
    create_surface_fan_tractions,
    kernel_footprint_ports,
    kernel_surface_fan_profiles,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.provenance import (
    StressSourceStatus,
    create_sources,
    kernel_capture_rigid_sources,
    kernel_resolve_sources,
)
from genesis.engine.solvers.rigid.stress.source_vertices import (
    StressSourceVertexInfo,
    StressSourceVertexStatus,
    kernel_bind_source_vertices,
    kernel_source_vertex_mapping,
)
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.engine.solvers.rigid.stress.traction import create_traction_info
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
def test_shared_edge_roundoff_preserves_finite_profile():
    mesh = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1/elastic.npz"
    with np.load(Path(__file__).with_name("assets") / "nonconvex_shared_edge_roundoff.npz") as asset:
        saved = {key: asset[key].copy() for key in asset.files}
    model = StressModel(RigidStressOptions(mesh=mesh, method="direct"))
    n_envs, n_sources = saved["force"].shape[:2]
    surface, paths = StressSurface(10, model.info), StressSurfacePaths(model.info, 8)
    contacts, sources = create_contacts(n_sources, n_envs, 0.006), create_sources(n_sources, n_envs)
    for name, key in (("position", "point"), ("force", "force"), ("radius", "radius"), ("friction", "friction")):
        getattr(contacts, name).from_numpy(np.moveaxis(saved[key], 0, 1).copy())
    contacts.valid.fill(True)
    contacts.frame_position.from_numpy(saved["frame_position"])
    sources.geom.fill(24)
    sources.pair.fill(1)
    for name in ("witness", "uncertainty", "free_moment"):
        getattr(sources, name).from_numpy(np.moveaxis(saved[name], 0, 1).copy())
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-5, epsilon, contacts, sources, model.info, surface.info)
    assert_equal(qd_to_numpy(sources.status), StressSourceStatus.SUCCESS)
    anchors = qd_to_numpy(contacts.center, transpose=True)
    assert_allclose(anchors, saved["witness"], rtol=0, atol=2e-16)
    moment = qd_to_numpy(sources.anchor_moment, transpose=True, copy=True)
    moment += np.cross(anchors - saved["point"], saved["force"])
    assert_allclose(moment, saved["free_moment"], rtol=0, atol=1e-20)
    footprints = create_footprints(paths.info.positions.shape[0], n_sources, n_envs)
    tractions = create_surface_fan_tractions(surface.info, n_sources, n_envs)
    kernel_footprint_ports(contacts, sources, footprints, surface.info, paths.info)
    kernel_surface_fan_profiles(
        contacts, footprints, tractions, model.info, surface.info, paths.info, create_traction_info(surface.info), epsilon
    )
    profile = qd_to_numpy(tractions.profile, transpose=True)
    indices = qd_to_numpy(tractions.sample_indices, transpose=True)
    counts = qd_to_numpy(tractions.sample_count, transpose=True)
    # Atomic packing order varies. A roundoff-sized fan may also contribute negligible new samples;
    # compare the physical integration weights on the union of their shared sample identities.
    for source in range(n_sources):
        labels = np.union1d(indices[0, : counts[0, source], source], indices[1, : counts[1, source], source])
        weights = np.zeros((n_envs, len(labels)))
        for environment in range(n_envs):
            count = counts[environment, source]
            weights[environment, np.searchsorted(labels, indices[environment, :count, source])] = profile[
                environment, :count, source
            ]
        assert_allclose(weights[0], weights[1], rtol=1e-10, atol=1e-18)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_source_uncertainty_on_same_surface(nonconvex_material_mesh, asset_tmp_path):  # noqa: F811
    vertices, tetrahedra, triangles, shape = nonconvex_material_mesh
    mesh = asset_tmp_path / f"source-{shape}.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=triangles)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    sampled = StressSurface(4, model.info)
    anchor = np.array([0.001, 0.00237 if shape == "u" else 0.00137, 0.00042 if shape == "u" else 0.00142])
    corner = np.array([0.001, 0.003, 0.0] if shape == "u" else [0.001, 0.001, 0.001])
    witnesses = np.stack((anchor, anchor, corner))[None]
    points = witnesses + [0.00003, 0.00001, -0.00002]
    force = np.broadcast_to([-0.001, 0.0002, -0.0001], witnesses.shape).copy()
    moment = np.broadcast_to([1e-8, -2e-8, 3e-8], witnesses.shape).copy()
    contacts = create_contacts(1, 3, 0.0015)
    sources = create_sources(1, 3)
    contacts.position.from_numpy(points)
    contacts.force.from_numpy(force)
    contacts.valid.fill(True)
    sources.geom.fill(7)
    sources.pair.fill(11)
    sources.witness.from_numpy(witnesses)
    sources.free_moment.from_numpy(moment)
    sources.uncertainty.from_numpy(np.array([[0.00008, 0.00055, 0.0]]))
    # The anchor is 35 micrometres from a same-wall triangle edge. An 80 micrometre
    # uncertainty includes that edge, while 550 micrometres also reaches another branch.
    kernel_resolve_sources(0.0006, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, sampled.info)
    assert_equal(
        qd_to_numpy(sources.status, transpose=True)[:, 0],
        [StressSourceStatus.SUCCESS, StressSourceStatus.AMBIGUOUS, StressSourceStatus.SUCCESS],
    )
    centers = qd_to_numpy(contacts.center, transpose=True)[:, 0]
    assert_allclose(centers[[0, 2]], witnesses[0, [0, 2]], atol=1e-15)
    assert np.isnan(centers[1]).all()
    recovered_moment = qd_to_numpy(sources.anchor_moment, transpose=True)[[0, 2], 0]
    recovered_moment += np.cross(centers[[0, 2]] - points[0, [0, 2]], force[0, [0, 2]])
    assert_allclose(recovered_moment, moment[0, [0, 2]], rtol=0.0, atol=1e-22)
    # A known collision vertex differs slightly from its unique reference
    # feature. Bind that geom/vertex correspondence rather than conflating
    # neighboring face projections or snapping general nearby witnesses.
    info = StressSourceVertexInfo(
        geom=V(dtype=gs.qd_int, shape=(4,)),
        positions=V_VEC(3, dtype=gs.qd_float, shape=(4,)),
        reference_vertex=V(dtype=gs.qd_int, shape=(4,)),
        reference_face=V(dtype=gs.qd_int, shape=(4,)),
        barycentric=V_VEC(3, dtype=gs.qd_float, shape=(4,)),
        offset=V(dtype=gs.qd_float, shape=(4,)),
        status=V(dtype=gs.qd_int, shape=(4,)),
    )
    collision_vertex = corner + [6.8e-10, 0.0, 0.0]
    node_positions = qd_to_numpy(model.info.vertices, transpose=True)
    first_face_nodes = qd_to_numpy(model.info.surface_nodes, transpose=True)[:, 0]
    edge_positions = node_positions[first_face_nodes[3:]]
    info.geom.fill(7)
    info.positions.from_numpy(np.concatenate((collision_vertex[None], edge_positions + [6.8e-10, 0.0, 0.0])))
    kernel_source_vertex_mapping(model.info, info, 1e-8)
    assert_equal(qd_to_numpy(info.status, transpose=True), StressSourceVertexStatus.SUCCESS)
    matched_faces = qd_to_numpy(info.reference_face, transpose=True)
    matched_bary = qd_to_numpy(info.barycentric, transpose=True)
    matched = np.einsum("va,vai->vi", matched_bary, vertices[triangles[matched_faces]])
    assert_allclose(matched[1:], edge_positions, rtol=0.0, atol=1e-15)
    vertex_witnesses = np.stack((collision_vertex, collision_vertex + [0.0, 1e-7, 0.0], collision_vertex))[None]
    sources.witness.from_numpy(vertex_witnesses)
    sources.geom.from_numpy(np.array([[7, 7, 8]], dtype=gs.np_int))
    sources.witness_face.fill(-1)
    sources.uncertainty.fill(0.0)
    contacts.frame_position.fill(0.0)
    kernel_bind_source_vertices(contacts, sources, info, float(np.finfo(gs.np_float).eps))
    witness_faces = qd_to_numpy(sources.witness_face, transpose=True)
    assert witness_faces[0, 0] >= 0
    assert_equal(witness_faces[1:, 0], -1)
    kernel_resolve_sources(1e-8, float(np.finfo(gs.np_float).eps), contacts, sources, model.info, sampled.info)
    assert_equal(qd_to_numpy(sources.status, transpose=True)[0, 0], StressSourceStatus.SUCCESS)
    bound = qd_to_numpy(contacts.center, transpose=True)[0, 0]
    assert_equal(bound, corner)
    recovered = qd_to_numpy(sources.anchor_moment, transpose=True)[0, 0].copy()
    recovered += np.cross(bound - points[0, 0], force[0, 0])
    assert_allclose(recovered, moment[0, 0], rtol=0.0, atol=1e-22)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("stress_side", (0, 1))
def test_native_rigid_source_frame_and_contact_transitions(asset_tmp_path, stress_side):
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))) * 0.01
    triangles = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    mesh = asset_tmp_path / "source-tetra.npz"
    collision = asset_tmp_path / "source-tetra.obj"
    np.savez(mesh, vertices=vertices, tetrahedra=np.array(((0, 1, 2, 3),)), surface_triangles=triangles)
    trimesh.Trimesh(vertices, triangles, process=False).export(collision)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
        ),
        show_viewer=False,
    )
    first = scene.add_entity(
        gs.morphs.Mesh(
            file=collision,
            convexify=True,
            decimate=False,
        ),
        vis_mode="collision",
    )
    second = scene.add_entity(
        gs.morphs.Mesh(
            file=collision,
            pos=(0.003, 0.002, 0.001),
            euler=(8, -6, 10),
            convexify=True,
            decimate=False,
        ),
        vis_mode="collision",
    )
    target = (first, second)[stress_side]
    target.base_link.configure_stress_recovery(RigidStressOptions(mesh=mesh))
    scene.build(n_envs=2)
    first.set_pos([[0.03, -0.04, 0.02], [-0.02, 0.01, -0.03]])
    second.set_pos([[0.033, -0.038, 0.021], [-0.017, 0.012, -0.029]])
    solver = scene.rigid_solver
    collider = solver.collider
    state = collider.collider_state
    entry = solver.stress_recovery.links[0]
    sources = create_sources(entry.contacts.valid.shape[0], 2)
    previous = None
    for phase in range(3):
        if phase == 1:
            target.set_pos([0.1, 0.1, 0.1], envs_idx=[0])
        elif phase == 2:
            scene.reset(envs_idx=[0])
        collider.detection()
        solver.check_errno()
        counts = qd_to_numpy(state.n_contacts, transpose=True, copy=True)
        assert_equal(counts > 0, [phase != 1, True])
        normal = qd_to_numpy(state.contact_data.normal, transpose=True, copy=True)
        # A kinematic source check with declared normal/tangential load input;
        # no live stress recovery or dynamics feedback is inferred from this fixture.
        rigid_force = -0.001 * normal + 0.00015 * np.cross(normal, [0, 0, 1])
        state.contact_data.force.from_numpy(np.moveaxis(rigid_force, 0, 1))
        kernel_associate(
            target.base_link.idx,
            solver._links_offset_pos,
            solver._links_offset_quat,
            entry.omega,
            solver.dyn_state,
            entry.contacts,
            state,
            solver._links_offset_quat.ndim == 3,
            True,
            entry.state.step_valid,
            solver._errno,
        )
        kernel_capture_rigid_sources(target.base_link.idx, entry.contacts, sources, state)
        kernel_resolve_sources(
            1e-8, float(np.finfo(gs.np_float).eps), entry.contacts, sources, entry.model.info, entry.surface.info
        )
        solver.check_errno()
        valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)
        centers = qd_to_numpy(entry.contacts.center, transpose=True, copy=True)
        moments = qd_to_numpy(sources.anchor_moment, transpose=True, copy=True)
        statuses = qd_to_numpy(sources.status, transpose=True, copy=True)
        assert_equal(statuses[valid], StressSourceStatus.SUCCESS)
        faces = qd_to_numpy(entry.contacts.anchor_face, transpose=True, copy=True)
        barycentric = qd_to_numpy(sources.barycentric, transpose=True, copy=True)
        order = qd_to_numpy(state.contact_sort_idx, transpose=True, copy=True)
        geom_a = qd_to_numpy(state.contact_data.geom_a, transpose=True, copy=True)
        link_a = qd_to_numpy(state.contact_data.link_a, transpose=True, copy=True)
        geom_b = qd_to_numpy(state.contact_data.geom_b, transpose=True, copy=True)
        point_a = qd_to_numpy(state.contact_data.witness.point_a, transpose=True, copy=True)
        point_b = qd_to_numpy(state.contact_data.witness.point_b, transpose=True, copy=True)
        positions = qd_to_numpy(state.contact_data.pos, transpose=True, copy=True)
        source_geoms = qd_to_numpy(sources.geom, transpose=True, copy=True)
        source_pairs = qd_to_numpy(sources.pair, transpose=True, copy=True)
        pairs = qd_to_numpy(state.contact_data.pair_idx, transpose=True, copy=True)
        local_positions = qd_to_numpy(entry.contacts.position, transpose=True, copy=True)
        local_forces = qd_to_numpy(entry.contacts.force, transpose=True, copy=True)
        reference_pos = tensor_to_array(target.base_link.get_pos())
        reference_quat = tensor_to_array(target.base_link.get_quat())
        for i_b, count in enumerate(counts):
            if count:
                slots = order[i_b, :count]
                side_a = link_a[i_b, slots] == target.base_link.idx
                points = np.where(side_a[:, None], point_a[i_b, slots], point_b[i_b, slots])
                signs = np.where(side_a, -1.0, 1.0)
                rotation = Rotation.from_quat(reference_quat[i_b, [1, 2, 3, 0]])
                expected_anchor = rotation.inv().apply(points - reference_pos[i_b])
                expected_point = rotation.inv().apply(positions[i_b, slots] - reference_pos[i_b])
                expected_force = rotation.inv().apply(signs[:, None] * rigid_force[i_b, slots])
                assert_allclose(centers[i_b, :count], expected_anchor, rtol=0.0, atol=1e-15)
                assert_allclose(local_positions[i_b, :count], expected_point, rtol=0.0, atol=1e-15)
                assert_allclose(local_forces[i_b, :count], expected_force, rtol=0.0, atol=1e-17)
                assert_allclose(
                    moments[i_b, :count] + np.cross(centers[i_b, :count] - expected_point, expected_force),
                    0.0,
                    rtol=0.0,
                    atol=1e-20,
                )
                assert_equal(source_geoms[i_b, :count], np.where(side_a, geom_a[i_b, slots], geom_b[i_b, slots]))
                assert_equal(source_pairs[i_b, :count], pairs[i_b, slots])
                mapped = np.einsum("ca,cai->ci", barycentric[i_b, :count], vertices[triangles[faces[i_b, :count]]])
                assert_allclose(mapped, centers[i_b, :count], rtol=0.0, atol=1e-15)
        if phase == 1:
            assert np.isnan(centers[0]).all() and np.isnan(moments[0]).all()
            assert_equal(centers[1], previous[0][1])
            assert_equal(moments[1], previous[1][1])
        previous = centers, moments
from pathlib import Path

