import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.topology import build_surface_topology
from genesis.utils.array_class import ContactWitnessKind, DataKind, fill_data
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("use_gjk,patch", ((False, False), (True, False), (True, True)))
def test_native_collision_witnesses_and_lifecycle(tmp_path, use_gjk, patch):
    vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))) * 0.01
    triangles = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    elastic = tmp_path / "tetra.npz"
    collision = tmp_path / "tetra.obj"
    np.savez(elastic, vertices=vertices, tetrahedra=np.array(((0, 1, 2, 3),)), surface_triangles=triangles)
    trimesh.Trimesh(vertices, triangles, process=False).export(collision)
    captures = []
    captured_counts = []
    for save_witness in (False, True):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(gravity=(0, 0, 0)),
            rigid_options=gs.options.RigidOptions(use_gjk_collision=use_gjk, enable_contact_patch=patch),
            show_viewer=False,
        )
        moving = scene.add_entity(gs.morphs.Mesh(file=collision, convexify=True, decimate=False))
        scene.add_entity(
            gs.morphs.Mesh(file=collision, convexify=True, decimate=False, fixed=True, pos=(0.003, 0.002, 0.001))
        )
        scene.add_entity(
            gs.morphs.Mesh(
                file=collision,
                convexify=True,
                decimate=False,
                fixed=True,
                pos=(-0.001, 0.002, 0.003),
                euler=(8, -6, 10),
            )
        )
        if save_witness:
            moving.base_link.configure_stress_recovery(gs.options.RigidStressOptions(mesh=elastic))
        scene.build(n_envs=3)
        moving.set_pos(np.array(((0, 0, 0), (0.0005, -0.0003, 0.0001), (-0.0002, 0.0004, 0.0003))))
        collider = scene.rigid_solver.collider
        collider.detection()
        scene.rigid_solver.check_errno()
        state = collider.collider_state
        counts = qd_to_numpy(state.n_contacts, transpose=True, copy=True)
        captured_counts.append(counts)
        order = qd_to_numpy(state.contact_sort_idx, transpose=True, copy=True)
        assert (counts >= 2).all()
        fields = (
            state.contact_data.geom_a,
            state.contact_data.geom_b,
            state.contact_data.pos,
            state.contact_data.normal,
            state.contact_data.penetration,
            state.contact_data.pair_idx,
        )
        capture = [qd_to_numpy(field, transpose=True, copy=True) for field in fields]
        # GPU atomics can reserve different raw slots for independent pairs.
        # Align every active contact by its identity and position before comparing scenes.
        for i_b, count in enumerate(counts):
            slots = order[i_b, :count]
            position = capture[2][i_b, slots]
            canonical = np.lexsort(
                (
                    position[:, 2],
                    position[:, 1],
                    position[:, 0],
                    capture[5][i_b, slots],
                    capture[1][i_b, slots],
                    capture[0][i_b, slots],
                )
            )
            order[i_b, :count] = slots[canonical]
        captures.append([field[np.arange(3)[:, None], order] for field in capture])
        witness = state.contact_data.witness
        assert (len(witness.point_a.shape) > 0) == save_witness
        if save_witness:
            points_a = qd_to_numpy(witness.point_a, transpose=True, copy=True)
            points_b = qd_to_numpy(witness.point_b, transpose=True, copy=True)
            methods = qd_to_numpy(witness.method, transpose=True, copy=True)
            geoms_pos = tensor_to_array(scene.rigid_solver.get_geoms_pos())
            geoms_quat = tensor_to_array(scene.rigid_solver.get_geoms_quat())
            for i_b in range(3):
                slots = order[i_b, : counts[i_b]]
                assert (methods[i_b, slots] != ContactWitnessKind.NORMAL_OFFSET).all()
                if use_gjk:
                    np.testing.assert_array_equal(methods[i_b, slots], ContactWitnessKind.GJK)
                assert np.isfinite(points_a[i_b, slots]).all() and np.isfinite(points_b[i_b, slots]).all()
                for side, points in ((0, points_a), (1, points_b)):
                    for geom in scene.rigid_solver.geoms:
                        selected = slots[capture[side][i_b, slots] == geom.idx]
                        if len(selected):
                            rotation = Rotation.from_quat(geoms_quat[i_b, geom.idx, [1, 2, 3, 0]])
                            local = rotation.inv().apply(points[i_b, selected] - geoms_pos[i_b, geom.idx])
                            _, distance, _ = trimesh.proximity.closest_point_naive(
                                trimesh.Trimesh(geom.init_verts, geom.init_faces, process=False), local
                            )
                            exact = methods[i_b, selected] != ContactWitnessKind.MPR_APPROXIMATE
                            np.testing.assert_allclose(distance[exact], 0.0, atol=2e-10)
            items = tuple(
                item for item in collider.data if item.kind == DataKind.STATE and ".contact_data." in item.name
            )
            assert any("witness.point_a" in item.name for item in items)
            # Records use native storage order, unlike public batched getters.
            record = {item.name: qd_to_torch(item.value, copy=True) for item in items}
            scene.reset(envs_idx=[1])
            for field, previous in ((witness.point_a, points_a), (witness.point_b, points_b)):
                actual = qd_to_numpy(field, transpose=True)
                np.testing.assert_array_equal(actual[[0, 2]], previous[[0, 2]])
                np.testing.assert_array_equal(actual[1], 0.0)
            fill_data(items, record)
            np.testing.assert_array_equal(qd_to_numpy(witness.point_a, transpose=True), points_a)
            np.testing.assert_array_equal(qd_to_numpy(witness.point_b, transpose=True), points_b)
            np.testing.assert_array_equal(qd_to_numpy(witness.method, transpose=True), methods)
            collider.clear()
            np.testing.assert_array_equal(qd_to_numpy(witness.point_a, transpose=True), 0.0)
            np.testing.assert_array_equal(qd_to_numpy(witness.point_b, transpose=True), 0.0)
    np.testing.assert_array_equal(*captured_counts)
    rounding = 32.0 * np.finfo(gs.np_float).eps
    for i_field, (without, with_witness) in enumerate(zip(*captures)):
        # Only active contacts define a rigid load; unused slots need not retain a sorting permutation.
        for i_b, count in enumerate(counts):
            if i_field in (0, 1, 5):
                np.testing.assert_array_equal(without[i_b, :count], with_witness[i_b, :count])
            else:
                # Added stores can change fast-math instruction scheduling by one FP64 rounding unit.
                scale = 1.0 if i_field == 3 else 0.01
                np.testing.assert_allclose(
                    without[i_b, :count], with_witness[i_b, :count], rtol=rounding, atol=rounding * scale
                )


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_sdf_emitting_vertex_provenance(nonconvex_material_mesh, tmp_path):  # noqa: F811
    vertices, tetrahedra, triangles, shape = nonconvex_material_mesh
    elastic = tmp_path / "nonconvex.npz"
    collision = tmp_path / "nonconvex.obj"
    np.savez(elastic, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=triangles)
    topology = build_surface_topology(tetrahedra, triangles)
    positions = vertices[triangles]
    normals = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    flip = (normals * (vertices[topology.opposite_vertices] - positions[:, 0])).sum(axis=1) > 0
    collision_faces = triangles.copy()
    collision_faces[flip] = collision_faces[flip, ::-1]
    mesh = trimesh.Trimesh(vertices, collision_faces, process=False)
    mesh.export(collision)
    scene = gs.Scene(show_viewer=False)
    target = scene.add_entity(
        gs.morphs.Mesh(file=collision, convexify=False, decimate=False),
        material=gs.materials.Rigid(sdf_cell_size=0.0001, sdf_min_res=16, sdf_max_res=64),
    )
    position = (0.0011, 0.002, 0.0005) if shape == "u" else (0.0011, 0.0015, 0.0015)
    scene.add_entity(gs.morphs.Box(size=(0.0004, 0.0004, 0.0004), pos=position, fixed=True))
    target.base_link.configure_stress_recovery(gs.options.RigidStressOptions(mesh=elastic))
    scene.build(n_envs=2)
    collider = scene.rigid_solver.collider
    collider.detection()
    scene.rigid_solver.check_errno()
    state = collider.collider_state
    counts = qd_to_numpy(state.n_contacts, transpose=True)
    order = qd_to_numpy(state.contact_sort_idx, transpose=True)
    witness = state.contact_data.witness
    methods = qd_to_numpy(witness.method, transpose=True)
    source_indices = qd_to_numpy(witness.vertex_a, transpose=True)
    source_points = qd_to_numpy(witness.point_a, transpose=True)
    geom_indices = qd_to_numpy(state.contact_data.geom_a, transpose=True)
    geom_positions = tensor_to_array(scene.rigid_solver.get_geoms_pos())
    geom_quaternions = tensor_to_array(scene.rigid_solver.get_geoms_quat())
    assert (counts > 0).all()
    for i_b in range(2):
        slots = order[i_b, : counts[i_b]]
        np.testing.assert_array_equal(methods[i_b, slots], ContactWitnessKind.SDF_VERTEX)
        for geom in scene.rigid_solver.geoms:
            selected = slots[geom_indices[i_b, slots] == geom.idx]
            if len(selected):
                local_indices = source_indices[i_b, selected] - geom.vert_start
                assert ((0 <= local_indices) & (local_indices < geom.n_verts)).all()
                rotation = Rotation.from_quat(geom_quaternions[i_b, geom.idx, [1, 2, 3, 0]])
                expected = rotation.apply(geom.init_verts[local_indices]) + geom_positions[i_b, geom.idx]
                np.testing.assert_allclose(source_points[i_b, selected], expected, rtol=0.0, atol=1e-15)
