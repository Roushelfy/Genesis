from itertools import permutations, product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.array_class import ContactWitnessKind
from genesis.utils.misc import qd_to_numpy, tensor_to_array


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_collision_witness_octagonal_manifold(tmp_path):
    vertices = np.array(tuple(product((0, 1), repeat=3)), dtype=np.float64) * 0.01 - 0.005
    tetrahedra = []
    for axes in permutations(range(3)):
        corner = np.zeros(3, dtype=np.int32)
        element = [0]
        for axis in axes:
            corner[axis] = 1
            element.append(int(corner @ np.array((4, 2, 1))))
        tetrahedra.append(element)
    tetrahedra = np.array(tetrahedra)
    faces = np.sort(tetrahedra[:, ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))].reshape(-1, 3), axis=1)
    unique, count = np.unique(faces, axis=0, return_counts=True)
    elastic = tmp_path / "cube.npz"
    np.savez(elastic, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=unique[count == 1])
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0, 0, 0)),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=False, use_gjk_collision=True, enable_contact_patch=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Box(size=(0.01, 0.01, 0.01), fixed=True))
    moving = scene.add_entity(gs.morphs.Box(size=(0.01, 0.01, 0.01), pos=(0, 0, 0.0099), euler=(0, 0, 45)))
    moving.base_link.configure_stress_recovery(gs.options.RigidStressOptions(mesh=elastic))
    scene.build(n_envs=2)
    collider = scene.rigid_solver.collider
    collider.detection()
    scene.rigid_solver.check_errno()
    state = collider.collider_state
    counts = qd_to_numpy(state.n_contacts, transpose=True, copy=True)
    np.testing.assert_array_equal(counts, 8)
    order = qd_to_numpy(state.contact_sort_idx, transpose=True, copy=True)
    witness = state.contact_data.witness
    point_a = qd_to_numpy(witness.point_a, transpose=True, copy=True)
    point_b = qd_to_numpy(witness.point_b, transpose=True, copy=True)
    methods = qd_to_numpy(witness.method, transpose=True, copy=True)
    positions = qd_to_numpy(state.contact_data.pos, transpose=True, copy=True)
    normals = qd_to_numpy(state.contact_data.normal, transpose=True, copy=True)
    penetrations = qd_to_numpy(state.contact_data.penetration, transpose=True, copy=True)
    geom_positions = tensor_to_array(scene.rigid_solver.get_geoms_pos())
    geom_quaternions = tensor_to_array(scene.rigid_solver.get_geoms_quat())
    corner = 0.005 * (np.sqrt(2.0) - 1.0)
    expected_xy = np.array(tuple(product((-0.005, 0.005), (-corner, corner))))
    expected_xy = np.concatenate((expected_xy, expected_xy[:, ::-1]))
    expected_xy = expected_xy[np.lexsort((expected_xy[:, 1], expected_xy[:, 0]))]
    for i_b, count in enumerate(counts):
        slots = order[i_b, :count]
        np.testing.assert_array_equal(methods[i_b, slots], ContactWitnessKind.GJK)
        np.testing.assert_allclose(
            positions[i_b, slots], (point_a[i_b, slots] + point_b[i_b, slots]) * 0.5, rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            point_b[i_b, slots] - point_a[i_b, slots],
            penetrations[i_b, slots, None] * normals[i_b, slots],
            rtol=1e-12,
            atol=1e-15,
        )
        for field, points in ((state.contact_data.geom_a, point_a), (state.contact_data.geom_b, point_b)):
            geoms = qd_to_numpy(field, transpose=True, copy=True)
            for geom in scene.rigid_solver.geoms:
                selected = slots[geoms[i_b, slots] == geom.idx]
                if len(selected):
                    rotation = Rotation.from_quat(geom_quaternions[i_b, geom.idx, [1, 2, 3, 0]])
                    local = rotation.inv().apply(points[i_b, selected] - geom_positions[i_b, geom.idx])
                    assert (np.abs(local) <= 0.005 + 2e-10).all()
                    np.testing.assert_allclose(np.abs(local).max(axis=1), 0.005, rtol=0.0, atol=2e-10)
        xy = point_a[i_b, slots, :2]
        canonical = np.lexsort((xy[:, 1].round(12), xy[:, 0].round(12)))
        np.testing.assert_allclose(xy[canonical], expected_xy, rtol=0.0, atol=2e-10)
