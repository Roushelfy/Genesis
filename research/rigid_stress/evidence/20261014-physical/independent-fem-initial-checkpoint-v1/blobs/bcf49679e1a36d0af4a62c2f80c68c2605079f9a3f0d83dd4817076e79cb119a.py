import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.collider.contact import func_set_contact
from genesis.utils import array_class, geom


@qd.kernel
def store_contact_frames(
    raw: qd.types.ndarray(),
    output: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    errno: qd.Tensor,
):
    for environment in range(raw.shape[0]):
        normal = qd.Vector([raw[environment, axis] for axis in qd.static(range(3))])
        func_set_contact(
            0,
            1,
            environment,
            0,
            0,
            normal,
            qd.Vector.zero(gs.qd_float, 3),
            0.001,
            dyn_state,
            collider_state,
            dyn_info,
            rigid_info,
            collider_info,
            errno,
        )
        normal = collider_state.contact_data.normal[0, environment]
        first, second = geom.qd_orthogonals(normal)
        for axis in qd.static(range(3)):
            output[environment, axis] = normal[axis]
        output[environment, 3] = normal.dot(first)
        output[environment, 4] = normal.dot(second)
        output[environment, 5] = first.dot(second)
        output[environment, 6] = first.norm()
        output[environment, 7] = second.norm()
        for angle in qd.static(range(4)):
            force = 2.0 * normal + 1.6 * (qd.cos(0.47 * angle) * first + qd.sin(0.47 * angle) * second)
            normal_force = force.dot(normal)
            output[environment, 8 + angle] = (force - normal_force * normal).norm() - 0.8 * normal_force


@pytest.mark.required
@pytest.mark.precision("64")
def test_contact_storage_preserves_unit_friction_frame():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Sphere(radius=0.01, pos=(0, 0, 0.011)))
    scene.build(n_envs=3)
    raw = np.array(
        [
            [0.04461867865796483, -0.9073095057226495, 0.41810135784069513],
            [0.12, 0.16, 0],
            [0, 0, 7.0],
        ]
    )
    output = np.empty((3, 12))
    solver, collider = scene.rigid_solver, scene.rigid_solver.collider
    store_contact_frames(
        raw,
        output,
        solver.dyn_state,
        collider.collider_state,
        solver.dyn_info,
        solver.rigid_info,
        collider.collider_info,
        solver.data_manager.errno,
    )
    np.testing.assert_allclose(output[:, :3], raw / np.linalg.norm(raw, axis=1)[:, None], atol=1e-14, rtol=0)
    np.testing.assert_allclose(output[:, 3:6], 0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(output[:, 6:8], 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(output[:, 8:], 0, atol=1e-14, rtol=0)
