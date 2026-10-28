import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.adaptive_pressure import kernel_adaptive_pressure
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_scatter
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace, kernel_scatter_faces
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("tasks_per_env", (1, 32))
def test_corner_pressure_cooperative_field_and_wrench(tmp_path, tasks_per_env):
    vertices = np.indices((2, 2, 2)).reshape((3, -1)).T * 0.001
    tetrahedra = np.array([[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7], [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]])
    local_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    faces, counts = np.unique(np.sort(tetrahedra[:, local_faces].reshape((-1, 3)), axis=1), axis=0, return_counts=True)
    mesh = tmp_path / "cube.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces[counts == 1])
    options = RigidStressOptions(
        mesh=mesh,
        output_mode="full",
        pressure_quadrature="adaptive",
    )
    model = StressModel(options)
    surface = StressSurface(16, model.info, adaptive=True)
    state = model.create_state(2, "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    omega.fill(0)
    errno = V(dtype=gs.qd_int, shape=(2,))
    coordinates = qd_to_numpy(model.info.vertices, transpose=True)
    point = np.array([0.00008, 0.00011, 0.001])
    forces = np.array([[[-0.0425, -0.0325, -0.05], [-0.085, -0.065, -0.1]]])
    serial_fields = None
    for cooperative in (False, True):
        contacts = create_contacts(1, 2, 0.00085, hertz_profile=True, adaptive=True)
        contacts.position.from_numpy(np.broadcast_to(point, (1, 2, 3)).copy())
        contacts.normal.from_numpy(np.broadcast_to([0.0, 0.0, -1.0], (1, 2, 3)).copy())
        contacts.force.from_numpy(forces)
        contacts.friction.fill(1.2)
        contacts.valid.fill(True)
        reuse = cooperative and gs.backend == gs.cuda
        workspace = create_scatter_workspace(2, tasks_per_env, reuse_moments=reuse, adaptive=True)
        errno.fill(0)
        kernel_begin_step(state)
        kernel_anchor(float(np.finfo(gs.np_float).eps), contacts, surface.info, cooperative)
        kernel_adaptive_pressure(contacts, state, model.info, surface.info, workspace, cooperative, True, reuse)
        if reuse:
            kernel_scatter_faces(contacts, state, model.info, surface.info, workspace, True, True)
        else:
            kernel_scatter(contacts, state, model.info, surface.info, cooperative=False)
        nodal_force = qd_to_numpy(state.force, transpose=True, copy=True)
        assert_allclose(nodal_force.sum(axis=1), forces[0], rtol=1e-8, atol=0.0)
        moment = np.cross(coordinates[None] - point, nodal_force).sum(axis=1)
        assert (np.linalg.norm(moment, axis=1) <= 1e-8 * np.linalg.norm(forces[0], axis=1) * 0.00085).all()
        model.recover(omega, state, options, surface_load=True)
        kernel_accept(state, contacts, errno)
        assert_equal(qd_to_numpy(errno), 0)
        fields = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
        assert np.isfinite(fields).all()
        assert_allclose(fields[1], 2.0 * fields[0], rtol=1e-9, atol=1e-7)
        if serial_fields is None:
            serial_fields = fields
        else:
            assert_allclose(fields, serial_fields, rtol=1e-9, atol=1e-7)
