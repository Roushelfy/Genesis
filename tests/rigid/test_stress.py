import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.mechanics import P2Shell, shell_mesh
from research.rigid_stress.peak_cpu import P2Peak


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_p2_shared_operators(tmp_path):
    vertices, tetrahedra, surface, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    options = RigidStressOptions(mesh=mesh)
    model = StressModel(options)
    oracle = P2Shell(vertices, tetrahedra, surface, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    columns = qd_to_numpy(model.info.columns)
    rows = qd_to_numpy(model.info.row_start)
    stiffness = sparse.bsr_matrix((qd_to_numpy(model.info.stiffness), columns, rows)).tocsr()
    mass = sparse.kron(sparse.csr_matrix((qd_to_numpy(model.info.mass), columns, rows)), sparse.eye(3)).tocsr()
    difference = stiffness * options.young - oracle.k
    assert np.max(np.abs(difference.data)) / np.max(np.abs(oracle.k.data)) < 2e-13
    difference = mass - oracle.m
    assert np.max(np.abs(difference.data)) / np.max(np.abs(oracle.m.data)) < 2e-13
    np.testing.assert_allclose(qd_to_numpy(model.info.mass_modes).reshape((-1, 6)), oracle.mr, atol=1e-17)
    np.testing.assert_allclose(qd_to_numpy(model.info.gram), oracle.gram, atol=1e-17)
    gram_inverse = qd_to_numpy(model.info.gram_inverse)
    np.testing.assert_allclose(gram_inverse @ oracle.gram, np.eye(6), atol=2e-12)
    pins = qd_to_numpy(model.info.pins)
    assert len(np.unique(pins)) == 6
    assert np.linalg.matrix_rank(oracle.r[pins]) == 6


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_batched_recovery_against_full_fp64_direct(tmp_path):
    vertices, tetrahedra, surface, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    model = StressModel(RigidStressOptions(mesh=mesh, tolerance=1e-7, max_iterations=4000))
    oracle = P2Shell(vertices, tetrahedra, surface, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    n_envs = 5
    state = model.create_state(n_envs)
    random = np.random.default_rng(8231)
    force = random.normal(size=(len(oracle.xyz), n_envs, 3)) * 0.01
    force[:, 0] = 0.0
    angular_velocity = random.normal(size=(n_envs, 3)) * 2.0
    angular_velocity[0] = 0.0
    state.force.from_numpy(force)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.from_numpy(angular_velocity)
    model.recover(omega, state)
    rhs = qd_to_numpy(state.rhs, transpose=True).reshape((n_envs, -1)).T
    relative = oracle.xyz - oracle.com
    centrifugal = -np.cross(angular_velocity[:, None], np.cross(angular_velocity[:, None], relative[None]))
    raw = force.transpose((1, 0, 2)).reshape((n_envs, -1)).T + oracle.m @ centrifugal.reshape((n_envs, -1)).T
    expected_rhs = raw - oracle.mr @ np.linalg.solve(oracle.gram, oracle.r.T @ raw)
    np.testing.assert_allclose(rhs, expected_rhs, atol=2e-16)
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    displacement = np.zeros_like(rhs)
    displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    peak_scan = P2Peak(oracle.glambda, oracle.elements, model.options.young / (2.0 * (1.0 + model.options.poisson)))
    peak = np.array([peak_scan(displacement[:, i])[0] for i in range(n_envs)])
    np.testing.assert_allclose(qd_to_numpy(state.peak), peak, rtol=1e-4, atol=1e-3)
    recovered = qd_to_numpy(state.displacement, transpose=True).reshape((n_envs, -1)).T
    residual = np.linalg.norm(oracle.k @ recovered - rhs, axis=0)
    print("native iterations", qd_to_numpy(state.iterations), "full residual", residual)
    assert qd_to_numpy(state.valid).all()
    assert (residual <= np.maximum(1e-11, 1e-7 * np.linalg.norm(rhs, axis=0))).all()
    np.testing.assert_array_equal(recovered[:, 0], 0.0)
