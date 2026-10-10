import numpy as np
import pytest
import trimesh
from scipy import sparse
from scipy.sparse.linalg import splu

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry, shell_mesh
from research.rigid_stress.peak_cpu import P2Peak
from research.rigid_stress.wrench import FinitePatchMapper, WrenchPatch


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


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_finite_pressure_matches_independent_cpu(tmp_path):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    surface = StressSurface(10, model.info)
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    geometry = SurfaceGeometry(oracle, 10)
    np.testing.assert_allclose(qd_to_numpy(surface.info.positions), geometry.coords.reshape((-1, 3)), atol=2e-17)
    np.testing.assert_allclose(
        qd_to_numpy(surface.info.weights), geometry.integration_weights.reshape((-1,)), rtol=2e-13
    )
    mapper = FinitePatchMapper(geometry, anchor_to_surface=True)
    n_contacts, n_envs = 3, 4
    contacts = create_contacts(n_contacts, n_envs, 0.006)
    state = model.create_state(n_envs)
    random = np.random.default_rng(7143)
    position = np.zeros((n_contacts, n_envs, 3))
    force = np.zeros_like(position)
    normal = np.zeros_like(position)
    radius = random.uniform(0.0048, 0.0075, size=(n_contacts, n_envs))
    valid = np.zeros((n_contacts, n_envs), dtype=bool)
    expected = np.zeros((len(oracle.xyz), n_envs, 3))
    for i_b in range(1, n_envs):
        for i_c in range(i_b):
            i_f = random.integers(len(faces))
            bary = random.uniform(0.2, 0.5, size=3)
            bary /= bary.sum()
            point = bary @ vertices[faces[i_f]]
            inward = -mapper.face_normals[i_f]
            tangent = np.cross(inward, np.eye(3)[np.argmin(np.abs(inward))])
            tangent /= np.linalg.norm(tangent)
            traction = (0.1 + random.random()) * (inward + 0.25 * tangent)
            patch = WrenchPatch(point, traction, radius[i_c, i_b], 0.8, np.zeros(3), inward)
            mapped = mapper.map(patch)
            position[i_c, i_b], force[i_c, i_b], normal[i_c, i_b] = point, traction, inward
            valid[i_c, i_b] = True
            expected[:, i_b] += mapped.nodal_force_n.reshape((-1, 3))
    contacts.position.from_numpy(position)
    contacts.force.from_numpy(force)
    contacts.normal.from_numpy(normal)
    contacts.radius.from_numpy(radius)
    contacts.friction.fill(0.8)
    contacts.valid.from_numpy(valid)
    kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
    kernel_pressure(contacts, surface.info)
    kernel_scatter(contacts, state, model.info, surface.info)
    print("patch statuses", qd_to_numpy(contacts.status), "evaluations", qd_to_numpy(contacts.evaluations))
    np.testing.assert_array_equal(qd_to_numpy(contacts.status), 0)
    np.testing.assert_allclose(qd_to_numpy(state.force), expected, atol=1e-12, rtol=2e-10)
    assert qd_to_numpy(contacts.force_error).max() < 1e-9
    assert qd_to_numpy(contacts.moment_error).max() < 1e-11


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_constrained_pressure_and_invalid_inputs(tmp_path):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    surface = StressSurface(10, model.info)
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    mapper = FinitePatchMapper(SurfaceGeometry(oracle, 10), anchor_to_surface=True)
    random = np.random.default_rng(8125)
    selected = None
    for _ in range(200):
        i_f = random.integers(len(faces))
        bary = random.dirichlet([0.4, 0.4, 0.4])
        point = bary @ vertices[faces[i_f]]
        inward = -mapper.face_normals[i_f]
        tangent = np.cross(inward, np.eye(3)[np.argmin(abs(inward))])
        tangent /= np.linalg.norm(tangent)
        traction = inward + random.uniform(-0.9, 0.9) * tangent
        patch = WrenchPatch(point, traction, random.uniform(0.003, 0.01), 1.0, np.zeros(3), inward)
        try:
            mapped = mapper.map(patch)
        except ValueError:
            continue
        if mapped.diagnostics.constrained_evaluations > 1:
            selected = patch, mapped
            break
    assert selected is not None, "Fixture must exercise the constrained pressure correction."
    patch, mapped = selected
    contacts = create_contacts(6, 1, patch.radius_m)
    state = model.create_state(1)
    position = np.tile(patch.center_m, (6, 1, 1))
    normal = np.tile(patch.inward_normal, (6, 1, 1))
    force = np.tile(patch.force_n, (6, 1, 1))
    radius = np.full((6, 1), patch.radius_m)
    force[1] = 0.0
    radius[2] = 0.0
    radius[3] = np.nan
    force[4] = -patch.inward_normal
    force[5] = np.inf
    contacts.position.from_numpy(position)
    contacts.normal.from_numpy(normal)
    contacts.force.from_numpy(force)
    contacts.radius.from_numpy(radius)
    contacts.friction.fill(1.0)
    contacts.valid.fill(True)
    kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
    kernel_pressure(contacts, surface.info)
    kernel_scatter(contacts, state, model.info, surface.info)
    np.testing.assert_array_equal(qd_to_numpy(contacts.status).ravel(), [0, 0, 1, 1, 1, 1])
    assert qd_to_numpy(contacts.evaluations)[0, 0] > 1
    np.testing.assert_allclose(
        qd_to_numpy(state.force)[:, 0], mapped.nodal_force_n.reshape((-1, 3)), rtol=2e-7, atol=1e-10
    )


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("preconditioner", ("diagonal", "block"))
def test_native_pcg_iteration_limit_and_freefall(tmp_path, preconditioner):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    options = RigidStressOptions(
        mesh=mesh, method="pcg", preconditioner=preconditioner, max_iterations=7, tolerance=1e-12
    )
    model = StressModel(options)
    state = model.create_state(2)
    random = np.random.default_rng(9786)
    force = np.zeros((model.info.vertices.shape[0], 2, 3))
    force[:, 1] = random.normal(size=(len(force), 3))
    state.force.from_numpy(force)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(2,))
    omega.fill(0)
    model.recover(omega, state)
    np.testing.assert_array_equal(qd_to_numpy(state.iterations), [0, 7])
    np.testing.assert_array_equal(qd_to_numpy(state.valid), [True, False])
    assert qd_to_numpy(state.peak)[0] == 0.0
    # Adding a uniform gravitational body force is annihilated by inferred acceleration relief.
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    gravity = oracle.m @ np.tile([0.0, 0.0, -9.81], len(oracle.xyz))
    balanced = gravity - oracle.mr @ np.linalg.solve(oracle.gram, oracle.r.T @ gravity)
    np.testing.assert_allclose(balanced, 0.0, atol=1e-16)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("substeps", (1, 4))
def test_native_rigid_lifecycle_and_partial_reset(tmp_path, substeps):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    collision = tmp_path / "shell.obj"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(collision)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=substeps),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            iterations=100,
            tolerance=1e-12,
            contact_pruning_tolerance=None,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    egg = scene.add_entity(gs.morphs.Mesh(file=collision, pos=(0, 0, 0.031), convexify=True, decimate=False))
    link = egg.base_link
    link.configure_stress_recovery(RigidStressOptions(mesh=mesh, tolerance=1e-7))
    scene.build(n_envs=3)
    assert not scene._pre_substep_callbacks and not scene._post_substep_callbacks
    egg.set_pos(np.array([[0.0, 0.0, 0.031], [0.0, 0.0, 0.045], [0.0, 0.0, 0.061]]))
    entry = scene.rigid_solver.stress_recovery.links[0]
    reached_contact = np.zeros(3, dtype=bool)
    for _ in range(30):
        scene.step()
        scene.rigid_solver.check_errno()
        peak = tensor_to_array(link.get_max_stress(copy=False))
        assert np.isfinite(peak).all()
        reached_contact |= peak > 1.0
        assert (peak >= qd_to_numpy(entry.state.peak)).all()
    assert reached_contact.all()
    before_u = qd_to_numpy(entry.state.displacement, transpose=True, copy=True)
    before_peak = tensor_to_array(link.get_max_stress())
    scene.reset(envs_idx=np.array([1]))
    after_u = qd_to_numpy(entry.state.displacement, transpose=True)
    after_peak = tensor_to_array(link.get_max_stress())
    np.testing.assert_array_equal(after_u[[0, 2]], before_u[[0, 2]])
    np.testing.assert_array_equal(after_peak[[0, 2]], before_peak[[0, 2]])
    np.testing.assert_array_equal(after_u[1], 0.0)
    assert after_peak[1] == 0.0
