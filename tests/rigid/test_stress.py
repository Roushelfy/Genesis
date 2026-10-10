import numpy as np
import pytest
import quadrants as qd
import trimesh
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    create_contacts,
    func_anchor,
    func_pack_contacts,
    func_pressure_correct_warp,
    func_pressure_warp,
    func_refine_contacts,
    func_scatter_warp,
    kernel_anchor,
    kernel_pressure,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.history import StressHistory
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept
from genesis.engine.solvers.rigid.stress.solve import kernel_peak
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry, shell_mesh
from research.rigid_stress.peak_cpu import P2Peak
from research.rigid_stress.wrench import FinitePatchMapper, WrenchPatch


@qd.kernel(graph=True)
def kernel_apex_contact_graph(
    epsilon: float,
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
):
    func_anchor(epsilon, contacts, surface)
    func_pack_contacts(contacts)
    func_pressure_warp(contacts, surface, False)
    func_pressure_correct_warp(contacts, surface)
    func_refine_contacts(contacts)
    func_pressure_warp(contacts, surface, True)
    func_pressure_correct_warp(contacts, surface)
    func_scatter_warp(contacts, state, info, surface, True)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_full_field_same_state_and_invalid_solve(tmp_path):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh))
    maximum = model.create_state(3)
    complete = model.create_state(3, "full")
    assert maximum.stress_tensor.shape[0] == maximum.von_mises.shape[0] == 0
    random = np.random.default_rng(10102026)
    forces = random.normal(size=(model.info.vertices.shape[0], 3, 3)) * 1e-4
    omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
    omega.from_numpy(np.array([[0.2, -0.3, 0.4], [0.8, 0.1, -0.5], [-0.6, 0.9, 0.2]]))
    for state in (maximum, complete):
        state.force.from_numpy(forces)
        model.recover(omega, state)
        assert qd_to_numpy(state.valid).all()
    # Separate CUDA load reductions may differ by rounding. Compare output modes on one identical recovered state.
    complete.displacement.from_numpy(qd_to_numpy(maximum.displacement))
    kernel_peak(1e10, 0.3, complete, model.info)
    np.testing.assert_array_equal(qd_to_numpy(maximum.peak), qd_to_numpy(complete.peak))
    tensor = qd_to_numpy(complete.stress_tensor, transpose=True)
    vm = qd_to_numpy(complete.von_mises, transpose=True)[..., 0]
    np.testing.assert_array_equal(vm.max(axis=(1, 2)), qd_to_numpy(maximum.peak))
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    rhs = qd_to_numpy(complete.rhs, transpose=True).reshape((3, -1)).T
    reference = np.zeros_like(rhs)
    reference[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    for i_b in range(3):
        expected_tensor, expected_vm = stress_field(oracle.glambda, oracle.elements, reference[:, i_b], 1e10, 0.3)
        np.testing.assert_allclose(tensor[i_b], expected_tensor, rtol=2e-9, atol=2e-6)
        np.testing.assert_allclose(vm[i_b], expected_vm, rtol=2e-9, atol=2e-6)
    contacts = create_contacts(1, 3, 0.006)
    contacts.valid.fill(False)
    complete.valid.from_numpy(np.array([True, False, True]))
    from genesis.utils.array_class import V

    errno = V(dtype=gs.qd_int, shape=(3,))
    errno.fill(0)
    kernel_accept(complete, contacts, errno)
    np.testing.assert_array_equal(qd_to_numpy(complete.stress_tensor, transpose=True)[[0, 2]], tensor[[0, 2]])
    assert np.isnan(qd_to_numpy(complete.stress_tensor, transpose=True)[1]).all()
    assert np.isnan(qd_to_numpy(complete.von_mises, transpose=True)[1]).all()
    assert qd_to_numpy(errno)[1] != 0
    np.testing.assert_array_equal(qd_to_numpy(complete.invalid_steps), [0, 1, 0])
    kernel_accept(complete, contacts, errno)
    np.testing.assert_array_equal(qd_to_numpy(complete.invalid_steps), [0, 1, 0])


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_full_field_unbatched_shared_operator(tmp_path):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    collision = tmp_path / "shell.obj"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(collision)
    scene = gs.Scene(show_viewer=False)
    maximum = scene.add_entity(gs.morphs.Mesh(file=collision, pos=(-0.1, 0, 1), convexify=True, decimate=False))
    complete = scene.add_entity(gs.morphs.Mesh(file=collision, pos=(0.1, 0, 1), convexify=True, decimate=False))
    maximum.base_link.configure_stress_recovery(RigidStressOptions(mesh=mesh))
    complete.base_link.configure_stress_recovery(RigidStressOptions(mesh=mesh, output_mode="full"))
    scene.build()
    entries = scene.rigid_solver.stress_recovery.links
    assert entries[0].model is entries[1].model
    assert entries[0].state.stress_tensor.shape[0] == 0
    tensor, vm = complete.base_link.get_stress_field(copy=False)
    assert tensor.shape == (len(tetrahedra), 4, 6) and vm.shape == (len(tetrahedra), 4)
    assert np.isnan(tensor_to_array(tensor)).all()
    scene.step()
    scene.rigid_solver.check_errno()
    np.testing.assert_array_equal(tensor_to_array(vm), 0.0)
    np.testing.assert_array_equal(tensor_to_array(tensor), 0.0)
    assert complete.base_link.get_stress_field(copy=False)[0].data_ptr() == tensor.data_ptr()
    assert complete.base_link.get_stress_field(copy=True)[0].data_ptr() != tensor.data_ptr()
    scene.reset()
    assert np.isnan(tensor_to_array(tensor)).all()
    with pytest.raises(gs.GenesisException, match="before building"):
        complete.base_link.configure_stress_recovery(RigidStressOptions(mesh=mesh))
    complete.base_link.stress_options.output_mode = "max"
    with pytest.raises(gs.GenesisException, match="rebuilding"):
        scene.step()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("young,poisson,density", ((1e10, 0.3, 2000.0), (3.2e9, 0.22, 1100.0)))
def test_native_p2_shared_operators(tmp_path, young, poisson, density):
    vertices, tetrahedra, surface, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    options = RigidStressOptions(mesh=mesh, young=young, poisson=poisson, density=density)
    model = StressModel(options)
    oracle = P2Shell(vertices, tetrahedra, surface, young, poisson, density, 2, factor_backend="none")
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
@pytest.mark.parametrize(
    "method,cooperative,inverse_precision,inverse_corrections,surface_load",
    (
        ("direct", False, "64", 2, False),
        ("direct", True, "64", 2, False),
        ("inverse", True, "64", 2, False),
        ("inverse", True, "64", 2, True),
        ("inverse", True, "32", 2, False),
        ("inverse", True, "32", 0, False),
    ),
)
def test_native_batched_recovery_against_full_fp64_direct(
    tmp_path, method, cooperative, inverse_precision, inverse_corrections, surface_load
):
    vertices, tetrahedra, surface, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=surface)
    model = StressModel(
        RigidStressOptions(
            mesh=mesh,
            tolerance=1e-7,
            max_iterations=4000,
            cooperative_solve=cooperative,
            method=method,
            inverse_precision=inverse_precision,
            inverse_corrections=inverse_corrections,
        )
    )
    oracle = P2Shell(vertices, tetrahedra, surface, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    n_envs = 5
    state = model.create_state(n_envs)
    random = np.random.default_rng(8231)
    force = random.normal(size=(len(oracle.xyz), n_envs, 3)) * 0.01
    if surface_load:
        interior = np.setdiff1d(np.arange(len(force)), np.unique(qd_to_numpy(model.info.surface_nodes)))
        force[interior] = 0.0
    force[:, 0] = 0.0
    angular_velocity = random.normal(size=(n_envs, 3)) * 2.0
    angular_velocity[0] = 0.0
    state.force.from_numpy(force)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.from_numpy(angular_velocity)
    model.recover(omega, state, surface_load=surface_load)
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
    cached_peak = qd_to_numpy(state.peak, copy=True)
    from genesis.engine.solvers.rigid.stress.solve import kernel_peak

    kernel_peak(model.options.young, model.options.poisson, state, model.info, False)
    np.testing.assert_array_equal(qd_to_numpy(state.peak), cached_peak)
    recovered = qd_to_numpy(state.displacement, transpose=True).reshape((n_envs, -1)).T
    residual = np.linalg.norm(oracle.k @ recovered - rhs, axis=0)
    displacement_error = np.linalg.norm(recovered - displacement, axis=0) / np.maximum(
        np.linalg.norm(displacement, axis=0), 1e-30
    )
    print(
        "native method",
        method,
        cooperative,
        inverse_precision,
        "full residual",
        residual,
        "relative displacement error",
        displacement_error,
        "peak absolute error Pa",
        np.abs(qd_to_numpy(state.peak) - peak),
        "corrections",
        qd_to_numpy(state.corrections),
        "fallbacks",
        qd_to_numpy(state.fallbacks),
    )
    assert displacement_error.max() < 1e-4
    assert qd_to_numpy(state.valid).all()
    if inverse_precision == "32":
        if inverse_corrections == 2:
            np.testing.assert_array_equal(qd_to_numpy(state.fallbacks), 0)
        else:
            np.testing.assert_array_equal(qd_to_numpy(state.fallbacks), [0, 1, 1, 1, 1])
    assert (residual <= np.maximum(1e-11, 1e-7 * np.linalg.norm(rhs, axis=0))).all()
    np.testing.assert_array_equal(recovered[:, 0], 0.0)
    if surface_load:
        # Refresh dense, sparse and empty loads in the same state; no stale
        # column or environment may survive a changed contact/reset load.
        boundary = qd_to_numpy(model.surface_inverse.info.nodes)
        for refresh in range(3):
            updated = force.copy()
            if refresh == 1:
                updated.fill(0.0)
                for i_b in range(1, n_envs):
                    updated[boundary[3 * i_b], i_b] = force[boundary[3 * i_b], i_b]
            elif refresh == 2:
                updated.fill(0.0)
            state.force.from_numpy(updated)
            model.surface_inverse.apply(model.options.young, omega, state, packed=False)
            dense = qd_to_numpy(state.displacement, copy=True)
            model.surface_inverse.apply(model.options.young, omega, state, packed=True)
            np.testing.assert_array_equal(qd_to_numpy(state.displacement), dense)
            if gs.backend == gs.cuda:
                count = np.count_nonzero(np.any(updated[boundary] != 0.0, axis=2), axis=0)
                np.testing.assert_array_equal(qd_to_numpy(state.boundary_count), count)
            model.recover(omega, state, surface_load=True)
            assert qd_to_numpy(state.valid).all()


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("cooperative", (False, True))
def test_native_finite_pressure_matches_independent_cpu(tmp_path, cooperative):
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
    mapper = FinitePatchMapper(geometry, anchor_to_surface=True, adaptive_integration=True)
    n_contacts, n_envs = 3, 5
    contacts = create_contacts(n_contacts, n_envs, 0.006, cooperative)
    state = model.create_state(n_envs)
    random = np.random.default_rng(7143)
    position = np.zeros((n_contacts, n_envs, 3))
    force = np.zeros_like(position)
    normal = np.zeros_like(position)
    radius = random.uniform(0.0048, 0.0075, size=(n_contacts, n_envs))
    valid = np.zeros((n_contacts, n_envs), dtype=bool)
    friction = np.broadcast_to([0.0, 0.0, 0.3, 0.6, 1.4], (n_contacts, n_envs)).copy()
    expected = np.zeros((len(oracle.xyz), n_envs, 3))
    for i_b in range(1, n_envs):
        for i_c in range(min(i_b, n_contacts)):
            i_f = random.integers(len(faces))
            bary = random.uniform(0.2, 0.5, size=3)
            bary /= bary.sum()
            point = bary @ vertices[faces[i_f]]
            inward = -mapper.face_normals[i_f]
            tangent = np.cross(inward, np.eye(3)[np.argmin(np.abs(inward))])
            tangent /= np.linalg.norm(tangent)
            tangent_scale = 0.0 if friction[0, i_b] == 0.0 else 0.25
            traction = (0.1 + random.random()) * (inward + tangent_scale * tangent)
            patch = WrenchPatch(point, traction, radius[i_c, i_b], friction[i_c, i_b], np.zeros(3), inward)
            mapped = mapper.map(patch)
            position[i_c, i_b], force[i_c, i_b], normal[i_c, i_b] = point, traction, inward
            valid[i_c, i_b] = True
            expected[:, i_b] += mapped.nodal_force_n.reshape((-1, 3))
    contacts.position.from_numpy(position)
    contacts.force.from_numpy(force)
    contacts.normal.from_numpy(normal)
    contacts.radius.from_numpy(radius)
    contacts.friction.from_numpy(friction)
    contacts.valid.from_numpy(valid)
    kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
    kernel_pressure(contacts, surface.info, cooperative)
    kernel_scatter(contacts, state, model.info, surface.info)
    print("patch statuses", qd_to_numpy(contacts.status), "evaluations", qd_to_numpy(contacts.evaluations))
    np.testing.assert_array_equal(qd_to_numpy(contacts.status), 0)
    np.testing.assert_allclose(qd_to_numpy(state.force), expected, atol=1e-12, rtol=2e-10)
    assert qd_to_numpy(contacts.force_error).max() < 1e-9
    assert qd_to_numpy(contacts.moment_error).max() < 1e-11


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("case", ("random", "live_sliding"))
def test_native_constrained_pressure_and_invalid_inputs(tmp_path, case):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh, method="pcg"))
    surface = StressSurface(10, model.info)
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    mapper = FinitePatchMapper(SurfaceGeometry(oracle, 10), anchor_to_surface=True, adaptive_integration=True)
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
    if case == "live_sliding":
        patch = WrenchPatch(
            np.array([-0.010732644239792082, -0.018915982500044032, 8.744938177757137e-7]),
            np.array([-0.0024097694624780827, 0.010202571207786422, -0.007137609867135896]),
            0.00721357873242662,
            1.0,
            np.zeros(3),
            np.array([0.273174876142152, 0.9616158684519065, 0.025892249539301006]),
        )
        mapped = mapper.map(patch)
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
@pytest.mark.parametrize("execution", ("serial", "warp", "graph"))
@pytest.mark.parametrize("case", ("batch2048", "batch16384", "batch32768"))
def test_native_live_apex_wrench_recovery(tmp_path, execution, case):
    if execution == "graph" and gs.backend != gs.cuda:
        pytest.skip("The native fused contact graph runs on CUDA.")
    cooperative = execution != "serial"
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh))
    surface = StressSurface(10, model.info)
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    mapper = FinitePatchMapper(SurfaceGeometry(oracle, 10), anchor_to_surface=True, adaptive_integration=True)
    # Actual B=2048 rollout, environment 1996, tick 632. Do not widen its footprint to accept it.
    patch = WrenchPatch(
        np.array([4.0606709579473816e-5, 1.3926954540923124e-5, 0.0299207658748014]),
        np.array([0.004688181349230815, 0.0023709449777518836, -0.0015963625737377865]),
        0.005998121586384792,
        1.0,
        np.zeros(3),
        np.array([0.45061547141681174, 0.1545483800796497, -0.8792385882879348]),
    )
    if case == "batch16384":
        # Actual varied-mu B=16384, environment 14658, timed trajectory tick 1017.
        patch = WrenchPatch(
            np.array([8.423108194112004e-6, -9.491831147112614e-6, 0.02997678241766882]),
            np.array([0.0014871093110934814, -0.0017833685020508818, -0.0009493177723757649]),
            0.0059312792977239135,
            0.8132615716182551,
            np.zeros(3),
            np.array([0.3183675187512961, -0.3587619511864121, -0.8774576829597318]),
        )
    elif case == "batch32768":
        # Actual seed 623001, environment 13456, trajectory tick 943, radius unchanged.
        # Local Q10 is LP-feasible; ordinary FP64 dual fitting stalls numerically.
        patch = WrenchPatch(
            np.array([2.5870448174229687e-6, -2.0128764122128245e-6, 0.029993827413302653]),
            np.array([0.003926793922944923, -0.00281597196231237, -0.0020435164841347774]),
            0.004110711675411422,
            0.8134515011539788,
            np.zeros(3),
            np.array([0.37028065349108424, -0.28810061127035297, -0.8831139651459852]),
        )
    anchor = mapper.surface_anchor(patch.center_m, patch.force_n)
    ids = np.array(mapper.tree.query_ball_point(anchor, patch.radius_m, return_sorted=True))
    direction = patch.force_n / np.linalg.norm(patch.force_n)
    first = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    first /= np.linalg.norm(first)
    transverse = (mapper.positions[ids] - anchor) @ np.stack((first, np.cross(direction, first))).T
    operator = np.column_stack((np.ones(len(ids)), transverse / patch.radius_m))
    feasibility = linprog(np.zeros(len(ids)), A_eq=operator.T, b_eq=[1, 0, 0], bounds=(0, None))
    assert feasibility.status == 2
    mapped = mapper.map(patch)
    assert mapped.diagnostics.minimum_normal_force_n >= -1e-15
    assert mapped.diagnostics.maximum_cone_excess_n < 1e-14
    contacts = create_contacts(1, 1, patch.radius_m, cooperative)
    state = model.create_state(1)
    contacts.position.from_numpy(patch.center_m.reshape((1, 1, 3)))
    contacts.force.from_numpy(patch.force_n.reshape((1, 1, 3)))
    contacts.normal.from_numpy(patch.inward_normal.reshape((1, 1, 3)))
    contacts.friction.fill(patch.friction)
    contacts.valid.fill(True)
    if execution == "graph":
        kernel_apex_contact_graph(float(np.finfo(float).eps), contacts, state, model.info, surface.info)
    else:
        kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info, cooperative)
        kernel_scatter(contacts, state, model.info, surface.info)
    print("apex", cooperative, "status", qd_to_numpy(contacts.status), "iterations", qd_to_numpy(contacts.evaluations))
    np.testing.assert_array_equal(qd_to_numpy(contacts.status), 0)
    assert qd_to_numpy(contacts.evaluations).max() <= 110
    np.testing.assert_allclose(qd_to_numpy(state.force)[:, 0].reshape(-1), mapped.nodal_force_n, rtol=2e-7, atol=1e-10)
    assert qd_to_numpy(contacts.force_error).max() <= 1e-8 * np.linalg.norm(patch.force_n)
    assert qd_to_numpy(contacts.moment_error).max() <= 1e-8 * np.linalg.norm(patch.force_n) * patch.radius_m
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    omega.fill(0)
    model.recover(omega, state)
    rhs = mapped.nodal_force_n - oracle.mr @ np.linalg.solve(oracle.gram, oracle.r.T @ mapped.nodal_force_n)
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    displacement = np.zeros_like(rhs)
    displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(rhs[free])
    recovered = qd_to_numpy(state.displacement)[:, 0].reshape(-1)
    assert np.linalg.norm(oracle.k @ recovered - rhs) <= max(1e-11, 1e-8 * np.linalg.norm(rhs))
    scan = P2Peak(oracle.glambda, oracle.elements, 1e10 / 2.6)
    np.testing.assert_allclose(qd_to_numpy(state.peak), scan(displacement)[0], rtol=1e-4, atol=1e-3)
    assert qd_to_numpy(state.valid).all()


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
@pytest.mark.parametrize("output_mode", ("max", "full"))
def test_native_rigid_lifecycle_and_partial_reset(tmp_path, substeps, output_mode, monkeypatch):
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
    link.configure_stress_recovery(
        RigidStressOptions(
            mesh=mesh, tolerance=1e-7, history_size=4 if output_mode == "max" else 0, output_mode=output_mode
        )
    )
    scene.build(n_envs=3)
    assert not scene._pre_substep_callbacks and not scene._post_substep_callbacks
    egg.set_pos(np.array([[0.0, 0.0, 0.031], [0.0, 0.0, 0.045], [0.0, 0.0, 0.061]]))
    angles = np.array([0.23, -0.31, 0.47])
    quaternions = np.column_stack((np.cos(angles / 2), np.zeros((3, 2)), np.sin(angles / 2)))
    egg.set_quat(quaternions)
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
    # Independent frame/contact audit of the actual final solved contact snapshot.
    frame_quat = qd_to_numpy(entry.contacts.frame_quaternion)
    frame_pos = qd_to_numpy(entry.contacts.frame_position)
    rotations = Rotation.from_quat(frame_quat[:, [1, 2, 3, 0]])
    collider = scene.rigid_solver.collider.collider_state
    sorted_indices = qd_to_numpy(collider.contact_sort_idx, transpose=True)
    source_positions = qd_to_numpy(collider.contact_data.pos, transpose=True)
    source_forces = qd_to_numpy(collider.contact_data.force, transpose=True)
    source_normals = qd_to_numpy(collider.contact_data.normal, transpose=True)
    source_a = qd_to_numpy(collider.contact_data.link_a, transpose=True)
    local_positions = qd_to_numpy(entry.contacts.position, transpose=True)
    local_forces = qd_to_numpy(entry.contacts.force, transpose=True)
    local_normals = qd_to_numpy(entry.contacts.normal, transpose=True)
    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
    for i_b, i_c in np.argwhere(valid):
        source = sorted_indices[i_b, i_c]
        sign = -1.0 if source_a[i_b, source] == link.idx else 1.0
        inverse = rotations[i_b].inv()
        np.testing.assert_allclose(
            local_positions[i_b, i_c], inverse.apply(source_positions[i_b, source] - frame_pos[i_b]), atol=1e-15
        )
        np.testing.assert_allclose(local_forces[i_b, i_c], inverse.apply(sign * source_forces[i_b, source]), atol=1e-14)
        np.testing.assert_allclose(
            local_normals[i_b, i_c], inverse.apply(-sign * source_normals[i_b, source]), atol=1e-14
        )
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    mapper = FinitePatchMapper(SurfaceGeometry(oracle, 10), anchor_to_surface=True, adaptive_integration=True)
    radius = qd_to_numpy(entry.contacts.radius, transpose=True)
    friction = qd_to_numpy(entry.contacts.friction, transpose=True)
    expected_force = np.zeros((oracle.ndof, 3))
    for i_b, i_c in np.argwhere(valid):
        if np.linalg.norm(local_forces[i_b, i_c]) > 1e-30:
            patch = WrenchPatch(
                local_positions[i_b, i_c],
                local_forces[i_b, i_c],
                radius[i_b, i_c],
                friction[i_b, i_c],
                np.zeros(3),
                local_normals[i_b, i_c],
            )
            expected_force[:, i_b] += mapper.map(patch).nodal_force_n
    np.testing.assert_allclose(
        qd_to_numpy(entry.state.force, transpose=True).reshape((3, -1)).T, expected_force, rtol=2e-7, atol=1e-10
    )
    angular_velocity = qd_to_numpy(entry.omega)
    centrifugal = -np.cross(
        angular_velocity[:, None], np.cross(angular_velocity[:, None], (oracle.xyz - oracle.com)[None])
    )
    raw = expected_force + oracle.m @ centrifugal.reshape((3, -1)).T
    expected_rhs = raw - oracle.mr @ np.linalg.solve(oracle.gram, oracle.r.T @ raw)
    actual_rhs = qd_to_numpy(entry.state.rhs, transpose=True).reshape((3, -1)).T
    np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=2e-7, atol=1e-10)
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(entry.model.info.pins))
    displacement = np.zeros_like(expected_rhs)
    displacement[free] = splu(oracle.k[free][:, free].tocsc()).solve(expected_rhs[free])
    scan = P2Peak(oracle.glambda, oracle.elements, 1e10 / 2.6)
    expected_peak = np.array([scan(displacement[:, i_b])[0] for i_b in range(3)])
    np.testing.assert_allclose(qd_to_numpy(entry.state.peak), expected_peak, rtol=1e-4, atol=1e-3)
    print("actual rotated contacts substeps", substeps, "peak error Pa", qd_to_numpy(entry.state.peak) - expected_peak)
    before_field = None
    if output_mode == "full":
        field, vm = link.get_stress_field(copy=False)
        assert field.shape == (3, len(tetrahedra), 4, 6)
        assert vm.shape == (3, len(tetrahedra), 4)
        before_field = (tensor_to_array(field).copy(), tensor_to_array(vm).copy())
        for i_b in range(3):
            expected_tensor, expected_vm = stress_field(
                oracle.glambda, oracle.elements, displacement[:, i_b], 1e10, 0.3
            )
            np.testing.assert_allclose(before_field[0][i_b], expected_tensor, rtol=1e-4, atol=1e-3)
            np.testing.assert_allclose(before_field[1][i_b], expected_vm, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(before_field[1].max(axis=(1, 2)), qd_to_numpy(entry.state.peak), rtol=2e-15)
        if substeps == 1:
            np.testing.assert_array_equal(before_field[1].max(axis=(1, 2)), tensor_to_array(link.get_max_stress()))
        else:
            assert (tensor_to_array(link.get_max_stress()) >= before_field[1].max(axis=(1, 2))).all()
    else:
        assert entry.state.stress_tensor.shape[0] == entry.state.von_mises.shape[0] == 0
        with pytest.raises(gs.GenesisException, match="output_mode='full'"):
            link.get_stress_field()
    link.set_stress_contact_radius(np.array([[0.004], [0.008]]), envs_idx=[0, 2])
    radius = qd_to_numpy(entry.contacts.radius, transpose=True)
    np.testing.assert_array_equal(radius[0], 0.004)
    np.testing.assert_array_equal(radius[1], 0.006)
    np.testing.assert_array_equal(radius[2], 0.008)
    with monkeypatch.context() as patch:
        patch.setattr(gs, "use_zerocopy", False)
        link.set_stress_contact_radius(0.007, envs_idx=[1])
        np.testing.assert_array_equal(qd_to_numpy(entry.contacts.radius, transpose=True)[1], 0.007)
    before_u = qd_to_numpy(entry.state.displacement, transpose=True, copy=True)
    before_peak = tensor_to_array(link.get_max_stress())
    before_count = qd_to_numpy(entry.history.state.count, copy=True) if entry.history is not None else None
    checkpoint = scene.rigid_solver.__getstate__()
    assert checkpoint.configs["stress.0.options"].mesh == str(mesh.resolve())
    json.dumps(vars(checkpoint.configs["stress.0.options"]))
    scene.reset(envs_idx=np.array([1]))
    after_u = qd_to_numpy(entry.state.displacement, transpose=True)
    after_peak = tensor_to_array(link.get_max_stress())
    np.testing.assert_array_equal(after_u[[0, 2]], before_u[[0, 2]])
    np.testing.assert_array_equal(after_peak[[0, 2]], before_peak[[0, 2]])
    np.testing.assert_array_equal(after_u[1], 0.0)
    assert after_peak[1] == 0.0
    if entry.history is not None:
        after_count = qd_to_numpy(entry.history.state.count)
        np.testing.assert_array_equal(after_count[[0, 2]], before_count[[0, 2]])
        assert after_count[1] == 0
    if before_field is not None:
        for actual, previous in zip(link.get_stress_field(copy=False), before_field):
            actual = tensor_to_array(actual)
            np.testing.assert_array_equal(actual[[0, 2]], previous[[0, 2]])
            assert np.isnan(actual[1]).all()
    scene.rigid_solver.__setstate__(checkpoint)
    # Restoring state broadcasts the ordinary geometry/dynamics notices, invalidating derived observations.
    np.testing.assert_array_equal(tensor_to_array(checkpoint.arrays["stress.0.state.step_peak"]), before_peak)
    np.testing.assert_array_equal(tensor_to_array(link.get_max_stress()), 0.0)
    if entry.history is not None:
        np.testing.assert_array_equal(qd_to_numpy(entry.history.state.count), 0)
    if before_field is not None:
        assert all(np.isnan(tensor_to_array(value)).all() for value in link.get_stress_field(copy=False))
    with pytest.raises(gs.GenesisException, match="fixed link mass"):
        link.set_mass(0.01)
    with pytest.raises(gs.GenesisException, match="finite surface loads"):
        link.apply_external_wrench(force=(0, 0, 1))
    baseline = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=substeps),
        rigid_options=scene.rigid_solver._options.model_copy(deep=True),
        show_viewer=False,
    )
    baseline.add_entity(gs.morphs.Plane())
    bare_egg = baseline.add_entity(gs.morphs.Mesh(file=collision, pos=(0, 0, 0.031), convexify=True, decimate=False))
    baseline.build(n_envs=3)
    assert baseline.rigid_solver.stress_recovery is None
    bare_egg.set_pos(np.array([[0.0, 0.0, 0.031], [0.0, 0.0, 0.045], [0.0, 0.0, 0.061]]))
    bare_egg.set_quat(quaternions)
    for _ in range(30):
        baseline.step()
    # The partial reset above changed only environment 1; compare the other actual trajectories.
    np.testing.assert_allclose(
        tensor_to_array(egg.get_pos())[[0, 2]], tensor_to_array(bare_egg.get_pos())[[0, 2]], rtol=0, atol=1e-10
    )


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_history_reuse_abrupt_load_and_reset(tmp_path):
    vertices, tetrahedra, faces, _ = shell_mesh(1, 2, 0.0005)
    mesh = tmp_path / "shell.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    model = StressModel(RigidStressOptions(mesh=mesh))
    state = model.create_state(3)
    history = StressHistory(model.info.vertices.shape[0], 3)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
    omega.fill(0)
    random = np.random.default_rng(9137)
    loads = random.normal(size=(5, model.info.vertices.shape[0], 3, 3)) * 0.01
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    factor = splu(oracle.k[free][:, free].tocsc())
    scan = P2Peak(oracle.glambda, oracle.elements, 1e10 / 2.6)
    sequence = [*loads[:4], 0.2 * loads[0] + 0.7 * loads[1] - 0.3 * loads[2] + 0.8 * loads[3], loads[4]]
    for tick, load in enumerate(sequence):
        state.force.from_numpy(load)
        model.recover(omega, state, history=history)
        history.append(state)
        rhs = qd_to_numpy(state.rhs, transpose=True).reshape((3, -1)).T
        displacement = np.zeros_like(rhs)
        displacement[free] = factor.solve(rhs[free])
        peak = np.array([scan(displacement[:, i_b])[0] for i_b in range(3)])
        np.testing.assert_allclose(qd_to_numpy(state.peak), peak, rtol=1e-4, atol=1e-3)
        assert qd_to_numpy(state.valid).all()
        np.testing.assert_array_equal(qd_to_numpy(history.state.hit), tick == 4)
    before = qd_to_numpy(history.state.count, copy=True)
    history.reset(np.array([1], dtype=gs.np_int))
    np.testing.assert_array_equal(qd_to_numpy(history.state.count)[[0, 2]], before[[0, 2]])
    state.force.from_numpy(loads[4])
    model.recover(omega, state, history=history)
    np.testing.assert_array_equal(qd_to_numpy(history.state.hit), [True, False, True])


import json
