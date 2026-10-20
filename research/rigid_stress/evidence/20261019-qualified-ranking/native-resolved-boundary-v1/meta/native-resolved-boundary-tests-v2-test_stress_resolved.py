from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.resolved import kernel_scatter_resolved
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from tests.rigid.test_stress_nonconvex import nonconvex_material_mesh  # noqa: F401
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
def test_resolved_boundary_work_release_and_failure_isolation(nonconvex_material_mesh, asset_tmp_path):  # noqa: F811
    vertices, tetrahedra, triangles, shape = nonconvex_material_mesh
    mesh = asset_tmp_path / f"resolved-{shape}.npz"
    np.savez(mesh, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=triangles)
    model = StressModel(RigidStressOptions.dpr(mesh=mesh, load_transfer="resolved", method="direct"))
    surface = StressSurface(1, model.info)
    nodes = qd_to_numpy(model.info.surface_nodes, transpose=True).T
    coordinates = qd_to_numpy(model.info.vertices, transpose=True)
    normals = qd_to_numpy(surface.info.face_normal, transpose=True)
    n_faces = len(nodes)
    # Quadratic virtual displacement tests exact work without reproducing the
    # scatter implementation's P2 weights. Include a genuine interior boundary.
    bary = np.column_stack((np.linspace(0.15, 0.3, n_faces), np.linspace(0.2, 0.4, n_faces)))
    bary = np.column_stack((bary, 1 - bary.sum(axis=1)))
    points = np.einsum("fc,fci->fi", bary, coordinates[nodes[:, :3]])
    x, y, z = (np.concatenate((coordinates, points)) / 0.003).T
    displacement = 1e-5 * np.stack((x * x + 2 * y * z, x * z - y * y, x * y + 3 * z * z), axis=-1)
    nodal_displacement, point_displacement = np.split(displacement, [len(coordinates)])
    tangent = coordinates[nodes[:, 1]] - coordinates[nodes[:, 0]]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    forces = 0.01 * (-normals + 0.2 * tangent)
    target = np.array([1.5, 1.5, 1.0] if shape == "cavity" else [1.0, 2.0, 0.5]) * 0.001
    selected = np.argmin(np.linalg.norm(coordinates[nodes[:, :3]].mean(axis=1) - target, axis=1))
    assert ((coordinates[nodes[selected, :3]] > 0).any(axis=0)).all()
    active = np.ones((n_faces, 3), dtype=gs.np_bool)
    active[:, 1] = False
    active[selected, 1] = True
    point_inputs = np.broadcast_to(points[:, None], (n_faces, 3, 3)).copy()
    force_inputs = forces[:, None] * np.array([1.0, 1.0, 2.0])[None, :, None]
    contacts = create_contacts(n_faces, 3, 0.006)
    contacts.position.from_numpy(point_inputs)
    contacts.force.from_numpy(force_inputs)
    contacts.normal.from_numpy(np.broadcast_to(-normals[:, None], (n_faces, 3, 3)).copy())
    contacts.friction.fill(0.3)
    contacts.valid.from_numpy(active)
    sources = create_sources(n_faces, 3)
    sources.geom.fill(1)
    sources.pair.fill(2)
    sources.witness.from_numpy(point_inputs)
    sources.witness_face.from_numpy(np.broadcast_to(np.arange(n_faces)[:, None], (n_faces, 3)).copy())
    sources.witness_barycentric.from_numpy(np.broadcast_to(bary[:, None], (n_faces, 3, 3)).copy())
    state = model.create_state(3, "full")
    omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
    omega.fill(0)
    errno = V(dtype=gs.qd_int, shape=(3,))
    epsilon = float(np.finfo(gs.np_float).eps)
    previous = None
    for released in (False, True):
        active[:, 0] = not released
        contacts.valid.from_numpy(active)
        kernel_begin_step(state)
        kernel_resolve_sources(1e-4, epsilon, contacts, sources, model.info, surface.info)
        kernel_scatter_resolved(epsilon, contacts, sources, state, model.info, surface.info)
        assert_equal(qd_to_numpy(contacts.status, transpose=True), 0)
        actual = qd_to_numpy(state.force, transpose=True)
        expected = np.moveaxis(force_inputs * active[:, :, None], 1, 0)
        assert_allclose(actual.sum(axis=1), expected.sum(axis=1), rtol=0, atol=1e-14)
        assert_allclose(
            np.cross(coordinates, actual).sum(axis=1),
            np.cross(points, expected).sum(axis=1),
            rtol=0,
            atol=1e-16,
        )
        assert_allclose(
            np.einsum("bni,ni->b", actual, nodal_displacement),
            np.einsum("bfi,fi->b", expected, point_displacement),
            rtol=1e-12,
            atol=1e-20,
        )
        exterior = np.ones(len(coordinates), dtype=bool)
        exterior[nodes[selected]] = False
        assert_equal(actual[1, exterior], 0)
        model.recover(omega, state, surface_load=True)
        kernel_accept(state, contacts, errno)
        assert_equal(qd_to_numpy(errno, transpose=True), 0)
        field = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)
        if released:
            assert previous is not None
            assert_equal(field[0], 0)
            assert_allclose(field[1:], previous[1:], rtol=1e-10, atol=1e-6)
        else:
            assert_allclose(field[2], 2 * field[0], rtol=1e-10, atol=1e-6)
            previous = field
    # Moving the application point tangentially adds an unrepresented moment.
    # The other environments must retain their exact admitted boundary loads.
    point_inputs[0, 2] += 1e-5 * tangent[0]
    contacts.position.from_numpy(point_inputs)
    kernel_begin_step(state)
    kernel_resolve_sources(1e-4, epsilon, contacts, sources, model.info, surface.info)
    kernel_scatter_resolved(epsilon, contacts, sources, state, model.info, surface.info)
    assert_equal(qd_to_numpy(contacts.status, transpose=True)[2, 0], 4)
    model.recover(omega, state, surface_load=True)
    kernel_accept(state, contacts, errno)
    field = qd_to_numpy(state.stress_tensor, transpose=True)
    assert np.isnan(field[2]).all()
    assert_equal(field[0], 0)
    assert_allclose(field[1], previous[1], rtol=1e-10, atol=1e-6)


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("output_mode", ("max", "full"))
def test_resolved_native_observation_reset_and_checkpoint(show_viewer, output_mode):
    assets = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            enable_contact_patch=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    egg = scene.add_entity(
        morph=gs.morphs.URDF(
            file=assets / "egg_shell.urdf",
            pos=(0.0, 0.0, 0.031),
            align=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=True,
        ),
    )
    link = egg.base_link
    link.configure_stress_recovery(
        gs.options.RigidStressOptions.dpr(
            mesh=assets / "elastic.npz",
            output_mode=output_mode,
            load_transfer="resolved",
        )
    )
    scene.build(n_envs=3)
    # These points are an explicitly declared discrete load baseline. This
    # lifecycle test makes no physical pressure-resolution or accuracy claim.
    for _ in range(8):
        scene.step()
    peak = tensor_to_array(link.get_max_stress())
    assert np.isfinite(peak).all() and (peak > 0).all()
    if output_mode == "full":
        tensor, mises = (tensor_to_array(value) for value in link.get_stress_field())
        assert_allclose(mises.max(axis=(1, 2)), peak)
    checkpoint = scene.get_state()
    scene.step()
    position, next_peak = tensor_to_array(egg.get_pos()), tensor_to_array(link.get_max_stress())
    scene.reset(state=checkpoint)
    if output_mode == "full":
        assert all(np.isnan(tensor_to_array(value)).all() for value in link.get_stress_field())
    scene.step()
    assert_allclose(tensor_to_array(egg.get_pos()), position, rtol=0, atol=1e-15)
    assert_allclose(tensor_to_array(link.get_max_stress()), next_peak, rtol=1e-8, atol=1e-6)
    before = tensor_to_array(link.get_max_stress())
    scene.reset(envs_idx=[1])
    assert_equal(tensor_to_array(link.get_max_stress(envs_idx=[0, 2])), before[[0, 2]])
    assert_equal(tensor_to_array(link.get_max_stress(envs_idx=[1])), 0)
