import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs
import genesis.utils.geom as gu

_TABLE_TOP = 0.4
_BOX_HALF = 0.025
_DROP_CLEARANCE = 0.02


@pytest.fixture
def multi_free_body_mjcf():
    mujoco = ET.Element("mujoco", model="qipc_rigid_body_multi_free")
    worldbody = ET.SubElement(mujoco, "worldbody")
    for name, joint_name, pos in (("first", "root", "-0.2 0 0.5"), ("second", "extra", "0.2 0 0.5")):
        body = ET.SubElement(worldbody, "body", name=name, pos=pos)
        ET.SubElement(body, "freejoint", name=joint_name)
        ET.SubElement(body, "geom", type="box", size="0.05 0.05 0.05")
    return ET.tostring(mujoco, encoding="unicode")


@pytest.fixture
def mixed_root_body_mjcf():
    mujoco = ET.Element("mujoco", model="qipc_rigid_body_mixed_roots")
    worldbody = ET.SubElement(mujoco, "worldbody")
    free_body = ET.SubElement(worldbody, "body", name="free_root", pos="-0.2 0 0.5")
    ET.SubElement(free_body, "freejoint", name="root")
    ET.SubElement(free_body, "geom", type="box", size="0.05 0.05 0.05")
    fixed_body = ET.SubElement(worldbody, "body", name="fixed_root", pos="0.2 0 0.5")
    ET.SubElement(fixed_body, "geom", type="box", size="0.05 0.05 0.05")
    return ET.tostring(mujoco, encoding="unicode")


@pytest.fixture
def fixed_then_free_body_mjcf():
    mujoco = ET.Element("mujoco", model="qipc_rigid_body_late_free")
    worldbody = ET.SubElement(mujoco, "worldbody")
    fixed_body = ET.SubElement(worldbody, "body", name="fixed_root", pos="-0.2 0 0.5")
    ET.SubElement(fixed_body, "geom", type="box", size="0.05 0.05 0.05")
    free_body = ET.SubElement(worldbody, "body", name="free_root", pos="0.2 0 0.5")
    ET.SubElement(free_body, "freejoint", name="late_free")
    ET.SubElement(free_body, "geom", type="box", size="0.05 0.05 0.05")
    return ET.tostring(mujoco, encoding="unicode")


@pytest.fixture
def offset_com_mjcf():
    mujoco = ET.Element("mujoco", model="qipc_rigid_body_offset_com")
    worldbody = ET.SubElement(mujoco, "worldbody")
    body = ET.SubElement(worldbody, "body", name="offset_body", pos="0 0 0.7")
    ET.SubElement(body, "freejoint", name="root")
    ET.SubElement(body, "inertial", pos="0.04 -0.02 0.01", mass="1.0", diaginertia="0.02 0.03 0.04")
    ET.SubElement(body, "geom", type="box", pos="0.04 -0.02 0.01", size="0.03 0.04 0.05")
    child = ET.SubElement(body, "body", name="fixed_child", pos="0.12 0.03 0.02")
    ET.SubElement(child, "inertial", pos="0.01 0 0", mass="0.4", diaginertia="0.006 0.007 0.008")
    ET.SubElement(child, "geom", type="box", pos="0.01 0 0", size="0.02 0.025 0.03")
    return ET.tostring(mujoco, encoding="unicode")


def _scene():
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=1.0e-3,
            init_collision_pair_capacity=100_000,
            contact_friction=0.5,
            contact_resistance=1.0e7,
        ),
        show_viewer=False,
    )


def _add_table(scene):
    return scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.5 * _TABLE_TOP), size=(0.6, 0.6, _TABLE_TOP), fixed=True),
        material=gs.materials.Rigid(coup_friction=0.5, contact_resistance=1.0e7),
    )


def _add_box(scene, *, y: float, rigid_body: bool):
    return scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, y, _TABLE_TOP + _BOX_HALF + _DROP_CLEARANCE),
            size=(2.0 * _BOX_HALF,) * 3,
            fixed=False,
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coup_friction=0.5,
            contact_resistance=1.0e7,
            qipc_rigid_body=rigid_body,
        ),
    )


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_initial_transforms_preserve_authored_mjcf_root_poses(mixed_root_body_mjcf):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.MJCF(file=mixed_root_body_mjcf, align=False),
        material=gs.materials.Rigid(),
    )
    assert entity.morph.pos is None

    scene.build()

    actual = entity.get_links_pos(relative=False).reshape(-1, 3).detach().cpu()
    expected = torch.tensor(
        [[-0.2, 0.0, 0.5], [0.2, 0.0, 0.5]],
        dtype=actual.dtype,
        device="cpu",
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_rigid_body_box_settles_and_resets_exactly():
    scene = _scene()
    _add_table(scene)
    abd_box = _add_box(scene, y=-0.1, rigid_body=False)
    rigid_box = _add_box(scene, y=0.1, rigid_body=True)
    scene.build()

    coupler = scene.sim.coupler
    rigid_collection = coupler._scene.rigid_body
    assert rigid_collection.n_bodies == 1
    assert coupler._rigid_link_indices_t.numel() == 1
    assert coupler._rigid_reset_state is not None

    # Frame-zero writeback reproduces the morph pose through the principal
    # inertia frame (A = R V^T) round trip.
    drop_z = _TABLE_TOP + _BOX_HALF + _DROP_CLEARANCE
    initial_pos = rigid_box.get_pos().reshape(-1).detach().cpu()
    torch.testing.assert_close(
        initial_pos,
        torch.tensor([0.0, 0.1, drop_z], dtype=initial_pos.dtype, device="cpu"),
        rtol=0.0,
        atol=1.0e-12,
    )
    initial_quat = rigid_box.get_quat().reshape(-1).detach().cpu()
    t0 = rigid_collection.t.detach().clone()
    quat0 = rigid_collection.quat.detach().clone()

    for _ in range(60):
        scene.step()

    assert torch.isfinite(rigid_collection.t).all()
    assert torch.isfinite(rigid_collection.quat).all()
    settled = rigid_box.get_pos().reshape(-1).detach().cpu()
    settled_abd = abd_box.get_pos().reshape(-1).detach().cpu()
    # Both boxes fall the ~2 cm clearance and rest on the table within the
    # IPC gap; the two body models must agree.
    assert float(settled[2]) < drop_z - 0.5 * _DROP_CLEARANCE
    assert abs(float(settled[2]) - (_TABLE_TOP + _BOX_HALF)) < 2.0e-3
    assert abs(float(settled[2]) - float(settled_abd[2])) < 2.0e-3

    scene.reset()
    assert torch.equal(rigid_collection.t, t0)
    assert torch.equal(rigid_collection.quat, quat0)
    assert float(rigid_collection.v.abs().max()) == 0.0
    assert float(rigid_collection.omega.abs().max()) == 0.0
    reset_pos = rigid_box.get_pos().reshape(-1).detach().cpu()
    reset_quat = rigid_box.get_quat().reshape(-1).detach().cpu()
    torch.testing.assert_close(reset_pos, initial_pos, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reset_quat, initial_quat, rtol=0.0, atol=0.0)

    for _ in range(60):
        scene.step()
    resettled = rigid_box.get_pos().reshape(-1).detach().cpu()
    assert abs(float(resettled[2]) - float(settled[2])) < 1.0e-3

    scene.reset()
    assert torch.equal(rigid_collection.t, t0)
    assert torch.equal(rigid_collection.quat, quat0)


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_rigid_body_writeback_updates_velocities_and_free_dofs(offset_com_mjcf):
    scene = _scene()
    rigid_body = scene.add_entity(
        morph=gs.morphs.MJCF(file=offset_com_mjcf, align=False),
        material=gs.materials.Rigid(qipc_rigid_body=True),
    )
    scene.build()

    coupler = scene.sim.coupler
    rigid_collection = coupler._scene.rigid_body
    assert coupler._rigid_link_indices_t.numel() == 2
    com = torch.tensor([[0.2, -0.1, 0.8]], dtype=torch.float64, device="cuda")
    principal_quat = torch.tensor(
        [[0.9238795325, 0.0, 0.3826834324, 0.0]],
        dtype=torch.float64,
        device="cuda",
    )
    com_velocity = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64, device="cuda")
    angular_velocity = torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float64, device="cuda")
    rigid_collection.t[:] = com
    rigid_collection.quat[:] = principal_quat
    rigid_collection.v[:] = com_velocity
    rigid_collection.omega[:] = angular_velocity

    coupler._writeback_state()
    torch.cuda.synchronize()

    link_indices = list(range(rigid_body.link_start, rigid_body.link_start + rigid_body.n_links))
    link_positions = scene.sim.rigid_solver.get_links_pos(link_indices, ref="link_origin").reshape(-1, 3)
    link_com_positions = scene.sim.rigid_solver.get_links_pos(link_indices, ref="link_com").reshape(-1, 3)
    root_coms = scene.sim.rigid_solver.get_links_root_COM(link_indices).reshape(-1, 3)
    link_pos = link_positions[0]
    link_quat = rigid_body.get_quat(relative=False).reshape(4)
    expected_com = com.to(link_positions)
    expected_angular_velocity = angular_velocity.to(link_positions)
    expected_link_velocities = com_velocity.to(link_positions) + torch.linalg.cross(
        expected_angular_velocity.expand_as(link_positions),
        link_positions - expected_com,
    )
    expected_link_com_velocities = com_velocity.to(link_com_positions) + torch.linalg.cross(
        angular_velocity.to(link_com_positions).expand_as(link_com_positions),
        link_com_positions - com.to(link_com_positions),
    )
    expected_local_angular_velocity = gu.quat_to_R(link_quat).transpose(-1, -2) @ expected_angular_velocity[0]

    assert float(torch.linalg.vector_norm(link_pos - expected_com[0])) > 1.0e-2
    torch.testing.assert_close(root_coms, expected_com.expand_as(root_coms), rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(rigid_body.joints[0].get_anchor_pos(), link_pos, rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(rigid_body.get_links_vel(), expected_link_velocities, rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(
        scene.sim.rigid_solver.get_links_vel(link_indices, ref="link_com"),
        expected_link_com_velocities,
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        rigid_body.get_links_ang(),
        expected_angular_velocity.expand(rigid_body.n_links, -1),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        rigid_body.get_dofs_velocity().reshape(6),
        torch.cat((expected_link_velocities[0], expected_local_angular_velocity)),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        rigid_body.get_dofs_position().reshape(6),
        torch.cat((link_pos, gu.quat_to_xyz(link_quat))),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_rigid_body_refuses_nonfixed_internal_joints(multi_free_body_mjcf):
    for mjcf in ("xml/one_ball_joint.xml", multi_free_body_mjcf):
        scene = _scene()
        scene.add_entity(
            morph=gs.morphs.MJCF(file=mjcf),
            material=gs.materials.Rigid(qipc_rigid_body=True),
        )

        with pytest.raises(gs.GenesisException, match="only fixed joints and an optional root free joint"):
            scene.build()


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_refuses_unsupported_free_root_layouts(multi_free_body_mjcf, fixed_then_free_body_mjcf):
    for mjcf in (multi_free_body_mjcf, fixed_then_free_body_mjcf):
        scene = _scene()
        scene.add_entity(
            morph=gs.morphs.MJCF(file=mjcf),
            material=gs.materials.Rigid(),
        )

        with pytest.raises(gs.GenesisException, match="at most one free joint on its first root link"):
            scene.build()


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_rigid_body_failed_reset_restores_runtime_state(monkeypatch):
    scene = _scene()
    _add_table(scene)
    _add_box(scene, y=0.0, rigid_body=True)
    scene.build()

    coupler = scene.sim.coupler
    rigid_collection = coupler._scene.rigid_body
    rigid_collection.t[:] = torch.tensor([[0.2, -0.1, 0.7]], dtype=torch.float64, device="cuda")
    rigid_collection.quat[:] = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64, device="cuda")
    rigid_collection.v[:] = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device="cuda")
    rigid_collection.omega[:] = torch.tensor([[-0.4, 0.5, -0.6]], dtype=torch.float64, device="cuda")
    expected = tuple(
        buffer.detach().clone()
        for buffer in (
            rigid_collection.t,
            rigid_collection.quat,
            rigid_collection.v,
            rigid_collection.omega,
        )
    )

    def fail_reset(native_scene):
        native_scene.rigid_body.t.zero_()
        native_scene.rigid_body.quat.zero_()
        native_scene.rigid_body.v.zero_()
        native_scene.rigid_body.omega.zero_()
        raise RuntimeError("injected reset failure")

    monkeypatch.setattr(QIPCScene, "reset", fail_reset, raising=False)
    with pytest.raises(RuntimeError, match="injected reset failure"):
        coupler.reset()

    for actual, runtime in zip(
        (
            rigid_collection.t,
            rigid_collection.quat,
            rigid_collection.v,
            rigid_collection.omega,
        ),
        expected,
        strict=True,
    ):
        assert torch.equal(actual, runtime)


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_rigid_body_refuses_unsupported_couplings():
    scene = _scene()
    _add_table(scene)
    rigid_box = _add_box(scene, y=0.0, rigid_body=True)
    coupler = scene.sim.coupler

    with pytest.raises(gs.GenesisException, match="cannot be kinematically driven"):
        coupler.enable_soft_transform(rigid_box, strength=(1.0e5, 0.0))
    with pytest.raises(gs.GenesisException, match="initial-state override"):
        coupler.set_rigid_initial_state(
            rigid_box,
            body_q={rigid_box.links[0].name: np.zeros(12)},
            joint_theta={},
        )


@pytest.mark.required
@pytest.mark.precision("64")
def test_qipc_rigid_body_refuses_gravity_compensation():
    scene = _scene()
    _add_table(scene)
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, _TABLE_TOP + 0.05), size=(0.05, 0.05, 0.05), fixed=False),
        material=gs.materials.Rigid(
            rho=500.0,
            coup_friction=0.5,
            gravity_compensation=1.0,
            qipc_rigid_body=True,
        ),
    )

    with pytest.raises(gs.GenesisException, match="gravity"):
        scene.build()
