import numpy as np
import pytest
import torch

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs

_TABLE_TOP = 0.4
_BOX_HALF = 0.025
_DROP_CLEARANCE = 0.02


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
        torch.tensor([0.0, 0.1, drop_z], dtype=initial_pos.dtype),
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
