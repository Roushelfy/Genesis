"""Reusable QIPC tape-world configuration and lifecycle tests."""

import os

import numpy as np
import pytest

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC tape world requires 'quadrants' and 'qipc'.", allow_module_level=True)

from genesis.utils.misc import get_assets_dir

SOFT_TAPE_ASSET_PATH = os.path.join(get_assets_dir(), "qipc", "tape_roll_soft.npz")
DISTANCE_BOND_TAPE_ASSET_PATH = os.path.join(get_assets_dir(), "qipc", "tape_roll_distance_bond.npz")

needs_soft_tape_asset = pytest.mark.skipif(
    not os.path.exists(SOFT_TAPE_ASSET_PATH),
    reason="soft wound-roll asset not found",
)
needs_tape_assets = pytest.mark.skipif(
    not os.path.exists(SOFT_TAPE_ASSET_PATH) or not os.path.exists(DISTANCE_BOND_TAPE_ASSET_PATH),
    reason="soft or distance-bond wound-roll asset not found",
)


def _world_module():
    from genesis.engine.couplers.qipc_coupler import tape_world

    return tape_world


def test_tape_world_uses_original_solver_defaults():
    tape_world = _world_module()
    config = tape_world.TapeWorldConfig()

    assert config.newton_velocity_tol == 0.01
    assert config.linear_tol_rate is None
    assert config.kappa_pivot == 1e7
    assert config.kappa_axis == 1e7


@needs_tape_assets
def test_tape_world_config_maps_soft_and_bond_options():
    tape_world = _world_module()
    assert tape_world.resolve_tape_asset_path(tape_world.TapeWorldConfig(mode="bond")) == DISTANCE_BOND_TAPE_ASSET_PATH
    locked_asset = tape_world.TapeAsset.from_npz(DISTANCE_BOND_TAPE_ASSET_PATH)
    soft_asset = tape_world.TapeAsset.from_npz(SOFT_TAPE_ASSET_PATH)
    assert bool(int(soft_asset.params["LOCK"])) is False
    assert soft_asset.bond_topos is None

    bond = tape_world.tape_world_coupler_options(
        tape_world.TapeWorldConfig(
            mode="bond",
            newton_velocity_tol=0.02,
            linear_tol_rate=0.004,
            linear_max_iter=321,
        ),
        locked_asset,
    )
    assert bond["adhesion_bond_distance_lock"] is True
    assert bond["solver_newton_velocity_tol"] == 0.02
    assert bond["solver_linear_tol_rate"] == 0.004
    assert bond["solver_linear_max_iter"] == 321

    soft = tape_world.tape_world_coupler_options(
        tape_world.TapeWorldConfig(mode="soft"),
        soft_asset,
    )
    assert soft["adhesion_bond_distance_lock"] is False
    assert soft["adhesion_bond_max_bonds"] == 0


@needs_soft_tape_asset
def test_tape_world_build_control_and_repeat_reset(show_viewer):
    tape_world = _world_module()
    from genesis.engine.couplers.qipc_coupler.marvin_wuji import ARM_JOINTS
    from genesis.engine.couplers.qipc_coupler.tape_world_controller import (
        QIPCTapeRobotController,
    )

    world = tape_world.build_qipc_tape_world(
        tape_world.TapeWorldConfig(
            mode="soft",
            sticky_hands=False,
            show_viewer=False,
            tape_asset_path=SOFT_TAPE_ASSET_PATH,
        )
    )
    controller = QIPCTapeRobotController(world)
    scene_identity = id(world.scene)
    initial_tape = world.tape_positions()
    initial_tape_velocity = world.tape_velocities()
    initial_palm = world.palm_position("right")
    initial_hand = world.hand_dofs_position("right")
    arm_q_indices = [
        world.robot.get_joint(name).qs_idx_local[0] for side in ("right", "left") for name in ARM_JOINTS[side]
    ]

    def assert_joint_qpos_matches_dofs():
        qpos = world.robot.get_qpos().reshape(-1).cpu().numpy()
        dofs = world.robot.get_dofs_position(
            dofs_idx_local=[index for side in ("right", "left") for index in world.dofs[("arm", side)]]
        )
        np.testing.assert_allclose(qpos[arm_q_indices], dofs.cpu().numpy(), atol=1e-6)

    assert np.isfinite(initial_tape).all()
    assert initial_tape_velocity.shape == initial_tape.shape
    assert np.isfinite(initial_tape_velocity).all()
    assert world.hub is not None
    assert world.table is not None
    assert float(world.table.get_pos().reshape(-1)[1].item()) == pytest.approx(0.0, abs=1e-7)
    assert float(world.hub.get_pos().reshape(-1)[1].item()) == pytest.approx(0.0, abs=1e-7)
    assert not world.scene.sim.coupler.adhesion.has_bond_seed_requests()
    hand_opt_out = world.scene.sim.coupler.adhesion._requests[-1]
    table_opt_out = world.scene.sim.coupler.adhesion._requests[-2]
    assert table_opt_out.source is world.tape
    assert table_opt_out.targets == (world.table,)
    assert table_opt_out.enabled is False
    assert table_opt_out.distance_lock is False
    assert hand_opt_out.source is world.tape
    assert hand_opt_out.targets == (world.robot,)
    assert hand_opt_out.enabled is False
    assert hand_opt_out.distance_lock is False
    assert_joint_qpos_matches_dofs()

    controller.move_palm_target("right", (0.003, 0.0, 0.0))
    controller.set_grip("right", True)
    for _ in range(3):
        controller.apply()
        stats = world.step()

    assert stats.newton_iters > 0
    assert stats.max_pcg_iters > 0
    assert np.linalg.norm(world.palm_position("right") - initial_palm) > 1e-4
    assert np.max(np.abs(world.hand_dofs_position("right") - initial_hand)) > 1e-2
    assert_joint_qpos_matches_dofs()
    assert np.isfinite(world.tape_positions()).all()
    assert np.isfinite(world.tape_velocities()).all()

    for _ in range(2):
        world.reset()
        controller.reset()
        assert id(world.scene) == scene_identity
        assert world.reset_error() == 0.0
        assert not controller.grip_is_closed("right")
        assert_joint_qpos_matches_dofs()
        assert np.isfinite(world.tape_positions()).all()
        assert np.isfinite(world.tape_velocities()).all()
        controller.apply()
        world.step()
