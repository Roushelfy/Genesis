"""Reusable QIPC robot-world configuration and lifecycle tests."""

import numpy as np
import pytest

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC robot world requires 'quadrants' and 'qipc'.", allow_module_level=True)


def _world_module():
    from genesis.engine.couplers.qipc_coupler import robot_world

    return robot_world


def test_robot_world_config_maps_solver_options() -> None:
    robot_world = _world_module()
    options = robot_world.robot_world_coupler_options(
        robot_world.RobotWorldConfig(
            newton_velocity_tol=0.02,
            linear_tol_rate=0.004,
            linear_max_iter=321,
        )
    )

    assert options == {
        "contact_enable": False,
        "solver_newton_velocity_tol": 0.02,
        "solver_linear_tol_rate": 0.004,
        "solver_linear_max_iter": 321,
    }


def test_robot_world_uses_original_solver_defaults() -> None:
    robot_world = _world_module()
    config = robot_world.RobotWorldConfig()

    assert config.newton_velocity_tol == 0.01
    assert config.linear_tol_rate is None
    assert config.kappa_pivot == 1e7
    assert config.kappa_axis == 1e7


def test_robot_world_build_control_and_repeat_reset(show_viewer) -> None:
    robot_world = _world_module()
    world = robot_world.build_qipc_robot_world(robot_world.RobotWorldConfig(show_viewer=False))
    scene_identity = id(world.scene)
    initial_palm = world.palm_position("right")
    initial_hand = world.hand_dofs_position("right")
    target = initial_hand.copy()
    target[0] += 0.1
    world.robot.control_dofs_position(target, dofs_idx_local=world.dofs[("hand", "right")])

    for _ in range(3):
        stats = world.step()

    assert stats.newton_iters > 0
    assert np.linalg.norm(world.palm_position("right") - initial_palm) < 0.0025
    assert np.max(np.abs(world.hand_dofs_position("right") - initial_hand)) > 1e-4

    for _ in range(2):
        world.reset()
        assert id(world.scene) == scene_identity
        np.testing.assert_allclose(world.hand_dofs_position("right"), initial_hand, atol=1e-6)
        world.step()
