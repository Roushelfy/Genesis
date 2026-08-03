"""Free-base Wuji hands QIPC world tests."""

import glob
import os

import numpy as np
import pytest

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC Wuji hands world requires 'quadrants' and 'qipc'.", allow_module_level=True)


def _world_module():
    from genesis.engine.couplers.qipc_coupler import wuji_hands_world

    return wuji_hands_world


def _hand_urdf(side: str) -> str:
    matches = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/datasets--Genesis-Intelligence--internal_assets/"
                f"snapshots/*/wujihand-urdf/urdf/{side}/{side}.urdf"
            )
        )
    )
    if not matches:
        pytest.skip("Wuji hand URDF assets are not installed.")
    return matches[-1]


def _config(**kwargs):
    world_module = _world_module()
    values = {
        "right_urdf_path": _hand_urdf("right"),
        "left_urdf_path": _hand_urdf("left"),
        "right_wrist_position": (0.37896985, -0.24957745, 0.91086233),
        "right_wrist_quaternion": (-0.63471603, -0.24809101, -0.66627967, -0.30275056),
        "left_wrist_position": (0.37896705, 0.24956885, 0.91085660),
        "left_wrist_quaternion": (0.63470954, -0.24807896, 0.66628915, -0.30275330),
        "show_viewer": False,
    }
    values.update(kwargs)
    return world_module.WujiHandsWorldConfig(**values)


def test_wuji_hands_world_config_maps_solver_options() -> None:
    world_module = _world_module()
    options = world_module.wuji_hands_coupler_options(
        _config(newton_velocity_tol=0.02, linear_tol_rate=0.003, linear_max_iter=456)
    )

    assert options == {
        "contact_enable": False,
        "solver_newton_velocity_tol": 0.02,
        "solver_linear_tol_rate": 0.003,
        "solver_linear_max_iter": 456,
    }


def test_wuji_hands_world_build_control_and_reset(show_viewer) -> None:
    world_module = _world_module()
    config = _config()
    world = world_module.build_qipc_wuji_hands_world(config)

    assert set(world.entities) == {"right", "left"}
    assert world.hands["right"].finger_dofs == tuple(range(6, 26))
    assert world.hands["left"].finger_dofs == tuple(range(6, 26))
    np.testing.assert_allclose(world.palm_position("right"), config.right_wrist_position, atol=1e-5)
    np.testing.assert_allclose(world.palm_position("left"), config.left_wrist_position, atol=1e-5)

    right_target = np.asarray(config.right_wrist_position) + np.array([0.01, 0.0, 0.0])
    world.set_wrist_target("right", right_target, config.right_wrist_quaternion)
    for _ in range(3):
        stats = world.step()

    assert stats.newton_iters > 0
    assert np.linalg.norm(world.palm_position("right") - right_target) < 0.003
    assert np.linalg.norm(world.palm_position("left") - config.left_wrist_position) < 0.003

    world.reset()
    np.testing.assert_allclose(world.palm_position("right"), config.right_wrist_position, atol=1e-5)
    np.testing.assert_allclose(world.palm_position("left"), config.left_wrist_position, atol=1e-5)
    np.testing.assert_allclose(world.finger_dofs_position("right"), world.hands["right"].home_finger_qpos, atol=1e-6)
    np.testing.assert_allclose(world.finger_dofs_position("left"), world.hands["left"].home_finger_qpos, atol=1e-6)
