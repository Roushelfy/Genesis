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
LOCK_TAPE_ASSET_PATH = os.path.join(get_assets_dir(), "qipc", "tape_roll_lock.npz")

needs_soft_tape_asset = pytest.mark.skipif(
    not os.path.exists(SOFT_TAPE_ASSET_PATH),
    reason="soft wound-roll asset not found",
)
needs_tape_assets = pytest.mark.skipif(
    not os.path.exists(SOFT_TAPE_ASSET_PATH) or not os.path.exists(LOCK_TAPE_ASSET_PATH),
    reason="soft or locked wound-roll asset not found",
)


def _world_module():
    from genesis.engine.couplers.qipc_coupler import tape_world

    return tape_world


@needs_tape_assets
def test_tape_world_config_maps_soft_and_bond_options():
    tape_world = _world_module()
    locked_asset = tape_world.TapeAsset.from_npz(LOCK_TAPE_ASSET_PATH)
    soft_asset = tape_world.TapeAsset.from_npz(SOFT_TAPE_ASSET_PATH)

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
    world = tape_world.build_qipc_tape_world(
        tape_world.TapeWorldConfig(
            mode="soft",
            show_viewer=False,
            tape_asset_path=SOFT_TAPE_ASSET_PATH,
        )
    )
    scene_identity = id(world.scene)
    initial_tape = world.tape_positions()
    initial_palm = world.palm_position("right")
    initial_hand = world.hand_dofs_position("right")

    assert np.isfinite(initial_tape).all()
    assert world.hub is not None
    assert world.table is not None

    world.move_palm_target("right", (0.003, 0.0, 0.0))
    world.set_grip("right", True)
    for _ in range(3):
        stats = world.step()

    assert stats.newton_iters > 0
    assert stats.max_pcg_iters > 0
    assert np.linalg.norm(world.palm_position("right") - initial_palm) > 1e-4
    assert np.max(np.abs(world.hand_dofs_position("right") - initial_hand)) > 1e-2
    assert np.isfinite(world.tape_positions()).all()

    for _ in range(2):
        world.reset()
        assert id(world.scene) == scene_identity
        assert world.reset_error() == 0.0
        assert not world.grip_is_closed("right")
        assert np.isfinite(world.tape_positions()).all()
        world.step()
