"""Robot target controller used by the standalone QIPC tape example."""

from __future__ import annotations

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.couplers.qipc_coupler.tape_world import (
    HAND_JOINTS,
    QIPCTapeWorld,
    TapeWorldSide,
)


def _hand_postures(world: QIPCTapeWorld) -> dict[tuple[str, bool], np.ndarray]:
    postures: dict[tuple[str, bool], np.ndarray] = {}
    for side in ("right", "left"):
        for closed in (False, True):
            values = []
            for name in HAND_JOINTS[side]:
                finger = int(name.split("finger", 1)[1].split("_", 1)[0])
                joint = int(name.rsplit("joint", 1)[1])
                lower, upper = (float(value) for value in world.robot.get_joint(name).dofs_limit[0])
                if closed and finger == 1:
                    desired = (0.9, 0.35, 0.9, 0.9)[joint - 1]
                elif closed and finger == 2:
                    desired = (0.95, 0.0, 0.95, 0.95)[joint - 1]
                else:
                    desired = 0.08 if finger == 1 and joint == 1 else 0.0
                values.append(min(max(desired, lower), upper))
            postures[(side, closed)] = np.asarray(values, dtype=np.float64)
    return postures


class QIPCTapeRobotController:
    """Translate standalone palm and grip targets into Genesis robot commands."""

    def __init__(self, world: QIPCTapeWorld) -> None:
        self.world = world
        self._qpos_command = world.home_qpos.copy()
        self._hand_postures = _hand_postures(world)
        self._closed = {"right": False, "left": False}
        self._target_position: dict[str, np.ndarray] = {}
        self._target_quaternion: dict[str, np.ndarray] = {}
        self.reset_targets()

    def palm_target(self, side: TapeWorldSide) -> tuple[np.ndarray, np.ndarray]:
        return self._target_position[side].copy(), self._target_quaternion[side].copy()

    def move_palm_target(
        self,
        side: TapeWorldSide,
        delta_position: np.ndarray | tuple[float, float, float],
    ) -> None:
        delta = np.asarray(delta_position, dtype=gs.np_float)
        if delta.shape != (3,):
            raise ValueError(f"Palm position delta must have shape (3,), got {delta.shape}.")
        self._target_position[side] += delta

    def rotate_palm_target(self, side: TapeWorldSide, delta_euler: np.ndarray) -> None:
        delta = np.asarray(delta_euler, dtype=gs.np_float)
        if delta.shape != (3,):
            raise ValueError(f"Palm Euler delta must have shape (3,), got {delta.shape}.")
        self._target_quaternion[side] = gu.transform_quat_by_quat(
            self._target_quaternion[side],
            gu.xyz_to_quat(delta),
        )

    def reset_targets(self) -> None:
        for side in ("right", "left"):
            self._target_position[side] = self.world.palm_position(side)
            self._target_quaternion[side] = self.world.palm_quaternion(side)

    def reset(self) -> None:
        self._qpos_command[:] = self.world.home_qpos
        self._closed = {"right": False, "left": False}
        self.reset_targets()

    def set_grip(self, side: TapeWorldSide, closed: bool) -> None:
        self._closed[side] = closed

    def grip_is_closed(self, side: TapeWorldSide) -> bool:
        return self._closed[side]

    def apply(self) -> None:
        for side in ("right", "left"):
            arm_dofs = self.world.dofs[("arm", side)]
            qpos = self.world.robot.inverse_kinematics(
                link=self.world.palms[side],
                pos=self._target_position[side],
                quat=self._target_quaternion[side],
                init_qpos=self._qpos_command,
                max_samples=1,
                max_solver_iters=30,
                dofs_idx_local=arm_dofs,
            )
            qpos_array = qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos)
            self._qpos_command[arm_dofs] = qpos_array[arm_dofs]
            self.world.robot.control_dofs_position(
                self._qpos_command[arm_dofs],
                dofs_idx_local=arm_dofs,
            )
            self.world.robot.control_dofs_position(
                self._hand_postures[(side, self._closed[side])],
                dofs_idx_local=self.world.dofs[("hand", side)],
            )
