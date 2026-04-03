from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .loader import MotionSequence
from .uipc_driver import AffineStcDriver
from .urdf_kinematics import UrdfKinematics


@dataclass
class ReplayStatus:
    frame_index: int
    world_frame: int


class MotionReplayPlayer:
    def __init__(
        self,
        sequence: MotionSequence,
        urdf_kinematics: UrdfKinematics,
        driver: AffineStcDriver,
        loop: bool = True,
        warmup_start_joint: np.ndarray | None = None,
        warmup_frames: int = 0,
    ) -> None:
        self.sequence = sequence
        self.urdf_kinematics = urdf_kinematics
        self.driver = driver
        self.loop = bool(loop)
        self.warmup_frames = int(max(0, warmup_frames))
        self.warmup_start_joint = (
            None
            if warmup_start_joint is None
            else np.asarray(warmup_start_joint, dtype=np.float64).reshape(len(self.urdf_kinematics.joint_names))
        )
        self.last_status = ReplayStatus(frame_index=0, world_frame=0)

    @property
    def num_frames(self) -> int:
        return self.sequence.num_frames

    @property
    def current_root_transform(self) -> np.ndarray:
        return self.urdf_kinematics.root_transform

    def _index_from_world_frame(self, world_frame: int) -> int:
        replay_world_frame = int(max(world_frame - self.warmup_frames, 0))
        if self.num_frames <= 0:
            return 0
        if self.loop:
            return int(replay_world_frame % self.num_frames)
        return int(min(max(replay_world_frame, 0), self.num_frames - 1))

    def _apply_warmup(self, warmup_index: int, world_frame: int, snap_to_pose: bool) -> None:
        base = self.sequence.qpos_joint_order[0]
        root_pos = base[:3]
        root_quat = base[3:7]
        joint_clip0 = base[7:]
        if self.warmup_start_joint is None:
            joint_pos = joint_clip0
        else:
            alpha = 0.0
            if self.warmup_frames > 1:
                alpha = float(np.clip(warmup_index / float(self.warmup_frames - 1), 0.0, 1.0))
            joint_pos = (1.0 - alpha) * self.warmup_start_joint + alpha * joint_clip0
        self.urdf_kinematics.set_pose(root_pos=root_pos, root_quat_wxyz=root_quat, joint_pos=joint_pos)
        world_transforms = self.urdf_kinematics.node_world_transforms()
        self.driver.apply_target_transforms(world_transforms, snap_to_pose=snap_to_pose)
        self.last_status = ReplayStatus(frame_index=0, world_frame=int(world_frame))

    def apply_frame(self, frame_index: int, world_frame: int, snap_to_pose: bool = False) -> None:
        qpos = self.sequence.qpos_joint_order[frame_index]
        root_pos = qpos[:3]
        root_quat = qpos[3:7]
        joint_pos = qpos[7:]
        self.urdf_kinematics.set_pose(root_pos=root_pos, root_quat_wxyz=root_quat, joint_pos=joint_pos)
        world_transforms = self.urdf_kinematics.node_world_transforms()
        self.driver.apply_target_transforms(world_transforms, snap_to_pose=snap_to_pose)
        self.last_status = ReplayStatus(frame_index=int(frame_index), world_frame=int(world_frame))

    def apply_initial_pose(self, snap_to_pose: bool = True) -> None:
        if self.warmup_frames > 0 and self.warmup_start_joint is not None:
            self._apply_warmup(warmup_index=0, world_frame=0, snap_to_pose=snap_to_pose)
            return
        self.apply_frame(frame_index=0, world_frame=0, snap_to_pose=snap_to_pose)

    def animation_callback(self, info: Any) -> None:
        world_frame = max(int(info.frame()) - 1, 0)
        if self.warmup_frames > 0 and world_frame < self.warmup_frames:
            self._apply_warmup(warmup_index=world_frame, world_frame=world_frame, snap_to_pose=False)
            return
        frame_index = self._index_from_world_frame(world_frame)
        self.apply_frame(frame_index=frame_index, world_frame=world_frame)

    def bind_to_animator(self, scene: Any) -> None:
        if not self.driver.bindings:
            raise RuntimeError("Driver has no link bodies. Call create_bodies() first.")
        anchor_obj = self.driver.bindings[0].obj
        scene.animator().insert(anchor_obj, self.animation_callback)
