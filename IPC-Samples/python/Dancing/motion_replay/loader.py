from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


DEFAULT_G1_DOF_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@dataclass
class MotionSequence:
    episode: str
    qpos_joint_order: np.ndarray  # shape (T, 36), root(7)+dofs(29)

    @property
    def num_frames(self) -> int:
        return int(self.qpos_joint_order.shape[0])


def _available_episodes(data: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(set(k.split("/")[0] for k in data.files if "/" in k and not k.startswith("__")))


def _build_dof_remap(target_joint_names: list[str], source_dof_order: list[str]) -> list[int]:
    return [source_dof_order.index(name) for name in target_joint_names]


def _remap_qpos(qpos_source: np.ndarray, remap: list[int]) -> np.ndarray:
    root = qpos_source[:, :7]
    dofs_source = qpos_source[:, 7:]
    dofs_target = dofs_source[:, remap]
    return np.concatenate([root, dofs_target], axis=1)


def load_motion_sequence(
    npz_path: str | Path,
    episode: str,
    target_joint_names: list[str],
    source_dof_order: list[str] | None = None,
) -> MotionSequence:
    source_order = source_dof_order if source_dof_order is not None else DEFAULT_G1_DOF_ORDER
    npz_file = Path(npz_path)
    if not npz_file.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")

    data = np.load(npz_file, allow_pickle=True)
    key = f"{episode}/state/qpos"
    if key not in data:
        episodes = _available_episodes(data)
        raise KeyError(f"Episode '{episode}' not found in {npz_file}. Available episodes: {episodes}")

    qpos = np.asarray(data[key], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")

    remap = _build_dof_remap(target_joint_names, source_order)
    qpos_remapped = _remap_qpos(qpos, remap)
    return MotionSequence(episode=episode, qpos_joint_order=qpos_remapped)


def load_joint_pose_vector(
    json_path: str | Path,
    joint_names: list[str],
    fallback_joint_pos: np.ndarray | None = None,
) -> np.ndarray:
    pose_file = Path(json_path)
    if not pose_file.exists():
        raise FileNotFoundError(f"Joint pose file not found: {pose_file}")

    payload = json.loads(pose_file.read_text(encoding="utf-8"))
    raw = payload.get("joint_angles", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid joint pose json format: {pose_file}")

    out = np.zeros((len(joint_names),), dtype=np.float64)
    if fallback_joint_pos is not None:
        fallback = np.asarray(fallback_joint_pos, dtype=np.float64).reshape(len(joint_names))
        out[:] = fallback

    for i, joint_name in enumerate(joint_names):
        if joint_name in raw:
            out[i] = float(raw[joint_name])
    return out
