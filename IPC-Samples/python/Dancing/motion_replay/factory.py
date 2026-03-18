from __future__ import annotations

from pathlib import Path
import numpy as np

from .loader import DEFAULT_G1_DOF_ORDER, load_joint_pose_vector, load_motion_sequence
from .player import MotionReplayPlayer
from .uipc_driver import AffineStcDriver
from .urdf_kinematics import UrdfKinematics


def create_motion_replay_player(
    scene,
    urdf_path: str | Path,
    npz_path: str | Path,
    *,
    episode: str = "demo_0",
    source_dof_order: list[str] | None = None,
    warmup_joint_json_path: str | Path | None = None,
    warmup_frames: int = 60,
    loop: bool = True,
    object_prefix: str = "robot_link",
    kappa_mpa: float = 80.0,
    stc_strength: np.ndarray | None = None,
    shell_thickness: float = 0.0001,
    enable_robot_robot_contact: bool = False,
    enable_default_robot_contact: bool = True,
    default_robot_friction: float = 0.5,
    default_robot_resistance: float = 1.0e9,
    ignore_link_patterns: list[str] | None = None,
    bind_animator: bool = True,
) -> MotionReplayPlayer:
    urdf_kinematics = UrdfKinematics(urdf_path)
    sequence = load_motion_sequence(
        npz_path=npz_path,
        episode=episode,
        target_joint_names=urdf_kinematics.joint_names,
        source_dof_order=source_dof_order if source_dof_order is not None else DEFAULT_G1_DOF_ORDER,
    )

    driver = AffineStcDriver(
        scene=scene,
        object_prefix=object_prefix,
        kappa_mpa=kappa_mpa,
        stc_strength=stc_strength,
        shell_thickness=shell_thickness,
    )
    driver.create_bodies(urdf_kinematics.mesh_nodes, ignore_link_patterns=ignore_link_patterns)
    driver.configure_contact_policy(
        enable_robot_robot_contact=enable_robot_robot_contact,
        enable_default_robot_contact=enable_default_robot_contact,
        default_robot_friction=default_robot_friction,
        default_robot_resistance=default_robot_resistance,
    )

    warmup_joint = None
    warmup_path = Path(warmup_joint_json_path) if warmup_joint_json_path is not None else None
    if warmup_path is not None and warmup_path.exists():
        warmup_joint = load_joint_pose_vector(
            json_path=warmup_path,
            joint_names=urdf_kinematics.joint_names,
            fallback_joint_pos=sequence.qpos_joint_order[0, 7:],
        )

    player = MotionReplayPlayer(
        sequence=sequence,
        urdf_kinematics=urdf_kinematics,
        driver=driver,
        loop=loop,
        warmup_start_joint=warmup_joint,
        warmup_frames=warmup_frames if warmup_joint is not None else 0,
    )
    player.apply_initial_pose(snap_to_pose=True)
    if bind_animator:
        player.bind_to_animator(scene)
    return player
