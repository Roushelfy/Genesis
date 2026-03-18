"""
This script is used to load the G1 29-DOF URDF with symmetric initial qpos (no cloth) and export the IPC surface.

NEXT STEP: Use houdini to place the cloth on the robot.
"""

from pathlib import Path

import numpy as np
import genesis as gs

gs.init(backend=gs.cpu)

from genesis.engine.couplers.ipc_coupler.coupler import GenesisSolverContext, IPCBeforeWorldInitContext

ROBOT_Z = 0.75
REPO_ROOT = Path(__file__).resolve().parents[2]
G1_URDF = REPO_ROOT / "DemoAssets" / "locomotion" / "assets" / "g1_29dof_rev_1_0.urdf"

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    fem_options=gs.options.FEMOptions(
        use_rigid_compatible_transform=False,
    ),
    coupler_options=gs.options.IPCCouplerOptions(
        contact_enable=False,
        enable_rigid_rigid_contact=False,
        ignore_end_effector_check=True,
        _export_ipc_surface=True,
        _export_surface_dir="outputs/G1_QPos",
    ),
    show_viewer=True,
)

scene.add_entity(
    gs.morphs.Plane(),
    material=gs.materials.Rigid(coup_type="ipc_only"),
    name="ground_plane",
)

robot = scene.add_entity(
    gs.morphs.URDF(
        file=str(G1_URDF),
        pos=(0, 0, ROBOT_Z),
    ),
    material=gs.materials.Rigid(
        coup_type="two_way_soft_constraint",
    ),
    name="g1_robot",
)

scene.build()

# Joint angles in degrees, mirrored left -> right.
# Roll and yaw joints negate, pitch and elbow keep sign.
joint_angles_deg = {
    # Left leg
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": -9.0,
    "left_ankle_roll_joint": -5.7,
    # Right leg (mirrored)
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": -9.0,
    "right_ankle_roll_joint": 5.7,
    # Waist
    "waist_yaw_joint": 1.4,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # Left arm
    "left_shoulder_pitch_joint": 4.2,
    "left_shoulder_roll_joint": 37.7,
    "left_shoulder_yaw_joint": 20.1,
    "left_elbow_joint": 72.3,
    "left_wrist_roll_joint": -39.7,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    # Right arm (mirrored)
    "right_shoulder_pitch_joint": 4.2,
    "right_shoulder_roll_joint": -37.7,
    "right_shoulder_yaw_joint": -20.1,
    "right_elbow_joint": 72.3,
    "right_wrist_roll_joint": 39.7,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Build qpos: [base_pos(3), base_quat(4), dof_angles(29)]
qpos = np.zeros(robot.n_qs)
# Base position
qpos[0:3] = [0.0, 0.0, 0.75]
# Base quaternion (w, x, y, z)
qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
# Set each joint by name
for name, deg in joint_angles_deg.items():
    idx = robot.get_joint(name).qs_idx_local[0]
    qpos[idx] = np.deg2rad(deg)

robot.set_qpos(qpos)

scene.step()