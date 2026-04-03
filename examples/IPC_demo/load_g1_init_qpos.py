"""Load G1 29-DOF URDF with symmetric initial qpos and cashmere sweater."""

from pathlib import Path

import numpy as np

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEATER_GLB = REPO_ROOT / "DemoAssets" / "sweater" / "Cashmere_Sweater_adjusted.glb"
G1_URDF = REPO_ROOT / "DemoAssets" / "g1_robot" / "assets" / "g1_29dof_rev_1_0.urdf"
ROBOT_Z = 0.75

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=True,
    rigid_options=gs.options.RigidOptions(dt=0.01),
)

scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.URDF(
        file=str(G1_URDF),
        pos=(0, 0, ROBOT_Z),
    ),
)

# Sweater mesh from USD transform:
#   translate = (0.0244, 0, -0.175)  relative to robot at origin
#   orient    = quat(w=0.707, x=0, y=0, z=0.707)  => 90 deg about Z
#   scale     = 0.81  (GLB is already in meters, only artistic scale needed)
#   rotateX   = 90 deg  (Y-up to Z-up conversion)
# Combined rotation: Rz(90) * Rx(90) => euler=(90, 0, 90) in extrinsic XYZ
sweater = scene.add_entity(
    morph=gs.morphs.Mesh(
        file=str(SWEATER_GLB),
        scale=0.81,
        pos=(0.0244, 0.0, ROBOT_Z - 0.175),
        euler=(0, 0, 90),
    ),
    material=gs.materials.Rigid(),
    surface=gs.surfaces.Plastic(
        opacity=0.2,
    ),
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

# for i in range(1000):
#     scene.step()
while True:
    scene._visualizer.update()
