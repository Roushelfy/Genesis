"""
Replay the stitched full gear sequence in Genesis with a shoulder arm on the carrier.

The shoulder arm (marvin_pika, links Base_R/Link1_R/Link2_R/Link3_R) is mounted
on the carrier flange. Joint2_R follows the carrier's Z-rotation each frame.

Usage
-----
    python replay_full_sequence_with_arm.py [--traj gs_full_sequence.npz] [--loop]
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="pygltflib currently does not remove")

import sys
_REPO = Path(__file__).resolve().parents[2]
_IPC_SOLVER = _REPO / "examples" / "IPC_Solver"
if str(_IPC_SOLVER) not in sys.path:
    sys.path.insert(0, str(_IPC_SOLVER))

from _replay_common import TrajectoryReplay

_HERE = Path(__file__).resolve().parent
_DEMO = _REPO / "DemoAssets"
_GEAR = _DEMO / "planetary_gear"

MARVIN_URDF = str(_DEMO / "marvin_sharpa_description" / "marvin_sharpa.urdf")
PIKA_URDF   = str(_DEMO / "marvin_robot" / "urdf" / "marvin_pika.urdf")
DEFAULT_TRAJ = str(_HERE / "gs_full_sequence.npz")

MESH_SCALE = 0.0012
MODUL = 3
SUN_TEETH = 12
PLANET_TEETH = 9
NUM_PLANETS = 3
GEAR_WIDTH_MM = 12.0
GEAR_ROTATION_DEG = 270.0
ORBIT_R = MODUL * (SUN_TEETH + PLANET_TEETH) / 2 * MESH_SCALE
FLANGE_THICK_MM = 6.0

CX, CY, CZ = 0.5, 0.0, 0.79

SHOULDER_LINKS = {"Base_R", "Link1_R", "Link2_R", "Link3_R"}


def _planet_position(index: int) -> tuple[float, float, float]:
    orbit_angle = math.radians(index * 360.0 / NUM_PLANETS + GEAR_ROTATION_DEG)
    self_rot_deg = index * 360.0 * SUN_TEETH / PLANET_TEETH + GEAR_ROTATION_DEG
    tx = ORBIT_R * math.cos(orbit_angle)
    ty = ORBIT_R * math.sin(orbit_angle)
    return tx, ty, self_rot_deg


def _quat_wxyz_to_mat3(qw, qx, qy, qz):
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _carrier_z_angle(pose7):
    """Extract Z-rotation angle from carrier pose quaternion."""
    R = _quat_wxyz_to_mat3(*pose7[3:].astype(np.float64))
    return math.atan2(R[1, 0], R[0, 0])


class FullSequenceWithArmReplay(TrajectoryReplay):
    name = "gear_full_arm"
    cam_pos = (0.5, -0.8, 1.3)
    cam_lookat = (CX, CY, CZ)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ)

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        self._joint_qpos = traj["robot_qpos"].astype(np.float32)
        self._carrier_poses = traj["rigid_carrier"]

        self._rigid_data = {
            "sun_gear": traj["rigid_sun_gear"],
            "carrier": traj["rigid_carrier"],
            "planet_gear_0": traj["rigid_planet_gear_0"],
            "planet_gear_1": traj["rigid_planet_gear_1"],
            "planet_gear_2": traj["rigid_planet_gear_2"],
        }

        # Compute carrier initial Z angle for relative rotation
        self._carrier_angle_0 = _carrier_z_angle(self._carrier_poses[0])

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        for name, data in self._rigid_data.items():
            print(f"  {name}: {data.shape[0]} frames")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def build_scene(self, scene):
        import genesis as gs

        gear_assets = str(_GEAR / "assets")
        rot_off = GEAR_ROTATION_DEG
        rigid_mat = gs.materials.Rigid(rho=1000.0)

        # Table
        table_glb = str(_DEMO / "coat_hanger" / "work_table.glb")
        scene.add_entity(
            gs.morphs.Mesh(
                file=table_glb, pos=(0.8, 0.0, 0.0),
                scale=(1.14, 1.0, 1.47), fixed=True,
                file_meshes_are_zup=True, convexify=False,
            ),
            surface=gs.surfaces.BSDF(roughness=0.45, metallic=0.0, ior=1.45),
            vis_mode="visual",
        )

        # Ring gear
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/ring_gear.glb",
                pos=(CX, CY, CZ), euler=(0, 0, rot_off), scale=MESH_SCALE,
                fixed=True, convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # Sun gear
        self._rigid_entities["sun_gear"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/sun_gear_handle_v2.glb",
                pos=(CX, CY, CZ), euler=(0, 0, rot_off), scale=MESH_SCALE,
                fixed=False, convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # Planet gears
        for i in range(NUM_PLANETS):
            tx, ty, self_rot_deg = _planet_position(i)
            self._rigid_entities[f"planet_gear_{i}"] = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"{gear_assets}/planet_gear_v2.glb",
                    pos=(CX + tx, CY + ty, CZ), euler=(0, 0, self_rot_deg),
                    scale=MESH_SCALE, fixed=False, convexify=False, decimate=False,
                ),
                material=rigid_mat, vis_mode="visual",
            )

        # Carrier
        carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
        self._rigid_entities["carrier"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/carrier.glb",
                pos=(CX, CY, CZ + carrier_tz), euler=(0, 0, rot_off),
                scale=MESH_SCALE, fixed=False, convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # Support pin
        support_pin_tz = -12.0 * MESH_SCALE
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/support_pin.obj",
                pos=(CX, CY, CZ + support_pin_tz), scale=MESH_SCALE,
                fixed=True, convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # MARVIN_SHARPA robot (teleop arms)
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=MARVIN_URDF, fixed=True, collision=False,
                pos=(0, 0, 1.08),
            ),
            vis_mode="visual",
        )

        # Shoulder arm (marvin_pika) — mounted on carrier flange.
        # Placed at origin; post_build computes the correct root transform
        # so Joint2_R aligns with the carrier rotation axis.
        self._shoulder_robot = scene.add_entity(
            gs.morphs.URDF(
                file=PIKA_URDF, fixed=True, collision=False,
                pos=(0, 0, 0), scale=1.5,
            ),
            surface=gs.surfaces.BSDF(
                color=(0.40, 0.52, 0.65),
                roughness=0.3,
                metallic=0.2,
                opacity=0.45,
            ),
            vis_mode="visual",
        )

    def post_build(self):
        super().post_build()

        # Compute root_transform so Joint2_R sits at carrier flange centre.
        # Same logic as sim_planetary.py::load_robot_reference.
        shoulder = self._shoulder_robot

        # Get Joint2_R local transform with all joints at 0
        shoulder.set_qpos(np.zeros(shoulder.n_qs))

        # Find Link2_R world transform (with root at identity)
        T_j2_local = None
        for link in shoulder.links:
            if link.name == "Link2_R":
                T_j2_local = np.array(link.get_pos_quat_matrix()).reshape(4, 4) if hasattr(link, 'get_pos_quat_matrix') else None
                break

        if T_j2_local is None:
            # Fallback: just get link pos/quat
            for link in shoulder.links:
                if link.name == "Link2_R":
                    pos = np.array(link.get_pos())
                    quat = np.array(link.get_quat())  # wxyz
                    T_j2_local = np.eye(4, dtype=np.float64)
                    T_j2_local[:3, :3] = _quat_wxyz_to_mat3(*quat)
                    T_j2_local[:3, 3] = pos
                    break

        if T_j2_local is not None:
            # T_gear: carrier center in world = (CX, CY, CZ + carrier_tz)
            carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
            T_gear = np.eye(4, dtype=np.float64)
            T_gear[0, 3] = CX
            T_gear[1, 3] = CY
            T_gear[2, 3] = CZ + carrier_tz

            root_tf = T_gear @ np.linalg.inv(T_j2_local)
            shoulder.set_pos(root_tf[:3, 3])
            from scipy.spatial.transform import Rotation as Rot
            r = Rot.from_matrix(root_tf[:3, :3])
            qx, qy, qz, qw = r.as_quat()
            shoulder.set_quat(np.array([qw, qx, qy, qz]))
            print(f"[shoulder] root_pos = {np.round(root_tf[:3, 3], 4)}")

        # Find Joint2_R local qpos index
        self._j2_local_idx = None
        first_qs = self._shoulder_robot.joints[0].qs_idx
        first_start = first_qs[0] if isinstance(first_qs, (list, tuple)) else int(first_qs)
        for joint in shoulder.joints:
            if joint.name == "Joint2_R":
                j_qs = joint.qs_idx
                j_start = j_qs[0] if isinstance(j_qs, (list, tuple)) else int(j_qs)
                self._j2_local_idx = j_start - first_start
                break
        print(f"[shoulder] Joint2_R local_idx = {self._j2_local_idx}")

    def apply_frame(self, scene, frame_idx):
        super().apply_frame(scene, frame_idx)

        # Sync shoulder Joint2_R with carrier Z-rotation
        if self._j2_local_idx is not None:
            carrier_angle = _carrier_z_angle(self._carrier_poses[frame_idx])
            relative_angle = carrier_angle - self._carrier_angle_0
            qpos = self._shoulder_robot.get_qpos()
            if hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy().copy()
            else:
                qpos = np.array(qpos, dtype=np.float64).copy()
            if 0 <= self._j2_local_idx < len(qpos):
                qpos[self._j2_local_idx] = relative_angle
            self._shoulder_robot.set_qpos(qpos)


if __name__ == "__main__":
    FullSequenceWithArmReplay().run()
