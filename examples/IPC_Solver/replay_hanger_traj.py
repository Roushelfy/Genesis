"""
Replay a coat-hanger teleop trajectory.

Usage:
    python examples/IPC_Solver/replay_hanger_traj.py
    python examples/IPC_Solver/replay_hanger_traj.py --render --nyx
    python examples/IPC_Solver/replay_hanger_traj.py --render --camera-traj surround
    python examples/IPC_Solver/replay_hanger_traj.py --loop --speed 2.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from _replay_common import (
    EgoCamera,
    FullViewCamera,
    SurroundCamera,
    TrajectoryReplay,
)

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"
_HANGER = _DEMO / "coat_hanger"

MARVIN_URDF = str(_DEMO / "marvin_wuji/robot_capsule.urdf")
TABLE_GLB = str(_HANGER / "work_table.glb")
SHIRT_GLB = str(_HANGER / "genesis_shirt.glb")
COAT_HANGER_GLB = str(_HANGER / "coat_hanger.glb")
RACK_GLB = str(_HANGER / "rack_frame.glb")
SHOE_BOX_GLB = str(_HANGER / "shoe_box.glb")
SHOE_BOX_LID_GLB = str(_HANGER / "shoe_box_lid.glb")
DEFAULT_TRAJ = str(_HANGER / "trajectory.npz")


class HangerReplay(TrajectoryReplay):
    name = "hanger"
    cam_pos = (0.53, -1.2, 1.3)
    cam_lookat = (0.53, 0.0, 1.0)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: 54 DOFs + 7 base qpos (free joint at pos 0,0,1.08)
        # self._base_qpos = np.array([0.0, 0.0, 1.08, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        raw_joint = traj["robot_qpos"]
        full_qpos = np.zeros((n_frames, raw_joint.shape[1]), dtype=np.float32)
        full_qpos[:, :] = raw_joint
        self._joint_qpos = full_qpos

        # Rigid + FEM data
        rigid_names = ["table", "shoe_box", "shoe_box_lid", "coat_hanger", "drying_rack"]
        self._rigid_data = {name: traj[f"rigid_{name}"] for name in rigid_names}
        self._fem_data = {"shirt": traj["shirt_particles"]}

        print(f"Robot qpos: {raw_joint.shape[1]} DOF, Shirt: {self._fem_data['shirt'].shape[1]} vertices")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=(0.53, 0.0, 1.0),
                radius=1.5,
                height=1.3,
                angle_start=-60,
                angle_end=60,
            ),
            "full": lambda: FullViewCamera(pos=self.cam_pos, lookat=self.cam_lookat),
            "ego": lambda: EgoCamera(
                base_pos=(0.0, 0.0, 1.08),
                offset=(0.0, -0.15, 0.45),
                lookat_offset=(0.5, 0.0, -0.2),
            ),
        }
        if name in presets:
            return presets[name]()
        return super().make_camera_traj(name)

    def apply_frame(self, scene, frame_idx):
        super().apply_frame(scene, frame_idx)
        if frame_idx == 0:
            ch = self._rigid_entities["coat_hanger"]
            print("[DEBUG] frame 0 coat_hanger:")
            print(f"  get_pos: {ch.get_pos().cpu().numpy().flatten()}")
            print(f"  set data: {self._rigid_data['coat_hanger'][0]}")
            # Check visual mesh verts directly
            verts = ch.get_verts().cpu().numpy()
            print(f"  verts shape: {verts.shape}")
            print(f"  verts Z range: [{verts[..., 2].min():.5f}, {verts[..., 2].max():.5f}]")

    def build_scene(self, scene):
        import genesis as gs

        # Ground
        scene.add_entity(gs.morphs.Plane())

        # Table (static)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.95, 0.10, 0.0),
                euler=(0, 0, 90),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            vis_mode="visual",
        )

        # Drying rack
        scene.add_entity(
            gs.morphs.Mesh(
                file=RACK_GLB,
                pos=(0.53, 0.0, 0.0),
                euler=(0, 0, 90),
                scale=0.83,
                fixed=True,
                file_meshes_are_zup=False,
                convexify=False,
                decimate=True,
                decimate_face_num=1000,
            ),
            material=gs.materials.Rigid(rho=100.0),
            vis_mode="visual",
        )

        # Rigid objects — must match gs-core registry exactly (convexify, decimate, rho)
        self._rigid_entities = {
            "coat_hanger": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.53, -0.45, 1.165),
                    euler=(90, 180, 0),
                    scale=0.004,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=gs.materials.Rigid(rho=80.0),
                vis_mode="visual",
            ),
            "shoe_box": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHOE_BOX_GLB,
                    pos=(0.40, 0.30, 0.79),
                    scale=0.6,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=gs.materials.Rigid(rho=1000.0),
                surface=gs.surfaces.Default(smooth=False),
                vis_mode="visual",
            ),
            "shoe_box_lid": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHOE_BOX_LID_GLB,
                    pos=(0.55, 0.30, 0.90),
                    scale=0.6,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=gs.materials.Rigid(rho=200.0),
                surface=gs.surfaces.Default(smooth=False),
                vis_mode="visual",
            ),
        }

        # FEM cloth
        self._fem_entities = {
            "shirt": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHIRT_GLB,
                    pos=(0.53, -0.37, 1.035),
                    euler=(-90, 180, 0),
                    scale=0.80,
                    fixed=False,
                ),
                material=gs.materials.FEM.Cloth(
                    E=1e3,
                    nu=0.49,
                    rho=60.0,
                    thickness=0.0005,
                    bending_stiffness=0.1,
                ),
                surface=gs.surfaces.Plastic(),
                vis_mode="visual",
            ),
        }

        # Robot
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=MARVIN_URDF,
                fixed=True,
                collision=False,
                pos=(0, 0, 1.08),
            ),
            vis_mode="visual",
        )


if __name__ == "__main__":
    HangerReplay().run()
