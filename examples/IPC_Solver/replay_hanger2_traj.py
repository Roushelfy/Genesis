"""
Replay a coat-hanger teleop trajectory (Sharpa hands, ipc_hanger).

Shirt and coat hanger start on the table. Robot picks up hanger, hangs it
on the drying rack, then drapes the shirt over it.

Usage:
    python examples/IPC_Solver/replay_hanger2_traj.py
    python examples/IPC_Solver/replay_hanger2_traj.py --render --nyx
    python examples/IPC_Solver/replay_hanger2_traj.py --render --camera-traj surround
    python examples/IPC_Solver/replay_hanger2_traj.py --loop --speed 2.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _replay_common import (
    EgoCamera,
    FullViewCamera,
    SurroundCamera,
    TrajectoryReplay,
)

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"
_HANGER = _DEMO / "coat_hanger"

MARVIN_URDF = str(_DEMO / "marvin_sharpa_description/marvin_sharpa.urdf")
TABLE_GLB = str(_HANGER / "work_table.glb")
SHIRT_GLB = str(_HANGER / "genesis_shirt.glb")
COAT_HANGER_GLB = str(_HANGER / "coat_hanger.glb")
RACK_GLB = str(_HANGER / "rack_frame.glb")
DEFAULT_TRAJ = str(_HANGER / "trajectory_sharpa.npz")


class Hanger2Replay(TrajectoryReplay):
    name = "hanger2"
    cam_pos = (0.53, -1.2, 1.3)
    cam_lookat = (0.53, 0.0, 1.0)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument(
            "--traj",
            type=str,
            default=DEFAULT_TRAJ,
            help="Path to trajectory.npz",
        )

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: MARVIN_SHARPA, 58 DOF
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, drying_rack)
        self._rigid_data = {
            "coat_hanger": traj["rigid_coat_hanger"],
        }

        # FEM data
        self._fem_data = {
            "shirt": traj["fem_shirt"],
        }

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        print(f"Coat hanger frames: {self._rigid_data['coat_hanger'].shape[0]}")
        print(f"Shirt: {self._fem_data['shirt'].shape[1]} verts")
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
            "full": lambda: FullViewCamera(
                pos=self.cam_pos,
                lookat=self.cam_lookat,
            ),
            "ego": lambda: EgoCamera(
                base_pos=(0.0, 0.0, 1.08),
                offset=(0.0, -0.15, 0.45),
                lookat_offset=(0.5, 0.0, -0.2),
            ),
        }
        if name in presets:
            return presets[name]()
        return super().make_camera_traj(name)

    def build_scene(self, scene):
        import genesis as gs

        # Ground
        scene.add_entity(gs.morphs.Plane())

        # Table (ipc_hanger: rotated 90deg, shifted)
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

        # Drying rack (fixed)
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

        # Rigid objects (matches registry COAT_HANGER for ipc_hanger)
        self._rigid_entities = {
            "coat_hanger": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.51, -0.25, 0.84),
                    euler=(180, 180, 0),
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
        }

        # Shirt (FEM Cloth, matches registry SHIRT_HANGER)
        self._fem_entities = {
            "shirt": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHIRT_GLB,
                    pos=(0.55, 0.25, 1.09),
                    euler=(-90, 150, 90),
                    scale=0.80,
                    fixed=False,
                ),
                material=gs.materials.FEM.Cloth(
                    E=1e3,
                    nu=0.49,
                    rho=60.0,
                    thickness=0.0005,
                    bending_stiffness=0.1,
                    friction_mu=0.3,
                ),
                surface=gs.surfaces.Plastic(),
                vis_mode="visual",
            ),
        }

        # Robot (MARVIN_SHARPA, 58 DOF, fixed base)
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
    Hanger2Replay().run()
