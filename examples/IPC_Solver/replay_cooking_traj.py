"""
Replay a cooking teleop trajectory from NPZ.

Usage:
    python examples/IPC_Solver/replay_cooking_traj.py
    python examples/IPC_Solver/replay_cooking_traj.py --traj path/to/trajectory.npz
    python examples/IPC_Solver/replay_cooking_traj.py --render --camera-traj surround
    python examples/IPC_Solver/replay_cooking_traj.py --loop --speed 2.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _replay_common import (
    FullViewCamera,
    SurroundCamera,
    TrajectoryReplay,
    marvin_urdf,
)

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"

MARVIN_GSS_URDF = marvin_urdf("marvin_gss")
MARVIN_PIKA_URDF = marvin_urdf("marvin_pika")
PAN_USD = str(_DEMO / "cook/Pan025/Pan025.usd")
SPATULA_USD = str(_DEMO / "cook/Spatula018/Spatula018.usd")
DEFAULT_TRAJ = str(_DEMO / "cook_with_teleop/trajectory.npz")


class CookingReplay(TrajectoryReplay):
    name = "cooking"
    cam_pos = (1.5, -0.5, 1.5)
    cam_lookat = (0.6, 0.0, 0.85)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")
        parser.add_argument("--robot", type=str, default="gss", choices=["gss", "pika"], help="Robot type")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot qpos (54 DOF for GSS, 18 for Pika)
        raw_joint = traj["robot_qpos"]
        self._joint_qpos = raw_joint.astype(np.float32)

        # Rigid objects
        rigid_names = [k[len("rigid_") :] for k in traj if k.startswith("rigid_")]
        self._rigid_data = {name: traj[f"rigid_{name}"] for name in rigid_names}

        print(f"Robot qpos: {raw_joint.shape[1]} DOF")
        print(f"Rigid objects: {rigid_names}")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=(0.6, 0.0, 0.85),
                radius=1.5,
                height=1.5,
                angle_start=-30,
                angle_end=30,
            ),
            "full": lambda: FullViewCamera(pos=self.cam_pos, lookat=self.cam_lookat),
        }
        if name in presets:
            return presets[name]()
        return super().make_camera_traj(name)

    def build_scene(self, scene):
        import genesis as gs

        # Ground
        scene.add_entity(gs.morphs.Plane())

        # Pan
        self._rigid_entities = {
            "pan": scene.add_entity(
                gs.morphs.USD(
                    file=PAN_USD,
                    pos=(0.59, 0.0, 0.83),
                    euler=(0, 0, -90),
                    scale=1.2,
                    fixed=False,
                    convexify=False,
                    decimate=True,
                ),
                material=gs.materials.Rigid(rho=100.0),
                vis_mode="visual",
            ),
            "spatula": scene.add_entity(
                gs.morphs.USD(
                    file=SPATULA_USD,
                    pos=(0.49, 0.35, 0.80),
                    euler=(0, 0, -90),
                    scale=1.1,
                    fixed=False,
                    convexify=False,
                    decimate=True,
                ),
                material=gs.materials.Rigid(rho=100.0),
                vis_mode="visual",
            ),
        }

        # Robot
        robot_urdf = MARVIN_GSS_URDF if self.args.robot == "gss" else MARVIN_PIKA_URDF
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=robot_urdf,
                fixed=True,
                collision=False,
                pos=(0, 0, 1.08),
            ),
            vis_mode="visual",
        )


if __name__ == "__main__":
    CookingReplay().run()
