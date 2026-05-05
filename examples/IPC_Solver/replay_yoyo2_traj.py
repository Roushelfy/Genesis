"""
Replay gs-core yoyo teleop recordings (v5_init format).

Usage:
    python examples/IPC_Solver/replay_yoyo2_traj.py
    python examples/IPC_Solver/replay_yoyo2_traj.py --loop
    python examples/IPC_Solver/replay_yoyo2_traj.py --render --nyx
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import genesis as gs

from _replay_common import TrajectoryReplay, marvin_urdf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_YOYO_DIR = _REPO_ROOT / "DemoAssets" / "yoyo"
_SHARPA_URDF = marvin_urdf("marvin_sharpa")


class YoyoReplay(TrajectoryReplay):
    name = "yoyo2"
    cam_pos = (0.3, -0.8, 1.2)
    cam_lookat = (0.6, -0.1, 0.9)
    cam_fov = 50
    fps = 60

    def add_args(self, parser):
        parser.add_argument(
            "--traj-dir",
            type=str,
            default=str(_YOYO_DIR / "v5_init"),
            help="Directory containing trajectory.npz and meshes",
        )

    def load_trajectory(self) -> int:
        traj_dir = Path(self.args.traj_dir)
        data = np.load(traj_dir / "trajectory.npz")

        self._sim_times = data["sim_times"]
        self._robot_qpos_data = data["robot_qpos"]  # (N, 58)
        self._yoyo_pos = data.get("yoyo_pos")  # (N, 3)
        self._yoyo_quat = data.get("yoyo_quat")  # (N, 4)
        self._string_particles = data.get("yoyo_string_particles")  # (N, V, 3)

        self._traj_dir = traj_dir
        self._string_mesh = str(traj_dir / "yoyo_string.obj")
        self._ball_mesh = str(traj_dir / "yoyo_ball.obj")

        return len(self._sim_times)

    def build_scene(self, scene) -> None:
        # Robot (Sharpa)
        self._robot = scene.add_entity(
            morph=gs.morphs.URDF(
                file=_SHARPA_URDF,
                pos=(0.0, 0.0, 1.08),
                fixed=True,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # Yoyo ball
        if os.path.exists(self._ball_mesh):
            self._yoyo_ent = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=self._ball_mesh,
                    pos=(0.6, -0.12, 0.89),
                    euler=(90, 0, 0),
                    scale=1.2,
                    fixed=False,
                    convexify=False,
                    decimate=False,
                ),
                material=gs.materials.Rigid(rho=500),
                surface=gs.surfaces.Default(color=(0.0, 0.6, 0.6, 1.0)),
            )
            self._rigid_entities["yoyo"] = self._yoyo_ent

        # Yoyo string (FEM rope for visual)
        if os.path.exists(self._string_mesh):
            self._string_ent = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=self._string_mesh,
                    pos=(0.6, -0.12, 0.89),
                    euler=(90, 0, 0),
                    scale=1.2,
                ),
                material=gs.materials.FEM.Rope(E=5e9, rho=300, thickness=0.00024),
                surface=gs.surfaces.Default(color=(0.95, 0.95, 0.9, 1.0)),
            )
            self._fem_entities["yoyo_string"] = self._string_ent

        # Ground
        scene.add_entity(morph=gs.morphs.Plane())

    def post_build(self):
        # Robot qpos is already in Genesis DOF order (58 DOFs) — direct assignment
        self._joint_qpos = self._robot_qpos_data.astype(np.float32)

        # Rigid data: yoyo pos + quat → (N, 7)
        if self._yoyo_pos is not None and self._yoyo_quat is not None:
            self._rigid_data["yoyo"] = np.hstack(
                [
                    self._yoyo_pos,
                    self._yoyo_quat,
                ]
            ).astype(np.float32)

        # FEM data: string particle positions
        if self._string_particles is not None:
            self._fem_data["yoyo_string"] = self._string_particles.astype(np.float32)


if __name__ == "__main__":
    YoyoReplay().run()
