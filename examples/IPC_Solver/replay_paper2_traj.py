"""
Replay a paper-folding teleop trajectory (Sharpa hands).

Usage:
    python examples/IPC_Solver/replay_paper2_traj.py
    python examples/IPC_Solver/replay_paper2_traj.py --render --nyx
    python examples/IPC_Solver/replay_paper2_traj.py --render --camera-traj surround
    python examples/IPC_Solver/replay_paper2_traj.py --loop --speed 2.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _replay_common import (
    EgoCamera,
    FullViewCamera,
    SurroundCamera,
    TrajectoryReplay,
    marvin_urdf,
)

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"
_PAPER = _DEMO / "paper"

MARVIN_URDF = marvin_urdf("marvin_sharpa")
TABLE_GLB = str(_PAPER / "work_table.glb")
PAPER_GLB = str(_PAPER / "paper_plane_coarse.glb")
DEFAULT_TRAJ = str(_PAPER / "trajectory_sharpa.npz")


class Paper2Replay(TrajectoryReplay):
    name = "paper2"
    cam_pos = (0.5, -0.6, 1.1)
    cam_lookat = (0.5, 0.0, 0.78)
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

        # No rigid objects in this recording (no paper_press)
        self._rigid_data = {}

        # FEM data
        self._fem_data = {
            "paper_sheet": traj["fem_paper_sheet"],
        }

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        print(f"Paper: {self._fem_data['paper_sheet'].shape[1]} verts")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=(0.5, 0.0, 0.78),
                radius=0.7,
                height=1.1,
                angle_start=-60,
                angle_end=60,
            ),
            "full": lambda: FullViewCamera(
                pos=(0.5, -0.6, 1.1),
                lookat=(0.5, 0.0, 0.78),
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

        # Table (ipc_paper: pos raised +0.02)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.8, 0.0, 0.02),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            vis_mode="visual",
        )

        # No rigid entities in this recording
        self._rigid_entities = {}

        # Paper sheet (FEM Paper, matches registry PAPER_SHEET)
        self._fem_entities = {
            "paper_sheet": scene.add_entity(
                gs.morphs.Mesh(
                    file=PAPER_GLB,
                    pos=(0.53, 0.0, 0.80),
                    euler=(0, 0, -90),
                    scale=0.125,
                    fixed=False,
                    file_meshes_are_zup=False,
                ),
                material=gs.materials.FEM.Paper(
                    E=1e7,
                    nu=0.3,
                    rho=300.0,
                    thickness=0.0002,
                    plasticity_model="stress",
                    bending_stiffness=8e4,
                    yield_stress=1.8e5,
                    hardening_modulus=0.001,
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
    Paper2Replay().run()
