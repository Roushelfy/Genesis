"""
Replay a paper-folding teleop trajectory.

Usage:
    python examples/IPC_Solver/replay_paper_traj.py
    python examples/IPC_Solver/replay_paper_traj.py --render --nyx
    python examples/IPC_Solver/replay_paper_traj.py --render --camera-traj surround
    python examples/IPC_Solver/replay_paper_traj.py --loop --speed 2.0
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
_PAPER = _DEMO / "paper"

MARVIN_URDF = str(_DEMO / "marvin_wuji/robot_capsule.urdf")
TABLE_GLB = str(_PAPER / "work_table.glb")
PAPER_GLB = str(_PAPER / "genesis_paper.glb")
PRESS_OBJ = str(_PAPER / "brick_press.obj")
DEFAULT_TRAJ = str(_PAPER / "trajectory.npz")


class PaperReplay(TrajectoryReplay):
    name = "paper"
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

        # Robot: fixed base, 54 joint DOFs
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid + FEM data
        self._rigid_data = {
            "paper_press": traj["rigid_paper_press"],
        }
        self._fem_data = {
            "paper_sheet": traj["paper_particles"],
        }

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF, Paper: {self._fem_data['paper_sheet'].shape[1]} vertices")
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

        # Table
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.8, 0.0, 0.0),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            vis_mode="visual",
        )

        # Rigid objects
        self._rigid_entities = {
            "paper_press": scene.add_entity(
                gs.morphs.Mesh(
                    file=PRESS_OBJ,
                    pos=(0.49, 0.3, 0.79),
                    scale=1.0,
                    fixed=False,
                    convexify=False,
                    decimate=True,
                ),
                material=gs.materials.Rigid(rho=1500.0),
                surface=gs.surfaces.Default(smooth=False),
                vis_mode="visual",
            ),
        }

        # Paper sheet (FEM paper)
        self._fem_entities = {
            "paper_sheet": scene.add_entity(
                gs.morphs.Mesh(
                    file=PAPER_GLB,
                    pos=(0.53, 0.0, 0.78),
                    euler=(0, 0, -90),
                    scale=0.125,
                    fixed=False,
                ),
                material=gs.materials.FEM.Paper(
                    E=1e4,
                    nu=0.4,
                    rho=300.0,
                    thickness=0.001,
                    bending_stiffness=1e4,
                ),
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
    PaperReplay().run()
