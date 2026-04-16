"""
Replay a trashbag teleop trajectory (Sharpa hands, aerodynamic damping).

Usage:
    python examples/IPC_Solver/replay_trashbag2_traj.py
    python examples/IPC_Solver/replay_trashbag2_traj.py --render --nyx
    python examples/IPC_Solver/replay_trashbag2_traj.py --render --camera-traj surround
    python examples/IPC_Solver/replay_trashbag2_traj.py --loop --speed 2.0
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
_TRASH = _DEMO / "trashbag"

MARVIN_URDF = str(_DEMO / "marvin_sharpa_description/marvin_sharpa.urdf")
TABLE_GLB = str(_TRASH / "work_table.glb")
TRASHBAG_GLB = str(_TRASH / "Trashbag_rope.glb")
TRASHCAN_OBJ = str(_TRASH / "Trashcan033.obj")
SODA_CAN_GLB = str(_TRASH / "soda_can.glb")
PLASTIC_BOTTLE_GLB = str(_TRASH / "plastic_bottle.glb")
ROPE1_OBJ = str(_TRASH / "rope1.obj")
ROPE2_OBJ = str(_TRASH / "rope2.obj")
DEFAULT_TRAJ = str(_TRASH / "trajectory_sharpa.npz")


class Trashbag2Replay(TrajectoryReplay):
    name = "trashbag2"
    cam_pos = (0.7, -1.0, 1.3)
    cam_lookat = (0.7, 0.0, 0.9)
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

        # Robot: MARVIN_SHARPA, 58 DOF (fixed base in replay)
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, trashcan)
        self._rigid_data = {
            "soda_can": traj["rigid_soda_can"],
            "plastic_bottle": traj["rigid_plastic_bottle"],
        }

        # FEM data
        self._fem_data = {
            "trashbag": traj["fem_trashbag"],
            "rope1": traj["fem_rope1"],
            "rope2": traj["fem_rope2"],
        }

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        print(f"Trashbag: {self._fem_data['trashbag'].shape[1]} verts")
        print(f"Rope1: {self._fem_data['rope1'].shape[1]} verts")
        print(f"Rope2: {self._fem_data['rope2'].shape[1]} verts")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=(0.7, 0.0, 0.9),
                radius=1.2,
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
                lookat_offset=(0.7, 0.0, -0.2),
            ),
        }
        if name in presets:
            return presets[name]()
        return super().make_camera_traj(name)

    def build_scene(self, scene):
        import genesis as gs

        # Ground
        scene.add_entity(gs.morphs.Plane())

        # Table (matches MORPH_TABLE_IPC in registry)
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

        # Trashcan (fixed, uses .obj not .glb, Z-up)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TRASHCAN_OBJ,
                pos=(0.60, -0.15, 0.92),
                scale=0.8,
                fixed=True,
                convexify=True,
                decimate=True,
                decimate_face_num=1000,
            ),
            material=gs.materials.Rigid(rho=100.0),
            vis_mode="visual",
        )

        # Rigid objects (positions match registry)
        self._rigid_entities = {
            "soda_can": scene.add_entity(
                gs.morphs.Mesh(
                    file=SODA_CAN_GLB,
                    pos=(0.47, -0.28, 0.84),
                    scale=0.036,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=True,
                    decimate=True,
                    decimate_face_num=100,
                ),
                material=gs.materials.Rigid(rho=100.0),
                vis_mode="visual",
            ),
            "plastic_bottle": scene.add_entity(
                gs.morphs.Mesh(
                    file=PLASTIC_BOTTLE_GLB,
                    pos=(0.50, -0.37, 0.81),
                    scale=0.036,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=True,
                    decimate=True,
                    decimate_face_num=100,
                ),
                material=gs.materials.Rigid(rho=50.0),
                vis_mode="visual",
            ),
        }

        # FEM objects (matches current registry: pos, euler, scale, material params)
        self._fem_entities = {
            "trashbag": scene.add_entity(
                gs.morphs.Mesh(
                    file=TRASHBAG_GLB,
                    pos=(0.60, 0.23, 0.75),
                    euler=(0, 0, 45),
                    scale=0.76,
                    fixed=False,
                    file_meshes_are_zup=False,
                ),
                material=gs.materials.FEM.Paper(
                    E=2e4,
                    nu=0.49,
                    rho=50.0,
                    thickness=1e-4,
                    bending_stiffness=2.0,
                    plasticity_model="stress",
                    yield_stress=800,
                ),
                surface=gs.surfaces.Plastic(),
                vis_mode="visual",
            ),
        }

        # Ropes (euler = trashbag_euler + (90, 0, 0) = (90, 0, 45))
        self._fem_entities["rope1"] = scene.add_entity(
            gs.morphs.Mesh(
                file=ROPE1_OBJ,
                pos=(0.60, 0.23, 0.75),
                euler=(90, 0, 45),
                scale=0.76,
                fixed=False,
            ),
            material=gs.materials.FEM.Rope(
                E=5e5,
                rho=50.0,
                thickness=0.0022,
                bending_stiffness=1e3,
            ),
            surface=gs.surfaces.Plastic(color=(0.8, 0.15, 0.1, 1.0)),
            vis_mode="visual",
        )
        self._fem_entities["rope2"] = scene.add_entity(
            gs.morphs.Mesh(
                file=ROPE2_OBJ,
                pos=(0.60, 0.23, 0.75),
                euler=(90, 0, 45),
                scale=0.76,
                fixed=False,
            ),
            material=gs.materials.FEM.Rope(
                E=5e5,
                rho=50.0,
                thickness=0.0022,
                bending_stiffness=1e3,
            ),
            surface=gs.surfaces.Plastic(color=(0.8, 0.15, 0.1, 1.0)),
            vis_mode="visual",
        )

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
    Trashbag2Replay().run()
