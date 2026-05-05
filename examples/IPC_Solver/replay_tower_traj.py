"""
Replay a Tower of Hanoi ring stacking teleop trajectory — Marvin Pika.

Uses trajectory_tower.npz with the tower env setup from gs-core's
IPCTowerEnvConfig.

Usage
-----
    python examples/IPC_Solver/replay_tower_traj.py [OPTIONS]

    --traj PATH             Path to trajectory .npz file
    --loop                  Loop replay continuously
    --speed FLOAT           Playback speed multiplier
    --render                Record video
    --spp N                 Samples-per-pixel for render camera
    --camera-traj MODE      surround | full | ego | custom
    --preview               Live Luisa preview window
    --nyx                   Use Nyx renderer
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
_TOWER = _DEMO / "tower"
_TABLE_GLB = str(_DEMO / "coat_hanger" / "work_table.glb")

MARVIN_URDF = marvin_urdf("marvin_pika")
DEFAULT_TRAJ = str(_TOWER / "trajectory_20260422_155628.npz")

# Tower position (matching IPCTowerEnvConfig / registry)
TOWER_X, TOWER_Y = 0.54, 0.0
TOWER_TABLE_Z = 0.786

# Ring stacking constants
RING_H = 0.0204
BASE_TOP_Z = TOWER_TABLE_Z + 0.028
GAP = 0.002


def _tower_stack_z(layer: int) -> float:
    return BASE_TOP_Z + GAP + layer * (RING_H + GAP)


BALL_Z = _tower_stack_z(5) + RING_H / 2 + 0.0225 + GAP

# Ring names in stacking order (bottom to top), then ball
RING_NAMES = ["ring_01", "ring_02", "ring_03", "ring_04", "ring_06", "ring_05"]
RING_LAYERS = {name: i for i, name in enumerate(RING_NAMES)}


class TowerReplay(TrajectoryReplay):
    name = "tower"
    cam_pos = (0.54, -0.8, 1.2)
    cam_lookat = (0.54, 0.0, 0.85)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: MARVIN_PIKA, 18 DOF
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, base_pole)
        self._rigid_data = {}
        for name in RING_NAMES:
            key = f"rigid_{name}"
            if key in traj:
                self._rigid_data[name] = traj[key]
        if "rigid_ball" in traj:
            self._rigid_data["ball"] = traj["rigid_ball"]

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        for name, data in self._rigid_data.items():
            print(f"  {name}: {data.shape[0]} frames")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_radius=100.0,
            env_euler=(0, 0, 20),
            lights=[
                SphereLight(pos=(-1.15, 0.0, 2.7), radius=0.2, color=(1.0, 0.97, 0.92), intensity=10.0),
                SphereLight(pos=(1.85, 0.15, 2.5), radius=0.5, color=(0.88, 0.92, 1.0), intensity=20.0),
                SphereLight(pos=(0.0, -2.0, 3), radius=0.5, color=(1.0, 1.0, 1.0), intensity=15.0),
                SphereLight(pos=(0.0, 2.0, 3), radius=0.5, color=(1.0, 1.0, 1.0), intensity=15.0),
            ],
        )

    def nyx_lights(self):
        return [
            {"type": "point", "pos": (-1.15, 0.0, 2.7), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 10.0},
            {"type": "point", "pos": (1.85, 0.15, 2.5), "radius": 0.5, "color": (0.88, 0.92, 1.0), "intensity": 20.0},
        ]

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=(TOWER_X, TOWER_Y, 0.9),
                radius=0.8,
                height=1.2,
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

    def build_scene(self, scene):
        import genesis as gs

        # Table
        scene.add_entity(
            gs.morphs.Mesh(
                file=_TABLE_GLB,
                pos=(0.8, 0.0, 0.0),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            surface=gs.surfaces.BSDF(roughness=0.1, metallic=0.0, ior=1.5),
            vis_mode="visual",
        )

        rigid_mat = gs.materials.Rigid(rho=50.0)
        base_mat = gs.materials.Rigid(rho=2000.0)

        # Base pole (fixed)
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(_TOWER / "base_pole.glb"),
                pos=(TOWER_X, TOWER_Y, TOWER_TABLE_Z),
                fixed=True,
                convexify=False,
                decimate=False,
                file_meshes_are_zup=True,
            ),
            material=base_mat,
            vis_mode="visual",
        )

        # Rings
        for name in RING_NAMES:
            layer = RING_LAYERS[name]
            z = _tower_stack_z(layer)
            self._rigid_entities[name] = scene.add_entity(
                gs.morphs.Mesh(
                    file=str(_TOWER / f"{name}.glb"),
                    pos=(TOWER_X, TOWER_Y, z),
                    fixed=False,
                    convexify=False,
                    decimate=False,
                    file_meshes_are_zup=True,
                ),
                material=rigid_mat,
                vis_mode="visual",
            )

        # Ball
        self._rigid_entities["ball"] = scene.add_entity(
            gs.morphs.Mesh(
                file=str(_TOWER / "ball.glb"),
                pos=(TOWER_X, TOWER_Y, BALL_Z),
                fixed=False,
                convexify=False,
                decimate=False,
                file_meshes_are_zup=True,
            ),
            material=rigid_mat,
            vis_mode="visual",
        )

        # Robot (MARVIN_PIKA, 18 DOF, fixed base)
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
    TowerReplay().run()
