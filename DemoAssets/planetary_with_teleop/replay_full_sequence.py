"""
Replay the stitched full gear sequence in Genesis.

Loads gs_full_sequence.npz (Genesis convention: COM pos, physics quat)
and replays with GLB meshes + MARVIN_SHARPA robot.

Usage
-----
    # First convert UIPC sequence to Genesis convention:
    python DemoAssets/planetary_with_teleop/convert_to_genesis.py

    # Then replay:
    python DemoAssets/planetary_with_teleop/replay_full_sequence.py [--traj PATH]
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Reuse the replay infrastructure from IPC_Solver
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
DEFAULT_TRAJ = str(_HERE / "gs_full_sequence.npz")

MESH_SCALE = 0.0012
MODUL = 3
SUN_TEETH = 12
PLANET_TEETH = 9
NUM_PLANETS = 3
GEAR_WIDTH_MM = 12.0
GEAR_ROTATION_DEG = 270.0
ORBIT_R = MODUL * (SUN_TEETH + PLANET_TEETH) / 2 * MESH_SCALE

CX, CY, CZ = 0.5, 0.0, 0.79


def _planet_position(index: int) -> tuple[float, float, float]:
    orbit_angle = math.radians(index * 360.0 / NUM_PLANETS + GEAR_ROTATION_DEG)
    self_rot_deg = index * 360.0 * SUN_TEETH / PLANET_TEETH + GEAR_ROTATION_DEG
    tx = ORBIT_R * math.cos(orbit_angle)
    ty = ORBIT_R * math.sin(orbit_angle)
    return tx, ty, self_rot_deg


class FullSequenceReplay(TrajectoryReplay):
    name = "gear_full"
    cam_pos = (0.5, -0.8, 1.3)
    cam_lookat = (CX, CY, CZ)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ,
                            help="Path to gs_full_sequence.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        self._rigid_data = {
            "sun_gear": traj["rigid_sun_gear"],
            "carrier": traj["rigid_carrier"],
            "planet_gear_0": traj["rigid_planet_gear_0"],
            "planet_gear_1": traj["rigid_planet_gear_1"],
            "planet_gear_2": traj["rigid_planet_gear_2"],
        }

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
                file=table_glb,
                pos=(0.8, 0.0, 0.0),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            surface=gs.surfaces.BSDF(roughness=0.45, metallic=0.0, ior=1.45),
            vis_mode="visual",
        )

        # Ring gear (fixed)
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/ring_gear.glb",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=True,
                convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # Sun gear (free)
        self._rigid_entities["sun_gear"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/sun_gear_handle_v2.glb",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # Planet gears
        for i in range(NUM_PLANETS):
            tx, ty, self_rot_deg = _planet_position(i)
            self._rigid_entities[f"planet_gear_{i}"] = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"{gear_assets}/planet_gear_v2.glb",
                    pos=(CX + tx, CY + ty, CZ),
                    euler=(0, 0, self_rot_deg),
                    scale=MESH_SCALE,
                    fixed=False,
                    convexify=False, decimate=False,
                ),
                material=rigid_mat, vis_mode="visual",
            )

        # Carrier
        carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
        self._rigid_entities["carrier"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/carrier.glb",
                pos=(CX, CY, CZ + carrier_tz),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

        # Support pin
        support_pin_tz = -12.0 * MESH_SCALE
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/support_pin.obj",
                pos=(CX, CY, CZ + support_pin_tz),
                scale=MESH_SCALE,
                fixed=True,
                convexify=False, decimate=False,
            ),
            material=rigid_mat, vis_mode="visual",
        )

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
    FullSequenceReplay().run()
