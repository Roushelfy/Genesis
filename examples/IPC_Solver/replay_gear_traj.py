"""
Replay a planetary gear teleop trajectory — Sharpa hands.

Uses trajectory_gear_sharpa.npz with the gear env setup from gs-core's
IPCGearEnvConfig / IPCGearEnv.

Usage
-----
    python examples/IPC_Solver/replay_gear_traj.py [OPTIONS]

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

import math
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
_GEAR = _DEMO / "planetary_gear"
_TABLE_GLB = str(_DEMO / "coat_hanger" / "work_table.glb")

MARVIN_URDF = marvin_urdf("marvin_sharpa")
DEFAULT_TRAJ = str(_GEAR / "trajectory_gear_sharpa.npz")

# Gear geometry (matching IPCGearEnvConfig defaults)
MESH_SCALE = 0.0012  # mm -> m
MODUL = 3
SUN_TEETH = 12
PLANET_TEETH = 9
NUM_PLANETS = 3
GEAR_WIDTH_MM = 12.0
GEAR_ROTATION_DEG = 270.0
ORBIT_R = MODUL * (SUN_TEETH + PLANET_TEETH) / 2 * MESH_SCALE

# Mechanism center
CX, CY, CZ = 0.5, 0.0, 0.79


def _planet_position(index: int) -> tuple[float, float, float]:
    """Compute planet gear orbit position and self-rotation."""
    orbit_angle = math.radians(index * 360.0 / NUM_PLANETS + GEAR_ROTATION_DEG)
    self_rot_deg = index * 360.0 * SUN_TEETH / PLANET_TEETH + GEAR_ROTATION_DEG
    tx = ORBIT_R * math.cos(orbit_angle)
    ty = ORBIT_R * math.sin(orbit_angle)
    return tx, ty, self_rot_deg


class GearReplay(TrajectoryReplay):
    name = "gear"
    cam_pos = (0.5, -0.8, 1.3)
    cam_lookat = (0.5, 0.0, 0.85)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: MARVIN_SHARPA, 58 DOF
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, ring_gear)
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

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.01, 0.01, 0.01))),
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
                center=(CX, CY, CZ),
                radius=0.8,
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

    def build_scene(self, scene):
        import genesis as gs

        rot_off = GEAR_ROTATION_DEG
        gear_assets = str(_GEAR / "assets")

        # Ground — large flat box (paper-plane background)
        scene.add_entity(
            gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -2.01), fixed=True),
            surface=gs.surfaces.BSDF(
                diffuse_texture=gs.textures.ColorTexture(color=(0.05, 0.05, 0.05))),
        )

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
            vis_mode=self.args.vis_mode,
        )

        housing_surface = gs.surfaces.Metal(color=(0.6, 0.6, 0.6, 1.0))
        sun_surface = gs.surfaces.Metal(color=(0.45, 0.45, 0.5, 1.0))
        planet_surface = gs.surfaces.Metal(color=(0.5, 0.55, 0.6, 1.0))
        carrier_surface = gs.surfaces.Metal(color=(0.55, 0.5, 0.45, 1.0))

        rigid_mat = gs.materials.Rigid(rho=1000.0)

        # Ring gear — fixed housing
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/ring_gear.obj",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=True,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            surface=housing_surface,
            vis_mode=self.args.vis_mode,
        )

        # Sun gear with handle — free
        self._rigid_entities["sun_gear"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/sun_gear_handle_v2.obj",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            surface=sun_surface,
            vis_mode=self.args.vis_mode,
        )

        # Planet gears
        for i in range(NUM_PLANETS):
            tx, ty, self_rot_deg = _planet_position(i)
            px = CX + tx
            py = CY + ty
            pz = CZ

            self._rigid_entities[f"planet_gear_{i}"] = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"{gear_assets}/planet_gear_v2.obj",
                    pos=(px, py, pz),
                    euler=(0, 0, self_rot_deg),
                    scale=MESH_SCALE,
                    fixed=False,
                    convexify=False,
                    decimate=False,
                ),
                material=rigid_mat,
                surface=planet_surface,
                vis_mode=self.args.vis_mode,
            )

        # Carrier — offset below gear plane
        carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
        self._rigid_entities["carrier"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/carrier.obj",
                pos=(CX, CY, CZ + carrier_tz),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            surface=carrier_surface,
            vis_mode=self.args.vis_mode,
        )

        # Support pin — fixed shaft
        support_pin_tz = -12.0 * MESH_SCALE
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/support_pin.obj",
                pos=(CX, CY, CZ + support_pin_tz),
                scale=MESH_SCALE,
                fixed=True,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            surface=housing_surface,
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
            vis_mode=self.args.vis_mode,
        )


if __name__ == "__main__":
    GearReplay().run()
