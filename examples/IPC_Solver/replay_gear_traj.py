"""
Replay a planetary gear teleop trajectory — Sharpa hands.

Uses trajectory_gear_sharpa.npz with the gear env setup from gs-core's
IPCGearEnvConfig / IPCGearEnv.

Usage
-----
    python examples/IPC_Solver/replay_gear_traj.py [OPTIONS]

Interactive viewer (default)
    --loop                  Loop replay continuously
    --speed FLOAT           Playback speed multiplier (default: 1.0)
                            Keyboard: , = half speed  . = double speed
    --start-frame N         Start from frame N (BACKSPACE also resets here)
    --end-frame N           Stop at frame N exclusive (default: last frame)
    --camera-traj MODE      Drive camera automatically: surround | full | ego | custom
                            K key   = log current frame + camera pose to stdout
                            [ / ]   = jump to prev/next custom keyframe (pauses)

Live preview (interactive mode only)
    --preview               Open an OpenCV window with live Luisa-rendered view
    --preview --nyx         Add a second window with Nyx render alongside
    --preview-spp N         Samples-per-pixel for the preview camera (default: 64)
    --no-raytracer          Suppress Luisa renderer:
                              with --nyx  -> Nyx-only preview window
                              without     -> no preview window at all

Render to video
    --render                Record a video (Luisa by default)
    --render --nyx          Record with Nyx renderer
    --render --camera-traj MODE  Use a camera trajectory for the render
    --spp N                 Samples-per-pixel for the render camera (default: 256)
    --save-frames           Also save each frame as a PNG alongside the video

Shared camera / render options
    --res W H               Resolution for viewer, preview, and render cameras
                            (default: 1920 1080)
    --nyx                   Use Nyx renderer (preview or render)
    --dof                   Enable depth-of-field (thinlens model)
    --aperture F            Aperture f-number for DOF (default: 1.4, lower = shallower)
    --focus-dist METRES     Focus distance (default: auto from cam_pos -> lookat)
    --focal-len METRES      Focal length (default: 0.05 = 50 mm)

Trajectory
    --traj PATH             Path to trajectory .npz file (default: trajectory_gear_sharpa.npz)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import numpy as np

from _replay_common import (
    EgoCamera,
    FullViewCamera,
    SurroundCamera,
    TrajectoryReplay,
)

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "DemoAssets"
_GEAR = _DEMO / "planetary_gear"

MARVIN_URDF = str(_DEMO / "marvin_sharpa_description" / "marvin_sharpa.urdf")
TABLE_GLB = str(_DEMO / "coat_hanger" / "work_table.glb")
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


class _LightDef(TypedDict):
    pos: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]
    intensity: float


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
    cam_lookat = (CX, CY, CZ)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, ring_gear)
        self._rigid_data = {
            "sun_gear":      traj["rigid_sun_gear"],
            "carrier":       traj["rigid_carrier"],
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

    # ── Shared light rig ──────────────────────────────────────────────────────
    # radius and intensity are Luisa units — Nyx scale factors applied below.
    _LIGHTS: list[_LightDef] = [
        # Key light: above-left, warm — rakes across the gear teeth for shadow detail
        {"pos": (0.1,   1.2,  2.0),  "radius": 0.15, "color": (1.0, 0.97, 0.92), "intensity": 160.0},
        # Fill light: right side, cooler, large — lifts shadow side without flattening
        {"pos": (1.0,  -1.8,  3.5),  "radius": 1.0,  "color": (0.48, 0.52, 0.6),  "intensity": 1.5},
        # Rim light: behind scene, cool, hard — catches gear edges against dark background
        {"pos": (-0.5, -2.5,  0.8),  "radius": 0.2,  "color": (0.8, 0.88, 1.0),   "intensity": 120.0},
    ]
    NYX_RADIUS_SCALE    = 1.0
    NYX_INTENSITY_SCALE = 0.2

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.1, 0.1, 0.1))),
            env_radius=100.0,
            lights=[
                SphereLight(
                    pos=l["pos"],
                    radius=l["radius"],
                    color=l["color"],
                    intensity=l["intensity"],
                )
                for l in self._LIGHTS
            ],
        )

    def nyx_lights(self):
        return [
            {
                "type": "point",
                "pos": l["pos"],
                "radius": float(l["radius"]) * self.NYX_RADIUS_SCALE,
                "color": l["color"],
                "intensity": float(l["intensity"]) * self.NYX_INTENSITY_SCALE,
            }
            for l in self._LIGHTS
        ]

    def nyx_light_field(self):
        return {
            "uri": str((_REPO / "DemoAssets/3dgs/0325_san_carlos_robot_station.ply").resolve()),
            "position": (1.5, 0.81, -3.0),
            "rotation": (0.0, 0.701707, 0.0, 0.701707),  # (w, x, y, z) — 180° around Y
            "scale": (1.0, 1.0, 1.0),
        }

    def custom_camera_keyframes(self):
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        # Use K key in interactive mode to log new keyframes.
        return [
            (1, (1.4129, -0.0083, 1.6242), (0.6747, -0.0016, 0.9496)),
            (2, (0.7976, 0.0379, 1.1304), (0.1417, -0.0456, 0.3802)),
            (3, (0.7346, 0.06, 1.0665), (0.0963, -0.1032, 0.3142)),
            (4, (0.7028, 0.0452, 1.0474), (0.0897, -0.0915, 0.2693)),
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

        gear_assets = str(_GEAR / "assets")
        rot_off = GEAR_ROTATION_DEG

        # Ground — dark backdrop so gear mechanism reads against black
        scene.add_entity(
            gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -2.01), fixed=True),
            surface=gs.surfaces.BSDF(
                diffuse_texture=gs.textures.ColorTexture(color=(0.03, 0.03, 0.03))),
        )

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
            surface=gs.surfaces.BSDF(roughness=0.45, metallic=0.0, ior=1.45),
            vis_mode="visual",
        )

        rigid_mat = gs.materials.Rigid(rho=1000.0)

        # Ring gear — fixed housing
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/ring_gear.glb",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            # surface=gs.surfaces.Metal(color=(0.25, 0.25, 0.27, 1.0), roughness=0.35),
            vis_mode="visual",
        )

        # Sun gear with handle — bright polished steel, low roughness for sharp specular
        self._rigid_entities["sun_gear"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/sun_gear_handle.glb",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                file_meshes_are_zup=True,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
#            surface=gs.surfaces.Metal(color=(0.80, 0.80, 0.78, 1.0), roughness=0.08),
            vis_mode="visual",
        )

        # Planet gears — mid roughness, warm brass/bronze tint to distinguish from sun+ring
        for i in range(NUM_PLANETS):
            tx, ty, self_rot_deg = _planet_position(i)
            self._rigid_entities[f"planet_gear_{i}"] = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"{gear_assets}/planet_gear.glb",
                    pos=(CX + tx, CY + ty, CZ),
                    euler=(0, 0, self_rot_deg),
                    scale=MESH_SCALE,
                    fixed=False,
                    file_meshes_are_zup=True,
                    convexify=False,
                    decimate=False,
                ),
                material=rigid_mat,
#                surface=gs.surfaces.Metal(color=(0.72, 0.60, 0.35, 1.0), roughness=0.25),
                vis_mode="visual",
            )

        # Carrier — matte aluminium, clearly different from the steel gears
        carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
        self._rigid_entities["carrier"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/carrier.glb",
                pos=(CX, CY, CZ + carrier_tz),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                file_meshes_are_zup=True,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
#            surface=gs.surfaces.Metal(color=(0.55, 0.57, 0.60, 1.0), roughness=0.45),
            vis_mode="visual",
        )

        # Robot (MARVIN_SHARPA, 58 DOF, fixed base)
        # Override paint_white_glossy (arm links 1-6) to add some shininess; GLB default is too matte.
        # aluminium_brushed and plastic_black_rough are left to the GLB PBR values.
        self._robot = scene.add_entity(
            gs.morphs.URDF(
                file=MARVIN_URDF,
                fixed=True,
                collision=False,
                pos=(0, 0, 1.08),
            ),
            surface={
                "paint_white_glossy": gs.surfaces.BSDF(
                    color=(0.74, 0.74, 0.74),
                    roughness=0.25,
                    metallic=0.25,
                ),
            },
            vis_mode="visual",
        )


if __name__ == "__main__":
    GearReplay().run()
