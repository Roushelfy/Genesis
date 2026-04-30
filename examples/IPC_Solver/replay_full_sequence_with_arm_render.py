"""
Replay the stitched full gear sequence with shoulder arm — high-quality render.

The shoulder arm (marvin_pika) is mounted on the carrier flange.
Joint2_R follows the carrier's Z-rotation each frame.

Uses gs_full_sequence.npz produced by
  python DemoAssets/planetary_with_teleop/convert_to_genesis.py

Usage
-----
    python examples/IPC_Solver/replay_full_sequence_with_arm_render.py [OPTIONS]

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
    --traj PATH             Path to trajectory .npz file (default: gs_full_sequence.npz)
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
_FULL_SEQ = _DEMO / "planetary_with_teleop"

MARVIN_URDF = str(_DEMO / "marvin_sharpa_description" / "marvin_sharpa.urdf")
PIKA_URDF   = str(_DEMO / "marvin_robot" / "urdf" / "marvin_pika.urdf")
TABLE_GLB   = str(_DEMO / "trashbag" / "work_table.glb")
DEFAULT_TRAJ = str(_FULL_SEQ / "gs_full_sequence.npz")

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
    orbit_angle = math.radians(index * 360.0 / NUM_PLANETS + GEAR_ROTATION_DEG)
    self_rot_deg = index * 360.0 * SUN_TEETH / PLANET_TEETH + GEAR_ROTATION_DEG
    tx = ORBIT_R * math.cos(orbit_angle)
    ty = ORBIT_R * math.sin(orbit_angle)
    return tx, ty, self_rot_deg


def _quat_wxyz_to_mat3(qw, qx, qy, qz):
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _carrier_z_angle(pose7):
    """Extract Z-rotation angle from carrier pose quaternion."""
    R = _quat_wxyz_to_mat3(*pose7[3:].astype(np.float64))
    return math.atan2(R[1, 0], R[0, 0])


class FullSequenceWithArmRender(TrajectoryReplay):
    name = "gear_full_arm"
    cam_pos = (0.5, -0.8, 1.3)
    cam_lookat = (CX, CY, CZ)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory .npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        self._joint_qpos = traj["robot_qpos"].astype(np.float32)
        self._carrier_poses = traj["rigid_carrier"]

        # Rigid data (skip static: ground, table, ring_gear)
        self._rigid_data = {
            "sun_gear": traj["rigid_sun_gear"],
            "carrier": traj["rigid_carrier"],
            "planet_gear_0": traj["rigid_planet_gear_0"],
            "planet_gear_1": traj["rigid_planet_gear_1"],
            "planet_gear_2": traj["rigid_planet_gear_2"],
        }

        # Compute carrier initial Z angle for relative rotation
        self._carrier_angle_0 = _carrier_z_angle(self._carrier_poses[0])

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
        {"pos": (0.1, 1.2, 2.0), "radius": 0.15, "color": (1.0, 0.97, 0.92), "intensity": 160.0},
        # Fill light: right side, cooler, large — lifts shadow side without flattening
        {"pos": (1.0, -1.8, 3.5), "radius": 1.0, "color": (0.48, 0.52, 0.6), "intensity": 1.5},
        # Rim light: behind scene, cool, hard — catches gear edges against dark background
        {"pos": (-0.5, -2.5, 0.8), "radius": 0.2, "color": (0.8, 0.88, 1.0), "intensity": 120.0},
    ]
    NYX_RADIUS_SCALE = 1.0
    NYX_INTENSITY_SCALE = 0.2

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(emissive_texture=gs.textures.ColorTexture(color=(0.1, 0.1, 0.1))),
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
            (1, (0.7976, 0.0379, 1.1304), (0.1417, -0.0456, 0.3802), (0, 0, 1), None),
            (480, (0.7976, 0.0379, 1.1304), (0.1417, -0.0456, 0.3802), (0, 0, 1), 4, 4),
#            (800, (2.2845, 0.2272, 2.8311), (1.6286, 0.1437, 2.0809)),
            (900, (1.6295, 0.028, 2.172), (0.9967, 0.0123, 1.3978)),
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

        #Ground — dark backdrop so gear mechanism reads against black
        scene.add_entity(
            gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -2.01), fixed=True),
            surface=gs.surfaces.BSDF(diffuse_texture=gs.textures.ColorTexture(color=(0.03, 0.03, 0.03))),
        )

        # Table
        # scene.add_entity(
        #     gs.morphs.Mesh(
        #         file=TABLE_GLB,
        #         pos=(0.8, 0.0, 0.0),
        #         scale=(1.14, 1.0, 1.47),
        #         fixed=True,
        #         file_meshes_are_zup=True,
        #         convexify=False,
        #     ),
        #     surface=gs.surfaces.BSDF(roughness=0.45, metallic=0.0, ior=1.45),
        #     vis_mode=self.args.vis_mode,
        # )

        rigid_mat = gs.materials.Rigid(rho=1000.0)

        # Ring gear — fixed housing
        scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/ring_gear_v1.glb",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            vis_mode=self.args.vis_mode,
        )

        # Sun gear with handle — polished steel surface override
        self._rigid_entities["sun_gear"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/sun_gear_handle_v2.glb",
                pos=(CX, CY, CZ),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
            surface=gs.surfaces.BSDF(color=(67.0/255.0, 79.0/255.0, 99.0/255.0), metallic=0.9, roughness=0.35),
            vis_mode=self.args.vis_mode,
        )

        # Planet gears — mid roughness, warm brass/bronze tint to distinguish from sun+ring
        for i in range(NUM_PLANETS):
            tx, ty, self_rot_deg = _planet_position(i)
            self._rigid_entities[f"planet_gear_{i}"] = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"{gear_assets}/planet_gear_v2.glb",
                    pos=(CX + tx, CY + ty, CZ),
                    euler=(0, 0, self_rot_deg),
                    scale=MESH_SCALE,
                    fixed=False,
                    convexify=False,
                    decimate=False,
                ),
                material=rigid_mat,
                vis_mode=self.args.vis_mode,
            )

        # Carrier — matte aluminium, clearly different from the steel gears
        carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
        self._rigid_entities["carrier"] = scene.add_entity(
            gs.morphs.Mesh(
                file=f"{gear_assets}/carrier_v1.glb",
                pos=(CX, CY, CZ + carrier_tz),
                euler=(0, 0, rot_off),
                scale=MESH_SCALE,
                fixed=False,
                file_meshes_are_zup=False,
                convexify=False,
                decimate=False,
            ),
            material=rigid_mat,
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
            surface=gs.surfaces.Metal(color=(0.6, 0.6, 0.6, 1.0)),
            vis_mode="visual",
        )

        # MARVIN_SHARPA robot (teleop arms)
        # Override paint_white_glossy (arm links 1-6) to add some shininess; GLB default is too matte.
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
            vis_mode=self.args.vis_mode,
        )

        # Shoulder arm (marvin_pika) — mounted on carrier flange.
        # Placed at origin; post_build computes the correct root transform
        # so Joint2_R aligns with the carrier rotation axis.
        self._shoulder_robot = scene.add_entity(
            gs.morphs.URDF(
                file=PIKA_URDF,
                fixed=True,
                collision=False,
                pos=(0, 0, 0),
                scale=1.5,
            ),
            surface=gs.surfaces.BSDF(
                color=(0.2, 0.3, 0.45),
                roughness=0.9,
                metallic=0.0,
                opacity=0.2,
            ),
            vis_mode=self.args.vis_mode,
        )

    def post_build(self):
        super().post_build()

        # Compute root_transform so Joint2_R sits at carrier flange centre.
        # Same logic as sim_planetary.py::load_robot_reference.
        shoulder = self._shoulder_robot

        # Get Joint2_R local transform with all joints at 0
        shoulder.set_qpos(np.zeros(shoulder.n_qs))

        # Find Link2_R world transform (with root at identity)
        T_j2_local = None
        for link in shoulder.links:
            if link.name == "Link2_R":
                T_j2_local = np.array(link.get_pos_quat_matrix()).reshape(4, 4) if hasattr(link, 'get_pos_quat_matrix') else None
                break

        if T_j2_local is None:
            # Fallback: just get link pos/quat
            for link in shoulder.links:
                if link.name == "Link2_R":
                    pos = link.get_pos().cpu().numpy()
                    quat = link.get_quat().cpu().numpy()  # wxyz
                    T_j2_local = np.eye(4, dtype=np.float64)
                    T_j2_local[:3, :3] = _quat_wxyz_to_mat3(*quat)
                    T_j2_local[:3, 3] = pos
                    break

        if T_j2_local is not None:
            # T_gear: carrier center in world = (CX, CY, CZ + carrier_tz)
            carrier_tz = -(GEAR_WIDTH_MM / 2) * MESH_SCALE
            T_gear = np.eye(4, dtype=np.float64)
            T_gear[0, 3] = CX
            T_gear[1, 3] = CY
            T_gear[2, 3] = CZ + carrier_tz

            root_tf = T_gear @ np.linalg.inv(T_j2_local)
            shoulder.set_pos(root_tf[:3, 3])
            from scipy.spatial.transform import Rotation as Rot
            r = Rot.from_matrix(root_tf[:3, :3])
            qx, qy, qz, qw = r.as_quat()
            shoulder.set_quat(np.array([qw, qx, qy, qz]))
            print(f"[shoulder] root_pos = {np.round(root_tf[:3, 3], 4)}")

        # Find Joint2_R local qpos index
        self._j2_local_idx = None
        first_qs = self._shoulder_robot.joints[0].qs_idx
        first_start = first_qs[0] if isinstance(first_qs, (list, tuple)) else int(first_qs)
        for joint in shoulder.joints:
            if joint.name == "Joint2_R":
                j_qs = joint.qs_idx
                j_start = j_qs[0] if isinstance(j_qs, (list, tuple)) else int(j_qs)
                self._j2_local_idx = j_start - first_start
                break
        print(f"[shoulder] Joint2_R local_idx = {self._j2_local_idx}")

    def apply_frame(self, scene, frame_idx):
        super().apply_frame(scene, frame_idx)

        # Sync shoulder Joint2_R with carrier Z-rotation
        if self._j2_local_idx is not None:
            carrier_angle = _carrier_z_angle(self._carrier_poses[frame_idx])
            relative_angle = carrier_angle - self._carrier_angle_0
            qpos = self._shoulder_robot.get_qpos()
            if hasattr(qpos, 'cpu'):
                qpos = qpos.cpu().numpy().copy()
            else:
                qpos = np.array(qpos, dtype=np.float64).copy()
            if 0 <= self._j2_local_idx < len(qpos):
                qpos[self._j2_local_idx] = relative_angle
            self._shoulder_robot.set_qpos(qpos)


if __name__ == "__main__":
    FullSequenceWithArmRender().run()
