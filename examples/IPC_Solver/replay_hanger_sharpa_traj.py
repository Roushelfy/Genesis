"""
Replay a coat-hanger teleop trajectory — Sharpa hands, physics matching registry.

Uses the sharpa-hand trajectory (trajectory_sharpa.npz) with updated FEM params
from the ipc_hanger registry, combined with the rendering tweaks from replay_hanger_traj.py.

Usage
-----
    python examples/IPC_Solver/replay_hanger_sharpa_traj.py [OPTIONS]

Interactive viewer (default)
    --loop                  Loop replay continuously
    --speed FLOAT           Playback speed multiplier (default: 1.0)
                            Keyboard: , = half speed  . = double speed
    --start-frame N         Start from frame N (BACKSPACE also resets here)
    --end-frame N           Stop at frame N exclusive (default: last frame)
    --camera-traj MODE      Drive camera automatically: surround | full | ego | custom
                            K key = log current frame + camera pose to stdout

Live preview (interactive mode only)
    --preview               Open an OpenCV window with live Luisa-rendered view
    --preview --nyx         Add a second window with Nyx render alongside
                            Note: --nyx skips the FEM shirt (IPC FEM incompatible with Nyx)
    --preview-spp N         Samples-per-pixel for the preview camera (default: 64)
    --no-raytracer          Suppress Luisa renderer:
                              with --nyx  -> Nyx-only preview window
                              without     -> no preview window at all

Render to video
    --render                Record a video (Luisa by default)
    --render --nyx          Record with Nyx renderer (shirt not rendered)
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
    --traj PATH             Path to trajectory .npz file (default: trajectory_sharpa.npz)
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
_HANGER = _DEMO / "coat_hanger"

MARVIN_URDF = str(_DEMO / "marvin_sharpa_description/marvin_sharpa.urdf")
TABLE_GLB = str(_HANGER / "work_table.glb")
SHIRT_GLB = str(_HANGER / "genesis_shirt.glb")
COAT_HANGER_GLB = str(_HANGER / "coat_hanger.glb")
RACK_GLB = str(_HANGER / "rack_frame.glb")
DEFAULT_TRAJ = str(_HANGER / "trajectory_sharpa.npz")


class HangerSharpaReplay(TrajectoryReplay):
    name = "hanger_sharpa"
    cam_pos = (0.53, -1.2, 1.3)
    cam_lookat = (0.53, 0.0, 1.0)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: MARVIN_SHARPA, 58 DOF
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, drying_rack)
        self._rigid_data = {
            "coat_hanger": traj["rigid_coat_hanger"],
        }

        # FEM data
        self._fem_data = {
            "shirt": traj["fem_shirt"],
        }

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        print(f"Coat hanger frames: {self._rigid_data['coat_hanger'].shape[0]}")
        print(f"Shirt: {self._fem_data['shirt'].shape[1]} verts")
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
                SphereLight(pos=(1.85, 0.15, 2.5), radius=0.5, color=(0.88, 0.92, 1.0),  intensity=20.0),
                SphereLight(pos=(0.0, -2.0, 3),    radius=0.5, color=(1.0, 1.0, 1.0),    intensity=15.0),
                SphereLight(pos=(0.0, 2.0, 3),     radius=0.5, color=(1.0, 1.0, 1.0),    intensity=15.0),
            ],
        )

    def nyx_lights(self):
        return [
            {"type": "point", "pos": (-1.15, 0.0, 2.7),  "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 10.0},
            {"type": "point", "pos": (1.85,  0.15, 2.5), "radius": 0.5, "color": (0.88, 0.92, 1.0),  "intensity": 20.0},
        ]

    def nyx_light_field(self):
        # San Carlos robot station splat — position/rotation tuned for this scene.
        # Run with --nyx --preview and tweak until the background aligns.
        return {
            "uri": str((_REPO / "DemoAssets/3dgs/0325_san_carlos_robot_station.ply").resolve()),
            "position": (1.5, 0.81, -3.0),
            "rotation": (0.0, 0.701707, 0.0, 0.701707),  # (w, x, y, z) — 180° around Y
            "scale": (1.0, 1.0, 1.0),
        }

    def custom_camera_keyframes(self):
        # Keyframes captured on the wuji trajectory — re-tune for sharpa with the K key.
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        return [
            (0, (0.0686, 0.8314, 1.7396), (0.3326, 0.1209, 1.0873), (0, 0, 1), None),
            (70, (0.0686, 0.8314, 1.7396), (0.3326, 0.1209, 1.0873)),
            (918, (0.9427, 0.6215, 1.8174), (0.5, 0.0807, 1.1022), (0.0, 0.2425, 0.9701)),
            (2559, (1.2199, -0.5131, 1.7283), (0.5531, -0.1367, 1.0851), (0.0, 0.2425, 0.9701)),
            (3721, (0.8442, -0.2222, 1.4817), (0.2294, 0.017, 0.7301), (0.0, 0.2425, 0.9701)),
            (4299, (-0.115, -0.4611, 1.8316), (0.3433, -0.0896, 1.0242), (0.0, 0.2425, 0.9701)),
        ]

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=(0.53, 0.0, 1.0),
                radius=1.5,
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

        # Table (ipc_hanger: rotated 90 deg, shifted)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.95, 0.10, 0.0),
                euler=(0, 0, 90),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            surface=gs.surfaces.BSDF(
                roughness=0.1,
                metallic=0.0,
                ior=1.5,
            ),
            vis_mode="visual",
        )

        # Drying rack (fixed)
        scene.add_entity(
            gs.morphs.Mesh(
                file=RACK_GLB,
                pos=(0.53, 0.0, 0.0),
                euler=(0, 0, 90),
                scale=0.83,
                fixed=True,
                file_meshes_are_zup=False,
                convexify=False,
                decimate=True,
                decimate_face_num=1000,
            ),
            material=gs.materials.Rigid(rho=100.0),
            vis_mode="visual",
        )

        # Rigid objects (matches registry COAT_HANGER for ipc_hanger)
        self._rigid_entities = {
            "coat_hanger": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.51, -0.25, 0.84),
                    euler=(180, 180, 0),
                    scale=0.004,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=gs.materials.Rigid(rho=80.0),
                vis_mode="visual",
            ),
        }

        # Shirt (FEM Cloth, matches registry SHIRT_HANGER)
        self._fem_entities = {
            "shirt": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHIRT_GLB,
                    pos=(0.55, 0.25, 1.09),
                    euler=(-90, 150, 90),
                    scale=0.80,
                    fixed=False,
                ),
                material=gs.materials.FEM.Cloth(
                    E=1e3,
                    nu=0.49,
                    rho=60.0,
                    thickness=0.0005,
                    bending_stiffness=0.1,
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
    HangerSharpaReplay().run()
