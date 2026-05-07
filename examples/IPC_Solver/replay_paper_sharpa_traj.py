"""
Replay a paper-folding teleop trajectory — Sharpa hands, high-quality rendering.

Uses the sharpa-hand trajectory (trajectory_sharpa.npz) with updated FEM params
from the ipc_demo registry, combined with the rendering tweaks from
replay_trashbag_sharpa_traj.py.

Usage
-----
    python examples/IPC_Solver/replay_paper_sharpa_traj.py [OPTIONS]

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
    --traj PATH             Path to trajectory .npz file (default: trajectory_sharpa.npz)
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np


class _LightDef(TypedDict):
    pos: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]
    intensity: float


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
DEFAULT_TRAJ = str(_PAPER / "trajectory_sharpa.npz")

# Auto-pick the paper GLB by FEM vertex count so older trajectories keep
# replaying. Add new variants here when their meshes change.
_PAPER_GLB_BY_VCOUNT = {
    736: str(_PAPER / "paper_plane_extra_coarse.glb"),
    795: str(_PAPER / "paper_plane_coarse.glb"),
    960: str(_PAPER / "genesis_paper.glb"),
    2339: str(_PAPER / "paper_plane_coarse_dense.glb"),
}


class PaperSharpaReplay(TrajectoryReplay):
    name = "paper_sharpa"
    cam_pos = (1.4097, -0.3759, 1.3223)
    cam_lookat = (0.5421, -0.0112, 0.9842)
    cam_fov = 40

    # ── Shared light rig ──────────────────────────────────────────────────────
    # Edit pos / color / intensity here; both renderers pick them up automatically.
    # radius and intensity are in Luisa units — Nyx scale factors are applied below.
    _LIGHTS: list[_LightDef] = [
        # Key light: above-left, warm, soft — highlights paper creases
        {"pos": (0.5, 1.1, 2.4), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 50.0},
        # Fill light: right side, cooler, large and soft
        {"pos": (0.5, -1.8, 4.2), "radius": 1.0, "color": (0.48, 0.52, 0.6), "intensity": 1.0},
        # Rim light: behind the scene, cool, hard — separates hands from dark background
        {"pos": (-0.8, -3.0, 0.5), "radius": 0.25, "color": (0.8, 0.88, 1.0), "intensity": 150.0},
    ]
    # Nyx uses different physical units for radius and intensity.
    # Tune these two scalars to match perceived brightness/softness without
    # touching individual light values.
    NYX_RADIUS_SCALE = 1.0
    NYX_INTENSITY_SCALE = 0.2

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(emissive_texture=gs.textures.ColorTexture(color=(0.01, 0.01, 0.01))),
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

        # No rigid objects in this recording
        self._rigid_data = {}

        # FEM data
        self._fem_data = {
            "paper_sheet": traj["fem_paper_sheet"],
        }

        v_count = self._fem_data["paper_sheet"].shape[1]
        if v_count not in _PAPER_GLB_BY_VCOUNT:
            raise ValueError(f"unknown paper FEM vertex count {v_count}; add a GLB mapping in _PAPER_GLB_BY_VCOUNT")
        self._paper_glb = _PAPER_GLB_BY_VCOUNT[v_count]

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        print(f"Paper: {v_count} verts → {Path(self._paper_glb).name}")
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
                pos=self.cam_pos,
                lookat=self.cam_lookat,
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

    def custom_camera_keyframes(self):
        # Use K key in interactive mode to capture keyframes, then paste them here.
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        return [
            (427, (1.4097, -0.3759, 1.3223), (0.5421, -0.0112, 0.9842)),
            (734, (1.0611, -0.6897, 1.3552), (0.4518, -0.0024, 0.9597)),
            (930, (1.0287, -0.7462, 1.3071), (0.4194, -0.0589, 0.9115)),
            (1112, (0.9679, -0.5581, 1.2046), (0.286, 0.0729, 0.8348)),
            (1367, (1.1963, 0.0209, 1.1793), (0.2985, 0.0318, 0.7389)),
            (1442, (1.1695, -0.1887, 1.1844), (0.3508, 0.1765, 0.7413)),
        ]

    def build_scene(self, scene):
        import genesis as gs

        # Ground — large flat box so surface color override works (Plane ignores it)
        scene.add_entity(
            gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -2.01), fixed=True),
            surface=gs.surfaces.BSDF(diffuse_texture=gs.textures.ColorTexture(color=(0.05, 0.05, 0.05))),
        )

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
            surface=gs.surfaces.BSDF(
                roughness=0.45,
                metallic=0.0,
                ior=1.45,
            ),
            vis_mode="visual",
        )

        # No rigid entities in this recording
        self._rigid_entities = {}

        # Paper sheet (FEM Paper, matches registry PAPER_SHEET)
        self._fem_entities = {
            "paper_sheet": scene.add_entity(
                gs.morphs.Mesh(
                    file=self._paper_glb,
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
                surface=gs.surfaces.BSDF(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.93, 0.91, 0.88)),
                    roughness=0.85,
                    metallic=0.0,
                    ior=1.5,
                ),
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
    PaperSharpaReplay().run()
