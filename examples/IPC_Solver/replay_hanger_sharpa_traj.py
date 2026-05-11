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
    --traj PATH             Path to trajectory .npz file (default: trajectory_sharpa.npz,
                            or trajectory_sharpa_subdiv.npz when --subdiv is set)
    --subdiv                Use Loop-subdivided shirt mesh + trajectory
                            (genesis_shirt_subdiv.glb + trajectory_sharpa_subdiv.npz)
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
_HANGER = _DEMO / "coat_hanger"

MARVIN_URDF = marvin_urdf("marvin_sharpa")
TABLE_GLB = str(_DEMO / "trashbag" / "work_table.glb")
SHIRT_GLB = str(_HANGER / "genesis_shirt.glb")
SHIRT_GLB_SUBDIV = str(_HANGER / "genesis_shirt_subdiv.glb")
COAT_HANGER_GLB = str(_HANGER / "coat_hanger.glb")
RACK_GLB = str(_HANGER / "rack_frame.glb")
DEFAULT_TRAJ_SUBDIV = str(_HANGER / "trajectory_sharpa_subdiv.npz")

_HF_REPO = "Genesis-Intelligence/internal_assets"
_HF_TRAJ = "ipc_trajectories/coat_hanger/trajectory_hanger_sharpa.npz"
_HF_TRAJ_REVISION = "52ed7d38d9f0dffc7794bd3941b6a60a5ddc0886"


def _resolve_default_traj() -> str:
    """Return local path to the default trajectory, downloading from HF if needed."""
    local = _HANGER / "trajectory_hanger_sharpa.npz"
    if local.exists():
        return str(local)
    print(f"Downloading trajectory from HuggingFace ({_HF_REPO}) ...")
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = hf_hub_download(
        repo_id=_HF_REPO,
        repo_type="dataset",
        filename=_HF_TRAJ,
        revision=_HF_TRAJ_REVISION,
    )
    return path


DEFAULT_TRAJ = _resolve_default_traj()


class HangerSharpaReplay(TrajectoryReplay):
    name = "hanger_sharpa"
    cam_pos = (1.5122, -0.767, 1.8931)
    cam_lookat = (0.838, -0.3497, 1.2837)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument(
            "--traj",
            type=str,
            default=None,
            help="Path to trajectory.npz (default: trajectory_sharpa.npz, "
            "or trajectory_sharpa_subdiv.npz when --subdiv is set)",
        )
        parser.add_argument(
            "--subdiv",
            action="store_true",
            help="Use Loop-subdivided shirt mesh + trajectory "
            "(genesis_shirt_subdiv.glb + trajectory_sharpa_subdiv.npz)",
        )

    def load_trajectory(self):
        traj_path = self.args.traj or (DEFAULT_TRAJ_SUBDIV if self.args.subdiv else DEFAULT_TRAJ)
        traj = np.load(traj_path)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: MARVIN_SHARPA, 58 DOF
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data (skip static: ground, table, drying_rack)
        self._rigid_data = {
            "coat_hanger": traj["rigid_coat_hanger"],
            "hung_hanger_1": traj["rigid_hung_hanger_1"],
            "hung_hanger_2": traj["rigid_hung_hanger_2"],
        }

        # FEM data
        self._fem_data = {
            "shirt": traj["fem_shirt"],
            "hung_shirt_1": traj["fem_hung_shirt_1"],
            "hung_shirt_2": traj["fem_hung_shirt_2"],
        }

        print(f"Robot qpos: {self._joint_qpos.shape[1]} DOF")
        for name, data in self._rigid_data.items():
            print(f"  {name}: {data.shape[0]} frames")
        for name, data in self._fem_data.items():
            print(f"  {name}: {data.shape[1]} verts, {data.shape[0]} frames")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    # ── Shared light rig ──────────────────────────────────────────────────────
    # Edit pos / color / intensity here; both renderers pick them up automatically.
    # radius and intensity are in Luisa units — Nyx scale factors are applied below.
    _LIGHTS: list[_LightDef] = [
        # Key light: above-left, warm, casting shadows across the scene
        {"pos": (0.5, 1.1, 2.4), "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 50.0},
        # Fill light: right side, cooler, large and soft
        {"pos": (0.5, -1.8, 4.2), "radius": 1.0, "color": (0.48, 0.52, 0.6), "intensity": 1.0},
        # Rim light: behind the scene, cool, hard — separates hands/shirt from dark background
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
            env_radius=100.0,
            env_euler=(0, 0, 20),
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
        # San Carlos robot station splat — position/rotation tuned for this scene.
        # Run with --nyx --preview and tweak until the background aligns.
        return {
            "uri": str((_REPO / "DemoAssets/3dgs/0325_san_carlos_robot_station.ply").resolve()),
            "position": (1.5, 0.81, -3.0),
            "rotation": (0.0, 0.701707, 0.0, 0.701707),  # (w, x, y, z) — 180° around Y
            "scale": (1.0, 1.0, 1.0),
        }

    def custom_camera_keyframes(self):
        # Keyframes captured on the wuji trajectory
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        return [
            (152, (1.5122, -0.767, 1.8931), (0.838, -0.3497, 1.2837)),
            (239, (1.2059, -0.0776, 1.3647), (0.4035, 0.0213, 0.7761)),
            (291, (1.3401, 0.0904, 1.0712), (0.3722, -0.0178, 0.8444)),
            (325, (0.6697, -0.5786, 1.3138), (0.5125, 0.2175, 0.7295)),
            (378, (0.1256, -0.852, 1.6709), (0.4728, -0.1608, 1.0371)),
            (459, (0.0621, 0.7607, 1.8796), (0.3539, 0.2161, 1.0933)),
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
                roughness=0.45,
                metallic=0.0,
                ior=1.45,
            ),
            vis_mode="visual",
        )

        # Drying rack (fixed, matches DRYING_RACK_X registry)
        scene.add_entity(
            gs.morphs.Mesh(
                file=RACK_GLB,
                pos=(0.62, -0.61, 0.0),
                euler=(0, 0, 0),
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

        rigid_mat = gs.materials.Rigid(rho=80.0)
        cloth_mat = gs.materials.FEM.Cloth(
            E=1e3,
            nu=0.49,
            rho=60.0,
            thickness=0.0005,
            bending_stiffness=0.1,
            friction_mu=0.3,
        )

        # Rigid objects (matches registry COAT_HANGER for ipc_hanger)
        self._rigid_entities = {
            "coat_hanger": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.51, -0.24, 0.84),
                    euler=(180, 180, 90),
                    scale=0.004,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=rigid_mat,
                vis_mode="visual",
            ),
            "hung_hanger_1": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.12, -0.607, 1.165),
                    euler=(90, 180, -90),
                    scale=0.004,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=rigid_mat,
                vis_mode="visual",
            ),
            "hung_hanger_2": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.44, -0.607, 1.165),
                    euler=(90, 180, -90),
                    scale=0.004,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=rigid_mat,
                vis_mode="visual",
            ),
        }

        # Shirt (FEM Cloth, matches registry)
        shirt_glb = SHIRT_GLB_SUBDIV if self.args.subdiv else SHIRT_GLB
        self._fem_entities = {
            "shirt": scene.add_entity(
                gs.morphs.Mesh(
                    file=shirt_glb,
                    pos=(0.59, 0.21, 1.1),
                    euler=(-90, 178, -90),
                    scale=0.80,
                    fixed=False,
                ),
                material=cloth_mat,
                surface=gs.surfaces.Plastic(),
                vis_mode="visual",
            ),
            "hung_shirt_1": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHIRT_GLB,
                    pos=(0.10, -0.60, 1.035),
                    euler=(-90, 180, 90),
                    scale=0.80,
                    fixed=False,
                ),
                material=cloth_mat,
                surface=gs.surfaces.Plastic(),
                vis_mode="visual",
            ),
            "hung_shirt_2": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHIRT_GLB,
                    pos=(0.42, -0.60, 1.035),
                    euler=(-90, 180, 90),
                    scale=0.80,
                    fixed=False,
                ),
                material=cloth_mat,
                surface=gs.surfaces.Plastic(),
                vis_mode="visual",
            ),
        }

        # Robot (MARVIN_SHARPA, 58 DOF, fixed base)
        # paint_white_glossy arm links (1-6): boost shininess; GLB default is too matte.
        # sharpa_aluminum hand links: left as GLB PBR values.
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
    HangerSharpaReplay().run()
