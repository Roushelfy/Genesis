"""
Replay a trashbag teleop trajectory — Sharpa hands, physics matching registry.

Uses the sharpa-hand trajectory (trajectory_sharpa.npz) with updated FEM/rope params
from the ipc_demo registry, combined with the rendering tweaks from replay_trashbag_traj.py.

Usage
-----
    python examples/IPC_Solver/replay_trashbag_sharpa_traj.py [OPTIONS]

Interactive viewer (default)
    --loop                  Loop replay continuously
    --speed FLOAT           Playback speed multiplier (default: 1.0)
                            Keyboard: , = half speed  . = double speed
    --start-frame N         Start from frame N (BACKSPACE also resets here)
    --end-frame N           Stop at frame N exclusive (default: last frame)
    --camera-traj MODE      Drive camera automatically: surround | full | ego | custom
                            K key   = log current frame + camera pose to stdout
                            [ / ]   = jump to prev/next custom keyframe (pauses)
                            /       = toggle ego camera on/off

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
TRASHCAN_GLB = str(_TRASH / "Trashcan033.glb")
SODA_CAN_GLB = str(_TRASH / "soda_can.glb")
PLASTIC_BOTTLE_GLB = str(_TRASH / "plastic_bottle.glb")
ROPE1_OBJ = str(_TRASH / "rope1.obj")
ROPE2_OBJ = str(_TRASH / "rope2.obj")
DEFAULT_TRAJ = str(_TRASH / "trajectory_sharpa.npz")


class TrashbagSharpaReplay(TrajectoryReplay):
    name = "trashbag_sharpa"
    cam_pos = (1.3205, -0.3625, 1.9686)
    cam_lookat = (0.6509, -0.1743, 1.2502)
    cam_fov = 40

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ColorTexture(color=(0.01, 0.01, 0.01))),
            env_radius=100.0,
            lights=[
                # Key light: above-left, warm, casting shadows across the table
                SphereLight(pos=(0.85, 1.25, 2.45), radius=0.2, color=(1.0, 0.97, 0.92), intensity=50.0),
                # Fill light: right side, cooler, softer
                SphereLight(pos=(0.6, -1.7, 4.3), radius=1.0, color=(0.48, 0.52, 0.6),  intensity=1.0),
                # Rim light: behind the scene, cool, hard — separates hands/bag from dark background
                SphereLight(pos=(-0.8, -3.16, 0.5), radius=0.25, color=(0.8, 0.88, 1.0), intensity=150.0),
            ],
        )

    def nyx_lights(self):
        return [
            {"type": "point", "pos": (-1.15, 0.0, 2.7),  "radius": 0.2, "color": (1.0, 0.97, 0.92), "intensity": 10.0},
            {"type": "point", "pos": (1.85,  0.15, 2.5), "radius": 0.5, "color": (0.88, 0.92, 1.0),  "intensity": 20.0},
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

        # Robot: MARVIN_SHARPA, 58 DOF (fixed base in replay)
        self._joint_qpos = traj["robot_qpos"].astype(np.float32)

        # Rigid data
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

    def custom_camera_keyframes(self):
        # Keyframes captured on the pika trajectory — will need re-tuning for sharpa.
        # Use K key in interactive mode to log new keyframes.
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        return [
            # (0, (1.6531, -0.6216, 2.0048), (0.9043, -0.2882, 1.4319), (0, 0, 1), None),
            # (64, (1.6531, -0.6216, 2.0048), (0.9043, -0.2882, 1.4319)),
            # (359, (1.0431, -0.2551, 1.81), (0.471, -0.1305, 0.9993)),
            # (585, (0.8324, -0.2202, 1.4868), (0.3867, -0.1278, 0.5964)),
            # (1007, (0.7981, -0.1766, 1.7551), (0.5121, -0.1717, 0.7969)),
            # (1492, (0.8892, 0.0118, 2.0495), (0.6032, 0.0166, 1.0913)),
            (8, (1.3205, -0.3625, 1.9686), (0.6509, -0.1743, 1.2502), (-0.583, 0.0086, 0.8124)),
            (10, (2.3313, -0.0087, 1.9842), (1.519, 0.0033, 1.4011), (-0.583, 0.0086, 0.8124)),
            (23, (1.0502, 0.3062, 1.274), (0.2742, 0.2846, 0.6436), (-0.583, 0.0086, 0.8124)),
            (41, (1.0275, 0.499, 1.1794), (0.354, 0.0108, 0.6243), (-0.583, 0.0086, 0.8124)),
            (87, (0.8742, 0.6937, 1.2944), (0.3189, 0.0744, 0.7393), (-0.583, 0.0086, 0.8124)),
            (170, (0.8384, 0.6602, 1.371), (0.2754, 0.0322, 0.8337), (-0.583, 0.0086, 0.8124)),
            (197, (0.8893, -0.4378, 1.3246), (0.3981, 0.3221, 0.899), (-0.583, 0.0086, 0.8124)),
            (690, (0.8246, -0.2951, 1.4988), (0.4017, 0.0223, 0.65), (-0.583, 0.0086, 0.8124)),
            (838, (0.8229, -0.2938, 1.4391), (0.4, 0.0236, 0.5903), (-0.583, 0.0086, 0.8124)),
            (982, (0.8124, -0.286, 1.418), (0.3895, 0.0315, 0.5692), (-0.583, 0.0086, 0.8124)),
            (1040, (0.6682, -0.215, 0.8907), (0.3521, 0.1057, 1.7836), (-0.583, 0.0086, 0.8124)),
            (1090, (0.6104, -0.1985, 0.8624), (0.4218, -0.0021, 1.8246), (-0.583, 0.0086, 0.8124)),
            (2118, (0.6169, -0.2011, 0.8638), (0.5252, -0.0436, 1.8471), (-0.583, 0.0086, 0.8124)),
            (2206, (-0.1851, 1.0792, 1.629), (0.353, 0.319, 1.2649), (-0.583, 0.0086, 0.8124)),
        ]
    
        # new camera shots
        # hero hand shot
        # cam_pos    = (0.3306, -0.2908, 1.3306)
        # cam_lookat = (0.5077, 0.5963, 0.9043)


    def build_scene(self, scene):
        import genesis as gs

        # Ground — large flat box so surface color override works (Plane ignores it)
        scene.add_entity(
            gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -2.01), fixed=True),
            surface=gs.surfaces.BSDF(
                diffuse_texture=gs.textures.ColorTexture(color=(0.05, 0.05, 0.05))),
        )

        # Table (pos matches registry)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TABLE_GLB,
                pos=(0.8, 0.0, 0.0),
                scale=(1.14, 1.0, 1.47),
                fixed=True,
                file_meshes_are_zup=True,
                convexify=False,
            ),
            surface=gs.surfaces.BSDF(
                #color=(0.8, 0.8, 0.95, 1.0),
                roughness=0.45,
                metallic=0.0,
                ior=1.45,
            ),
            vis_mode="visual",
        )

        # Trashcan (fixed, GLB)
        scene.add_entity(
            gs.morphs.Mesh(
                file=TRASHCAN_GLB,
                pos=(0.60, -0.15, 0.92),
                scale=0.8,
                fixed=True,
                file_meshes_are_zup=False,
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
                surface=gs.surfaces.BSDF(
#                    color=(0.8, 0.85, 0.95),
                    roughness=0.05,
                    ior=1.33,
                    specular_trans=0.83,
                ),
#             surface=gs.surfaces.Glass(
#                 color=(0.9, 0.95, 0.95),   # specular/reflection tint
#                 roughness=0.12,             # surface roughness (0 = perfect glass)
#                 ior=1.35,                   # PET plastic IOR
# #                subsurface=True,            # enables thickness-based BSSRDF
# #                thickness=0.001,            # shell thickness in metres (3 mm)
#             ),
                vis_mode="visual",
            ),
        }

        # FEM objects (params match registry)
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
    TrashbagSharpaReplay().run()
