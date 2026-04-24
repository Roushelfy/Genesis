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
    --subdiv                Use Loop-subdivided trashbag + rope meshes and matching trajectory
                            (Trashbag_rope_subdiv.glb, rope1/2_subdiv.obj, trajectory_sharpa_subdiv.npz)
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
TRASHBAG_GLB_SUBDIV = str(_TRASH / "Trashbag_rope_subdiv.glb")
TRASHCAN_GLB = str(_TRASH / "Trashcan033.glb")
SODA_CAN_GLB = str(_TRASH / "soda_can.glb")
PLASTIC_BOTTLE_GLB = str(_TRASH / "plastic_bottle.glb")
ROPE1_OBJ = str(_TRASH / "rope1.obj")
ROPE2_OBJ = str(_TRASH / "rope2.obj")
ROPE1_OBJ_SUBDIV = str(_TRASH / "rope1_subdiv.obj")
ROPE2_OBJ_SUBDIV = str(_TRASH / "rope2_subdiv.obj")
DEFAULT_TRAJ = str(_TRASH / "trajectory_sharpa.npz")
DEFAULT_TRAJ_SUBDIV = str(_TRASH / "trajectory_sharpa_subdiv.npz")


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
            default=None,
            help="Path to trajectory.npz (default: trajectory_sharpa.npz, "
                 "or trajectory_sharpa_subdiv.npz when --subdiv is set)",
        )
        parser.add_argument(
            "--subdiv",
            action="store_true",
            help="Use Loop-subdivided trashbag + rope meshes and matching trajectory "
                 "(Trashbag_rope_subdiv.glb, rope1/2_subdiv.obj, trajectory_sharpa_subdiv.npz)",
        )

    def load_trajectory(self):
        traj_path = self.args.traj or (
            DEFAULT_TRAJ_SUBDIV if getattr(self.args, "subdiv", False) else DEFAULT_TRAJ
        )
        traj = np.load(traj_path)
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
            (1, (1.3717, 0.0888, 1.501), (0.5382, 0.0954, 0.9486)), # grab bag
            (2, (1.1121, -0.1711, 1.3672), (0.2076, -0.1816, 0.9408)), # spread bag open
            (3, (0.8751, -0.684, 1.3839), (0.5325, 0.1772, 1.0084)), # step in & stretch bag
            (4, (0.8915, -0.3679, 1.3836), (0.3012, 0.1517, 0.766)), # pick up bottle
            (5, (0.8325, -0.3047, 1.3894), (0.3326, 0.0995, 0.6234)), # drop into can
            (6, (0.6946, -0.2428, 1.3516), (0.4141, -0.0268, 0.4164)), # drop soda
            (7, (0.7932, -0.4961, 1.3813), (0.4041, 0.1829, 0.7587)), # close bag
            (8, (1.1363, -0.0733, 1.2694), (0.1565, -0.0498, 1.071)), # lift it up
            (9, (1.179, 0.148, 1.0849), (0.1991, 0.1715, 0.8865)), # put it down
        ]


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
            surface = gs.surfaces.Plastic(
                color=(0.2353, 0.3529, 0.4078),
                roughness=0.55,                    # semi-matte, not fully diffuse
                metallic=0.0,                      # pure dielectric
                ior=1.52,                          # typical ABS/HDPE plastic
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
                    group_by_material=True,
                    convexify=True,
                    decimate=True,
                    decimate_face_num=100,
                ),
                material=gs.materials.Rigid(rho=50.0),
                surface={
                    "Material.Body": gs.surfaces.Glass(
                        color=(0.905, 0.905, 0.905, 1.0),
                        roughness=0.1,
                        ior=1.4,
                    ),
                    "Material.Cap": gs.surfaces.BSDF(
                        color=(0.0, 0.3451, 0.90588),
                        roughness=0.5,
                        ior=1.5,
                    ),
                },
                vis_mode="visual",
            ),
        }

        # FEM objects (params match registry)
        subdiv = getattr(self.args, "subdiv", False)
        trashbag_glb = TRASHBAG_GLB_SUBDIV if subdiv else TRASHBAG_GLB
        rope1_mesh   = ROPE1_OBJ_SUBDIV    if subdiv else ROPE1_OBJ
        rope2_mesh   = ROPE2_OBJ_SUBDIV    if subdiv else ROPE2_OBJ

        self._fem_entities = {
            "trashbag": scene.add_entity(
                gs.morphs.Mesh(
                    file=trashbag_glb,
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
                file=rope1_mesh,
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
                file=rope2_mesh,
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
    TrashbagSharpaReplay().run()
