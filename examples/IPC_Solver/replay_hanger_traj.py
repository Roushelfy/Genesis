"""
Replay a coat-hanger teleop trajectory.

Usage
-----
    python examples/IPC_Solver/replay_hanger_traj.py [OPTIONS]

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
    --traj PATH             Path to trajectory .npz file (default: bundled trajectory)
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

MARVIN_URDF = str(_DEMO / "marvin_wuji/robot_capsule.urdf")
TABLE_GLB = str(_HANGER / "work_table.glb")
SHIRT_GLB = str(_HANGER / "genesis_shirt.glb")
COAT_HANGER_GLB = str(_HANGER / "coat_hanger.glb")
RACK_GLB = str(_HANGER / "rack_frame.glb")
SHOE_BOX_GLB = str(_HANGER / "shoe_box.glb")
SHOE_BOX_LID_GLB = str(_HANGER / "shoe_box_lid.glb")
DEFAULT_TRAJ = str(_HANGER / "trajectory.npz")


class HangerReplay(TrajectoryReplay):
    name = "hanger"
    cam_pos = (0.53, -1.2, 1.3)
    cam_lookat = (0.53, 0.0, 1.0)
    cam_fov = 40

    def add_args(self, parser):
        parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ, help="Path to trajectory.npz")

    def load_trajectory(self):
        traj = np.load(self.args.traj)
        self.sim_time = traj["sim_time"]
        n_frames = len(self.sim_time)

        # Robot: 54 DOFs + 7 base qpos (free joint at pos 0,0,1.08)
        raw_joint = traj["robot_qpos"]
        full_qpos = np.zeros((n_frames, raw_joint.shape[1]), dtype=np.float32)
        full_qpos[:, :] = raw_joint
        self._joint_qpos = full_qpos

        # Rigid + FEM data
        rigid_names = ["table", "shoe_box", "shoe_box_lid", "coat_hanger", "drying_rack"]
        self._rigid_data = {name: traj[f"rigid_{name}"] for name in rigid_names}
        self._fem_data = {"shirt": traj["shirt_particles"]}

        print(f"Robot qpos: {raw_joint.shape[1]} DOF, Shirt: {self._fem_data['shirt'].shape[1]} vertices")
        if n_frames > 1:
            self.fps = min(int(1.0 / (self.sim_time[1] - self.sim_time[0])), 60)
        return n_frames

    def make_renderer(self):
        import genesis as gs
        from genesis.options.renderers import SphereLight

        _env_exr = str((_REPO / "DemoAssets/textures/san_carlos_left_marvin_modified.exr").resolve())
        return gs.renderers.RayTracer(
            logging_level="warning",
            tracing_depth=32,
            # env_surface=gs.surfaces.Emission(
            #     emissive_texture=gs.textures.ImageTexture(image_path=_env_exr, encoding="linear")),
            env_radius=100.0,
            env_euler=(0, 0, 20),
            lights=[
                SphereLight(pos=(-1.15, 0.0, 2.7), radius=0.2, color=(1.0, 0.97, 0.92), intensity=10.0),
                SphereLight(pos=(1.85, 0.15, 2.5), radius=0.5, color=(0.88, 0.92, 1.0),  intensity=20.0),
                SphereLight(pos=(0.0, -2.0, 3), radius=0.5, color=(1.0, 1.0, 1.0),  intensity=15.0),
                SphereLight(pos=(0.0, 2.0, 3), radius=0.5, color=(1.0, 1.0, 1.0),  intensity=15.0),
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
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        # up defaults to (0,0,1); ease_in/ease_out default to 2.0 (smooth).
        # Floats work: 1.0=linear, 2.0=smooth, 4.0=very slow start/end.
        # Use None for ease_in to cut (instant jump) instead of interpolating.
        # Paste [keyframe] log lines from the K key here:
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

        # Ground — large flat box so surface color override works (Plane ignores it)
        # scene.add_entity(
        #     gs.morphs.Box(size=(20.0, 20.0, 0.02), pos=(0.0, 0.0, -0.01), fixed=True),
        #     surface=gs.surfaces.Plastic(color=(0.05, 0.05, 0.05)),
        # )

        # Table (static)
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

        # Drying rack
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

        # Rigid objects — must match gs-core registry exactly (convexify, decimate, rho)
        self._rigid_entities = {
            "coat_hanger": scene.add_entity(
                gs.morphs.Mesh(
                    file=COAT_HANGER_GLB,
                    pos=(0.53, -0.45, 1.165),
                    euler=(90, 180, 0),
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
            "shoe_box": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHOE_BOX_GLB,
                    pos=(0.40, 0.30, 0.79),
                    scale=0.6,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=gs.materials.Rigid(rho=1000.0),
                surface=gs.surfaces.Default(smooth=False),
                vis_mode="visual",
            ),
            "shoe_box_lid": scene.add_entity(
                gs.morphs.Mesh(
                    file=SHOE_BOX_LID_GLB,
                    pos=(0.55, 0.30, 0.90),
                    scale=0.6,
                    fixed=False,
                    file_meshes_are_zup=False,
                    convexify=False,
                    decimate=True,
                    decimate_face_num=500,
                ),
                material=gs.materials.Rigid(rho=200.0),
                surface=gs.surfaces.Default(smooth=False),
                vis_mode="visual",
            ),
        }

        # FEM cloth — testing Nyx FEM support (ipc_demo branch)
        # if not self.args.nyx:  # re-enable if Nyx FEM support is not available
        self._fem_entities = {}
        self._fem_entities["shirt"] = scene.add_entity(
            gs.morphs.Mesh(
                file=SHIRT_GLB,
                pos=(0.53, -0.37, 1.035),
                euler=(-90, 180, 0),
                scale=0.80,
                fixed=False,
            ),
            material=gs.materials.FEM.Cloth(
                E=1e3,
                nu=0.49,
                rho=60.0,
                thickness=0.0005,
                bending_stiffness=0.1,
            ),
            surface=gs.surfaces.Plastic(),
            vis_mode="visual",
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
    HangerReplay().run()
