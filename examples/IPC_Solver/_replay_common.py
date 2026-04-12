"""
Shared replay utilities for trajectory replay scripts.

Provides:
- Scene creation with no-physics config
- Frame application (robot qpos, rigid poses, FEM particles)
- Camera trajectories (surround, full, ego)
- Interactive and render replay loops
- Nyx / LuisaRender camera setup
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

import genesis as gs

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Camera trajectories ─────────────────────────────────────────────────────


class CameraTrajectory:
    """Base class for camera trajectories. Subclasses implement get_pose()."""

    def get_pose(self, frame_idx: int, n_frames: int) -> tuple[tuple, tuple]:
        """Return (cam_pos, cam_lookat) for a given frame.

        Returns tuple of 3-tuples.
        """
        raise NotImplementedError


class SurroundCamera(CameraTrajectory):
    """Half-circle orbit around a center point, like the yoyo replay.

    Camera orbits from angle_start to angle_end at a fixed height,
    always looking at the center.
    """

    def __init__(
        self,
        center=(0.5, 0.0, 1.0),
        radius=1.5,
        height=1.3,
        angle_start=-60,
        angle_end=60,
    ):
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.angle_start = math.radians(angle_start)
        self.angle_end = math.radians(angle_end)

    def get_pose(self, frame_idx, n_frames):
        t = frame_idx / max(n_frames - 1, 1)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        angle = self.angle_start + (self.angle_end - self.angle_start) * t
        pos = (
            self.center[0] + self.radius * math.cos(angle),
            self.center[1] + self.radius * math.sin(angle),
            self.height,
        )
        lookat = tuple(self.center)
        return pos, lookat


class FullViewCamera(CameraTrajectory):
    """Static camera with full view of the workspace from the front."""

    def __init__(self, pos=(0.5, -1.5, 1.3), lookat=(0.5, 0.0, 1.0)):
        self._pos = tuple(pos)
        self._lookat = tuple(lookat)

    def get_pose(self, frame_idx, n_frames):
        return self._pos, self._lookat


class EgoCamera(CameraTrajectory):
    """Ego-view camera that tracks the robot's head/torso.

    Reads robot base position from trajectory data and offsets the camera
    to simulate a first-person view.
    """

    def __init__(
        self,
        base_pos=(0.0, 0.0, 1.08),
        offset=(0.0, -0.15, 0.45),
        lookat_offset=(0.5, 0.0, -0.2),
    ):
        self.base_pos = np.array(base_pos)
        self.offset = np.array(offset)
        self.lookat_offset = np.array(lookat_offset)

    def get_pose(self, frame_idx, n_frames):
        pos = self.base_pos + self.offset
        lookat = pos + self.lookat_offset
        return tuple(pos), tuple(lookat)


CAMERA_TRAJECTORIES = {
    "surround": SurroundCamera,
    "full": FullViewCamera,
    "ego": EgoCamera,
}


class TrajectoryReplay:
    """Base class for trajectory replay scripts.

    Handles argument parsing, scene/camera creation, and the render/interactive
    loop.  Subclasses override :meth:`add_args`, :meth:`load_trajectory`,
    :meth:`build_scene`, and :meth:`apply_frame` to specialise for each demo.

    Typical usage::

        class MyReplay(TrajectoryReplay):
            ...
        MyReplay().run()
    """

    # Subclass should set these for default output naming
    name: str = "replay"

    # Default camera pose (subclass can override)
    cam_pos: tuple = (1.0, -1.0, 1.0)
    cam_lookat: tuple = (0.5, 0.0, 0.5)
    cam_fov: float = 40
    fps: int = 60

    def __init__(self):
        parser = argparse.ArgumentParser(description=f"Replay trajectory: {self.name}")
        parser.add_argument("--loop", action="store_true", help="Loop replay")
        parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
        parser.add_argument(
            "--camera-traj",
            type=str,
            default=None,
            choices=list(CAMERA_TRAJECTORIES.keys()),
            help="Camera trajectory: surround, full, ego.",
        )
        parser.add_argument("--render", action="store_true", help="Record video")
        parser.add_argument("--nyx", action="store_true", help="Use Nyx renderer")
        parser.add_argument("--start-frame", type=int, default=0, help="Start from this frame")
        parser.add_argument("--save-frames", action="store_true", help="Save each frame as PNG")
        parser.add_argument("--follow", action="store_true", help="Camera follows the robot")
        self.add_args(parser)
        self.args = parser.parse_args()

        # Populated by subclass in load_trajectory / build_scene
        self._robot = None
        self._joint_qpos = None
        self._rigid_entities: dict = {}
        self._rigid_data: dict = {}
        self._fem_entities: dict = {}
        self._fem_data: dict = {}

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add subclass-specific CLI arguments."""

    def load_trajectory(self) -> int:
        """Load trajectory data and return the total number of frames.

        Called once before :meth:`build_scene`.  The subclass should store
        whatever data it needs as instance attributes.
        """
        raise NotImplementedError

    def build_scene(self, scene) -> None:
        """Add entities to *scene*.

        Called after the scene is created but before ``scene.build()``.
        """
        raise NotImplementedError

    def apply_frame(self, scene, frame_idx: int) -> None:
        """Apply trajectory state for *frame_idx* to the scene entities.

        Subclasses should populate ``_robot``, ``_joint_qpos``,
        ``_rigid_entities``, ``_rigid_data``, ``_fem_entities``, and
        ``_fem_data`` in :meth:`load_trajectory` / :meth:`build_scene`.
        """
        # Robot qpos
        if self._joint_qpos is not None and frame_idx < len(self._joint_qpos):
            self._robot.set_qpos(self._joint_qpos[frame_idx])

        # Rigid objects — values can be a single entity or a list
        for name, entities in self._rigid_entities.items():
            if name in self._rigid_data and frame_idx < len(self._rigid_data[name]):
                pose = self._rigid_data[name][frame_idx]
                if not isinstance(entities, (list, tuple)):
                    entities = [entities]
                for ent in entities:
                    ent.set_pos(pose[:3])
                    ent.set_quat(pose[3:])

                # if name == "coat_hanger":
                #     print(entities[0].get_AABB(), pose[:3], ent.get_pos())

        # FEM objects (particle positions)
        for name, entity in self._fem_entities.items():
            if name in self._fem_data and frame_idx < len(self._fem_data[name]):
                entity.set_position(self._fem_data[name][frame_idx])

    def post_build(self) -> None:
        """Called after ``scene.build()``.

        Use this to remap joint data now that the robot entity is fully built.
        Default implementation calls :meth:`_remap_joint_data` if the subclass
        set ``_robot``, ``_joint_names``, and ``_raw_joint_data``.
        """
        if hasattr(self, "_robot") and hasattr(self, "_joint_names") and hasattr(self, "_raw_joint_data"):
            self._remap_joint_data()

    def make_camera_traj(self, name: str) -> "CameraTrajectory":
        """Create a camera trajectory by name.

        Override to customise parameters (e.g. orbit center/radius).
        """
        return CAMERA_TRAJECTORIES[name]()

    # ------------------------------------------------------------------
    # Joint data remapping
    # ------------------------------------------------------------------

    def _remap_joint_data(self):
        """Remap raw joint trajectory data into full qpos arrays.

        Reads ``self._robot``, ``self._joint_names``, ``self._raw_joint_data``
        (and optionally ``self._base_qpos``).  Produces ``self._joint_qpos``
        of shape ``(n_frames, n_qs)`` ready for ``robot.set_qpos(qpos[i])``.
        """
        robot = self._robot
        raw = self._raw_joint_data
        base_qpos = getattr(self, "_base_qpos", None)
        n_frames = raw.shape[0]

        # Build index mapping: trajectory column j -> robot qpos index
        qs_idx_map = []
        for jname in self._joint_names:
            try:
                qs_idx_map.append(robot.get_joint(jname).qs_idx_local[0])
            except Exception:
                qs_idx_map.append(-1)
        matched = sum(1 for x in qs_idx_map if x >= 0)
        print(f"[replay] Joint mapping: {matched}/{len(self._joint_names)} matched")

        # Pre-compute full qpos array for every frame
        qpos_all = np.zeros((n_frames, robot.n_qs), dtype=np.float32)
        if base_qpos is not None:
            qpos_all[:, : len(base_qpos)] = base_qpos
        for j, qi in enumerate(qs_idx_map):
            if qi >= 0:
                qpos_all[:, qi] = raw[:, j]

        self._joint_qpos = qpos_all

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _make_scene(self):
        """Create a Genesis scene configured for replay (no physics)."""
        import genesis as gs

        use_render = self.args.render
        use_nyx = self.args.nyx

        renderer_kwargs = {}
        if use_render and not use_nyx:
            from _yoyo_common import make_raytracer

            renderer_kwargs["renderer"] = make_raytracer()

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / 60, gravity=(0, 0, 0)),
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, 0),
                enable_collision=False,
                enable_self_collision=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=self.cam_pos,
                camera_lookat=self.cam_lookat,
                camera_fov=self.cam_fov,
            ),
            vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.35)),
            show_viewer=not use_render,
            **renderer_kwargs,
        )

    def _add_camera(self):
        """Add a render camera (Nyx or Luisa) to the scene."""
        if self.args.nyx:
            from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
            from gs_nyx_plugin.nyx_camera_sensor import NyxCameraSensor  # noqa: F401 — registers sensor class
            import gs_nyx.nyx_py_renderer as npr
            import gs_nyx.nyx_py_sdk as ap

            env_map = ap.EnvironmentMapAsset()
            env_map.texture = str((_REPO_ROOT / "DemoAssets/textures/san_carlos_left_marvin_modified.exr").resolve())
            env_map.rotation = 0.0
            env_map.multiplier = 1.0

            self._cam = self._scene.add_sensor(
                NyxCameraOptions(
                    res=(1920, 1080),
                    pos=self.cam_pos,
                    lookat=self.cam_lookat,
                    fov=self.cam_fov,
                    spp=256,
                    denoise=True,
                    render_mode=npr.ERenderMode.RefPathTracer,
                    env_maps=(env_map,),
                )
            )
        else:
            self._cam = self._scene.add_camera(
                res=(1920, 1080),
                pos=self.cam_pos,
                lookat=self.cam_lookat,
                fov=self.cam_fov,
                spp=256,
            )

    def run(self) -> None:
        args = self.args
        use_render = args.render
        use_nyx = args.nyx

        gs.init(backend=gs.gpu if use_render else gs.cpu, logging_level="warning")

        self._n_frames = self.load_trajectory()
        print(f"[{self.name}] {self._n_frames} frames")

        self._make_scene()
        self.build_scene(self._scene)

        self._cam = None
        if use_render:
            self._add_camera()

        self._scene.build(n_envs=1 if use_nyx else 0)
        self.post_build()

        self._camera_traj = None
        if args.camera_traj:
            self._camera_traj = self.make_camera_traj(args.camera_traj)

        if use_render:
            self._run_render()
        else:
            self._run_interactive()

    def _apply(self, frame_idx: int) -> None:
        self.apply_frame(self._scene, frame_idx)
        self._scene._visualizer.update_visual_states(force_render=True)

    def _render_frame(self) -> np.ndarray:
        """Render one frame and return the RGB array."""
        use_nyx = self.args.nyx
        if use_nyx:
            # In replay mode scene.t never advances, so the camera's
            # staleness check thinks the cache is still fresh.  Force it
            # stale so _render_current_state() runs on each read().
            self._cam._stale = True
            data = self._cam.read()
            rgb = data.rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
        else:
            rgb_result = self._cam.render(rgb=True, force_render=True)
            rgb_tensor = rgb_result[0]
            rgb = rgb_tensor.cpu().numpy() if hasattr(rgb_tensor, "cpu") else np.array(rgb_tensor)
        return rgb

    def _run_render(self) -> None:
        import imageio

        args = self.args
        use_nyx = args.nyx
        renderer_name = "nyx" if use_nyx else "luisa"
        traj_name = getattr(args, "trajectory", "default")
        stem = f"ipc_{self.name}_{traj_name}_{renderer_name}"
        default_output = f"data/ipc_demo/ipc_{self.name}/{stem}.mp4"
        output = getattr(args, "output", None) or default_output
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        frames_dir = None
        if args.save_frames:
            frames_dir = out.parent / f"{stem}_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

        if not use_nyx:
            self._cam.start_recording()

        start = args.start_frame
        if start > 0:
            print(f"[render] Skipping to frame {start}")
            self._apply(start - 1)

        # Follow mode: camera tracks the robot with a fixed offset
        follow = args.follow
        if follow:
            cam_offset = np.array(self.cam_pos) - np.array(self.cam_lookat)

        frames_rgb = []
        for i in range(start, self._n_frames):
            self._apply(i)

            if follow and self._robot is not None:
                robot_pos = self._robot.get_pos()
                if robot_pos.ndim > 1:
                    robot_pos = robot_pos[0]
                robot_pos = robot_pos.cpu().numpy()
                lookat = (float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]) + 0.5)
                cam_pos = tuple(np.array(lookat) + cam_offset)
                if use_nyx:
                    self._cam.update_camera_pose(pos=cam_pos, lookat=lookat, up=(0, 0, 1))
                else:
                    self._cam.set_pose(pos=cam_pos, lookat=lookat, up=(0, 0, 1))
            elif self._camera_traj is not None:
                cam_pos, cam_lookat = self._camera_traj.get_pose(i, self._n_frames)
                if use_nyx:
                    self._cam.update_camera_pose(pos=cam_pos, lookat=cam_lookat, up=(0, 0, 1))
                else:
                    self._cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=(0, 0, 1))

            rgb = self._render_frame()
            if frames_dir is not None:
                imageio.imwrite(str(frames_dir / f"{i:05d}.png"), rgb)
            frames_rgb.append(rgb)
            if (i + 1) % 100 == 0:
                print(f"[render] Frame {i + 1}/{self._n_frames}")

        writer = imageio.get_writer(str(out), fps=self.fps)
        for rgb in frames_rgb:
            writer.append_data(rgb)
        writer.close()

        if not use_nyx:
            self._cam.stop_recording()

        print(f"[render] Saved {out} ({len(frames_rgb)} frames, {self.fps} fps)")
        if frames_dir is not None:
            print(f"[render] Individual frames in {frames_dir}/")

    def _run_interactive(self) -> None:
        args = self.args
        dt_frame = 1.0 / 60 / args.speed
        while True:
            for i in range(self._n_frames):
                t0 = time.perf_counter()
                self._apply(i)
                self._scene._visualizer.update(force=True)
                elapsed = time.perf_counter() - t0
                if elapsed < dt_frame:
                    time.sleep(dt_frame - elapsed)
                if (i + 1) % 200 == 0:
                    print(f"Frame {i + 1}/{self._n_frames}")
            print("Trajectory complete.")
            if not args.loop:
                break
        # Keep viewer open
        while True:
            self._scene._visualizer.update(force=True)
            time.sleep(1 / 60)
