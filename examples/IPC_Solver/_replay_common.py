"""
Shared replay utilities for trajectory replay scripts.

Provides:
- Scene creation with no-physics config
- Frame application (robot qpos, rigid poses, FEM particles)
- Camera trajectories (surround, full, ego, custom)
- Interactive and render replay loops
- Nyx / LuisaRender camera setup
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
import numpy as np

import genesis as gs

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Camera trajectories ─────────────────────────────────────────────────────


class CameraTrajectory:
    """Base class for camera trajectories. Subclasses implement get_pose()."""

    def get_pose(self, frame_idx: int, n_frames: int) -> tuple[tuple, tuple]:
        """Return (cam_pos, cam_lookat) for a given frame."""
        raise NotImplementedError

    def get_up(self, frame_idx: int) -> tuple:
        """Return the camera up vector for a given frame. Default: world-Z up."""
        return (0.0, 0.0, 1.0)


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


def _ease(t: float, ease_in: float, ease_out: float) -> float:
    """Remap t ∈ [0, 1] with independent ease-in / ease-out power curves.

    Both exponents default to 2.0 (smooth quadratic ease-in-out).
    Setting either exponent to 1.0 makes that half linear.
    Higher values produce a more abrupt hold then fast snap; lower values
    (e.g. 0.5) invert to ease-out-then-in (overshoot feel).

    Examples
    --------
    (1, 1)  — linear
    (2, 2)  — symmetric ease-in-out  (default)
    (3, 1)  — strong ease-in, linear ease-out
    (1, 3)  — linear ease-in, strong ease-out
    (4, 4)  — very slow start/end, quick middle
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if t < 0.5:
        return 0.5 * (2.0 * t) ** ease_in
    else:
        return 1.0 - 0.5 * (2.0 * (1.0 - t)) ** ease_out


class CustomCamera(CameraTrajectory):
    """Keyframe-driven camera with smooth ease-in/ease-out interpolation.

    Each keyframe entry: ``(frame, pos, lookat[, up[, ease_in[, ease_out]]])``

    - **frame**: animation frame index where this keyframe begins.
    - **pos / lookat**: 3-tuples (capture with the ``K`` key).
    - **up**: 3-tuple camera-up vector (captured by ``K``; defaults to ``(0,0,1)``).
    - **ease_in**: power of the acceleration curve leaving this keyframe (default 2.0).
    - **ease_out**: power of the deceleration curve arriving at the next keyframe (default 2.0).

    Before the first keyframe the scene default pos/lookat is used.
    After the last keyframe that pose is held.
    """

    def __init__(self, keyframes: list, default_pos: tuple, default_lookat: tuple):
        normalized = []
        for entry in keyframes:
            frame   = int(entry[0])
            pos     = np.array(entry[1], dtype=float)
            lookat  = np.array(entry[2], dtype=float)
            up      = np.array(entry[3] if len(entry) > 3 else (0.0, 0.0, 1.0), dtype=float)
            ein     = None if (len(entry) > 4 and entry[4] is None) else float(entry[4]) if len(entry) > 4 else 2.0
            eout    = None if (len(entry) > 5 and entry[5] is None) else float(entry[5]) if len(entry) > 5 else 2.0
            normalized.append((frame, pos, lookat, up, ein, eout))
        self._kf = sorted(normalized, key=lambda x: x[0])
        self._default_pos    = np.array(default_pos,    dtype=float)
        self._default_lookat = np.array(default_lookat, dtype=float)

    def _interp(self, frame_idx: int):
        """Return interpolated (pos, lookat, up) arrays for *frame_idx*."""
        if not self._kf:
            return self._default_pos, self._default_lookat, np.array([0.0, 0.0, 1.0])

        # Before first keyframe — hold scene defaults
        if frame_idx <= self._kf[0][0]:
            if frame_idx < self._kf[0][0]:
                return self._default_pos.copy(), self._default_lookat.copy(), np.array([0.0, 0.0, 1.0])
            f, p, l, u, _, _ = self._kf[0]
            return p.copy(), l.copy(), u.copy()

        # After last keyframe — hold last pose
        if frame_idx >= self._kf[-1][0]:
            f, p, l, u, _, _ = self._kf[-1]
            return p.copy(), l.copy(), u.copy()

        # Find the enclosing segment and interpolate
        for i in range(len(self._kf) - 1):
            f0, p0, l0, u0, ein, eout = self._kf[i]
            f1, p1, l1, u1, _,   _    = self._kf[i + 1]
            if f0 <= frame_idx < f1:
                if ein is None:  # cut: hold at this keyframe until the next
                    return p0.copy(), l0.copy(), u0.copy()
                t_raw = (frame_idx - f0) / (f1 - f0)
                t = _ease(t_raw, ein, eout)
                pos    = p0 + t * (p1 - p0)
                lookat = l0 + t * (l1 - l0)
                up     = u0 + t * (u1 - u0)
                norm   = np.linalg.norm(up)
                if norm > 1e-6:
                    up /= norm
                return pos, lookat, up

        f, p, l, u, _, _ = self._kf[-1]
        return p.copy(), l.copy(), u.copy()

    def get_pose(self, frame_idx: int, n_frames: int):
        pos, lookat, _ = self._interp(frame_idx)
        return tuple(pos), tuple(lookat)

    def get_up(self, frame_idx: int) -> tuple:
        _, _, up = self._interp(frame_idx)
        return tuple(up)


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
            choices=list(CAMERA_TRAJECTORIES.keys()) + ["custom"],
            help="Camera trajectory: surround, full, ego.",
        )
        parser.add_argument("--render", action="store_true", help="Record video")
        parser.add_argument("--render-all-keyframes", action="store_true",
                            help="Render one video per custom keyframe (fixed static camera for each). "
                                 "Videos are named kf01_, kf02_, … and share a datetime tag.")
        parser.add_argument("--start-keyframe", type=int, default=1, metavar="N",
                            help="Start the --render-all-keyframes batch at keyframe N (1-based). "
                                 "--start-frame applies to that first keyframe; later shots start at frame 0. "
                                 "Use this to resume an interrupted batch.")
        parser.add_argument("--spp", type=int, default=256, help="Samples-per-pixel for the render camera (default: 256)")
        parser.add_argument("--dof", action="store_true", help="Enable depth-of-field (thinlens camera model)")
        parser.add_argument("--aperture", type=float, default=1.4, help="Aperture f-number for DOF (default: 1.4, lower=shallower)")
        parser.add_argument("--focus-dist", type=float, default=None, help="Focus distance in metres (default: auto from cam_pos/lookat)")
        parser.add_argument("--focal-len", type=float, default=0.05, help="Focal length in metres for DOF (default: 0.05 = 50mm)")
        parser.add_argument("--nyx", action="store_true", help="Use Nyx renderer")
        parser.add_argument("--start-frame", type=int, default=0, help="Start from this frame (interactive + render)")
        parser.add_argument("--end-frame", type=int, default=None, help="Stop at this frame exclusive (default: last frame)")
        parser.add_argument("--no-raytracer", action="store_true", help="Suppress Luisa renderer: with --nyx shows only the Nyx preview; without --nyx shows no preview window")
        parser.add_argument("--save-frames", action="store_true", help="Save each frame as PNG")
        parser.add_argument("--follow", action="store_true", help="Camera follows the robot")
        parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame (subsample)")
        parser.add_argument("--dark-bg", action="store_true", help="Dark grey (0.01) background, no extra lights (Nyx only)")
        parser.add_argument(
            "--preview",
            action="store_true",
            help="Show live Luisa render preview in an OpenCV window (interactive mode only)",
        )
        parser.add_argument(
            "--preview-spp",
            type=int,
            default=64,
            help="Samples-per-pixel for the preview camera (default: 4)",
        )
        parser.add_argument(
            "--res",
            type=int,
            nargs=2,
            metavar=("W", "H"),
            default=[1920, 1080],
            help="Resolution in pixels for render and preview cameras (default: 1920 1080)",
        )
        parser.add_argument("--camera-keyframe", type=int, default=None, metavar="N",
                            help="Start camera at custom keyframe N (1-based, matches [ / ] display)")
        parser.add_argument("--debug-lights", action="store_true",
                            help="Add sphere markers in the viewer for each Luisa light (position + radius)")
        self.add_args(parser)
        self.args = parser.parse_args()

        # Populated by subclass in load_trajectory / build_scene
        self._robot = None
        self._joint_qpos = None
        self._rigid_entities: dict = {}
        self._rigid_data: dict = {}
        self._fem_entities: dict = {}
        self._fem_data: dict = {}
        self._light_markers: list = []  # sphere entities mirroring raytracer lights

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

    def make_renderer(self):
        """Create the RayTracer renderer. Override to use scene-specific lights."""
        from _yoyo_common import make_raytracer
        return make_raytracer()

    def nyx_lights(self) -> list:
        """Return lights to pass to Nyx cameras. Override to match make_renderer() lights.

        Each entry is a dict with keys: type ("point"/"directional"), pos, radius,
        color, intensity — same format as scene_runner.py LIGHTS.
        """
        return []

    def nyx_light_field(self) -> dict | None:
        """Return Gaussian splat light field config, or None to skip.

        Override in subclasses to provide a 3DGS background for Nyx renders.
        The ``position`` / ``rotation`` / ``scale`` transform aligns the captured
        splat to the scene's coordinate frame — tune these while running with
        ``--nyx --preview``.

        Example::

            def nyx_light_field(self):
                return {
                    "uri": str((_REPO_ROOT / "examples/IPC_Solver/0325_san_carlos_robot_station.ply").resolve()),
                    "position": (1.5, 0.81, -3.0),
                    "rotation": (0.0, 0.701707, 0.0, 0.701707),  # (w, x, y, z)
                    "scale": (1.0, 1.0, 1.0),
                }
        """
        return None

    def _dof_kwargs(self) -> dict:
        """Return thinlens DOF kwargs for add_camera if --dof is set, else empty dict."""
        if not self.args.dof:
            return {}
        import math
        if self.args.focus_dist is not None:
            focus_dist = self.args.focus_dist
        else:
            p = self.cam_pos
            l = self.cam_lookat
            focus_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, l)))
        return {
            "model": "thinlens",
            "aperture": self.args.aperture,
            "focal_len": self.args.focal_len,
            "focus_dist": focus_dist,
        }

    def custom_camera_keyframes(self) -> list:
        """Return keyframes for the ``custom`` camera trajectory.

        Override in subclasses.  Each entry: ``(frame_idx, pos, lookat)``
        where *pos* and *lookat* are 3-tuples.  Frames before the first entry
        use the scene default; after the last entry that pose is held.
        """
        return []

    def make_camera_traj(self, name: str) -> "CameraTrajectory":
        """Create a camera trajectory by name.

        Override to customise parameters (e.g. orbit center/radius).
        """
        if name == "custom":
            return CustomCamera(self.custom_camera_keyframes(), self.cam_pos, self.cam_lookat)
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

    def _build_nyx_light_field(self):
        """Instantiate a LightFieldAsset from nyx_light_field() config, or return None."""
        cfg = self.nyx_light_field()
        if cfg is None:
            return None
        import gs_nyx.nyx_py_sdk as nps
        lf = nps.LightFieldAsset()
        lf.uri = cfg["uri"]
        lf.type = nps.ELightFieldType.GaussianField
        lf.position = nps.float3(*cfg.get("position", (0.0, 0.0, 0.0)))
        lf.rotation = nps.quaternion(*cfg.get("rotation", (1.0, 0.0, 0.0, 0.0)))
        lf.scale = nps.float3(*cfg.get("scale", (1.0, 1.0, 1.0)))
        return lf

    def _make_scene(self):
        """Create a Genesis scene configured for replay (no physics)."""
        import genesis as gs

        use_render = self.args.render or getattr(self.args, "render_all_keyframes", False)
        use_nyx = self.args.nyx

        renderer_kwargs = {}
        if (use_render and not use_nyx) or self._use_luisa_preview:
            renderer_kwargs["renderer"] = self.make_renderer()

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / 60, gravity=(0, 0, 0)),
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, 0),
                enable_collision=False,
                enable_self_collision=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(self.args.res[0], self.args.res[1]),
                camera_pos=self.cam_pos,
                camera_lookat=self.cam_lookat,
                camera_fov=self.cam_fov,
            ),
            vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.35)),
            show_viewer=not use_render,
            **renderer_kwargs,
        )
        self._preview_cam = None
        self._preview_nyx_cam = None

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
            if getattr(self.args, "dark_bg", False):
                # Dark grey background (0.01, 0.01, 0.01) like trashbag scene.
                # No extra directional lights — relies on subclass nyx_lights().
                env_map.texture = str((_REPO_ROOT / "DemoAssets/textures/dark_grey.exr").resolve())
                env_map.multiplier = 1.0

            lights = self.nyx_lights()
            light_field = self._build_nyx_light_field()
            self._cam = self._scene.add_sensor(
                NyxCameraOptions(
                    res=(self.args.res[0], self.args.res[1]),
                    pos=self.cam_pos,
                    lookat=self.cam_lookat,
                    fov=self.cam_fov,
                    spp=self.args.spp,
                    denoise=True,
                    render_mode=npr.ERenderMode.RefPathTracer,
                    env_maps=(env_map,),
                    light_fields=(light_field,) if light_field is not None else (),
                    **({"lights": lights} if lights else {}),
                )
            )
        else:
            self._cam = self._scene.add_camera(
                res=(self.args.res[0], self.args.res[1]),
                pos=self.cam_pos,
                lookat=self.cam_lookat,
                fov=self.cam_fov,
                spp=self.args.spp,
                **self._dof_kwargs(),
            )

    def _add_preview_camera(self):
        """Add preview cameras — Luisa if not --no-raytracer, Nyx if --nyx is set."""
        w, h = self.args.res

        # Luisa preview camera (skipped when --no-raytracer)
        self._preview_cam = None
        if self._use_luisa_preview:
            self._preview_cam = self._scene.add_camera(
                res=(w, h),
                pos=self.cam_pos,
                lookat=self.cam_lookat,
                fov=self.cam_fov,
                spp=self.args.preview_spp,
                **self._dof_kwargs(),
            )

        # Nyx preview camera (independent of --no-raytracer; added when --nyx)
        self._preview_nyx_cam = None
        if self.args.nyx:
            from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
            from gs_nyx_plugin.nyx_camera_sensor import NyxCameraSensor  # noqa: F401
            import gs_nyx.nyx_py_renderer as npr
            import gs_nyx.nyx_py_sdk as ap

            env_map = ap.EnvironmentMapAsset()
            env_map.texture = str((_REPO_ROOT / "DemoAssets/textures/san_carlos_left_marvin_modified.exr").resolve())
            env_map.rotation = 0.0
            env_map.multiplier = 1.0

            lights = self.nyx_lights()
            light_field = self._build_nyx_light_field()
            self._preview_nyx_cam = self._scene.add_sensor(
                NyxCameraOptions(
                    res=(w, h),
                    pos=self.cam_pos,
                    lookat=self.cam_lookat,
                    fov=self.cam_fov,
                    spp=self.args.preview_spp,
                    denoise=True,
                    render_mode=npr.ERenderMode.FastPathTracer,
                    env_maps=(env_map,),
                    light_fields=(light_field,) if light_field is not None else (),
                    **({"lights": lights} if lights else {}),
                )
            )

    def run(self) -> None:
        args = self.args
        # --render-all-keyframes implies --render
        use_render = args.render or getattr(args, "render_all_keyframes", False)
        use_nyx = args.nyx

        # Luisa preview: --preview without --no-raytracer (and not in render mode)
        # Nyx preview:   --preview --nyx (independent of --no-raytracer)
        use_luisa_preview = args.preview and not use_render and not args.no_raytracer
        use_nyx_preview = args.preview and not use_render and use_nyx
        self._use_luisa_preview = use_luisa_preview

        gs.init(backend=gs.gpu if (use_render or use_luisa_preview or use_nyx) else gs.cpu, logging_level="warning")

        self._n_frames = self.load_trajectory()
        print(f"[{self.name}] {self._n_frames} frames")

        if args.camera_keyframe is not None:
            kfs = self.custom_camera_keyframes()
            idx = args.camera_keyframe - 1
            if 0 <= idx < len(kfs):
                entry = kfs[idx]
                self.cam_pos    = tuple(float(v) for v in entry[1])
                self.cam_lookat = tuple(float(v) for v in entry[2])
                print(f"[camera] keyframe {args.camera_keyframe}/{len(kfs)}  "
                      f"pos={self.cam_pos}  lookat={self.cam_lookat}")
            else:
                print(f"[camera] --camera-keyframe {args.camera_keyframe} out of range "
                      f"(1–{len(kfs)}), ignoring")

        self._make_scene()
        self.build_scene(self._scene)
        if getattr(args, "debug_lights", False):
            self._add_light_markers(self._scene)

        self._cam = None
        if use_render:
            self._add_camera()
        if use_luisa_preview or use_nyx_preview:
            self._add_preview_camera()

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
        """Render entry point. Dispatches to _render_to_file, looping over
        keyframes when --render-all-keyframes is set."""
        # All videos in a batch share one timestamp so they sort together.
        dt_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        if getattr(self.args, "render_all_keyframes", False):
            kfs = self.custom_camera_keyframes()
            if not kfs:
                print("[render] --render-all-keyframes: no keyframes defined, falling back to single render")
                self._render_to_file(dt_tag=dt_tag)
                return
            start_kf = max(0, getattr(self.args, "start_keyframe", 1) - 1)  # 0-based
            n_remaining = len(kfs) - start_kf
            print(f"[render] Rendering {n_remaining}/{len(kfs)} keyframe shot(s)"
                  + (f" (resuming from kf{start_kf + 1:02d})" if start_kf > 0 else "") + " …")
            for kf_idx, kf in enumerate(kfs):
                if kf_idx < start_kf:
                    continue
                print(f"\n[render] ── Shot {kf_idx + 1}/{len(kfs)} ──")
                # --start-frame applies only to the first shot of the batch; later shots start at 0
                start_frame_override = None if kf_idx == start_kf else 0
                self._render_to_file(
                    dt_tag=dt_tag,
                    kf_idx=kf_idx,
                    kf_pos=tuple(float(v) for v in kf[1]),
                    kf_lookat=tuple(float(v) for v in kf[2]),
                    start_frame_override=start_frame_override,
                )
        else:
            self._render_to_file(dt_tag=dt_tag)

    def _render_to_file(
        self,
        dt_tag: str,
        kf_idx: int | None = None,
        kf_pos: tuple | None = None,
        kf_lookat: tuple | None = None,
        start_frame_override: int | None = None,
    ) -> None:
        """Render the full frame range to a single MP4.

        When *kf_idx* / *kf_pos* / *kf_lookat* are supplied the camera is
        locked to that fixed position for the entire video (ignoring any
        --camera-traj).  Otherwise the normal camera-traj logic applies.

        *start_frame_override* replaces --start-frame for this specific call.
        Used by --render-all-keyframes to let --start-frame apply only to the
        first (possibly resumed) shot while later shots start at frame 0.
        """
        import imageio

        args = self.args
        use_nyx = args.nyx
        renderer_name = "nyx" if use_nyx else "luisa"
        traj_name = getattr(args, "trajectory", "default")

        if kf_idx is not None:
            stem = f"ipc_{self.name}_kf{kf_idx + 1:02d}_{renderer_name}_{dt_tag}"
        else:
            stem = f"ipc_{self.name}_{traj_name}_{renderer_name}_{dt_tag}"

        # --output is honoured only for single-render mode (would collide in batch)
        default_output = f"data/ipc_demo/ipc_{self.name}/{stem}.mp4"
        output = (getattr(args, "output", None) if kf_idx is None else None) or default_output
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        frames_dir = None
        if args.save_frames:
            frames_dir = out.parent / f"{stem}_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

        # Point the camera at the keyframe position (static for the whole video)
        if kf_pos is not None:
            if use_nyx:
                self._cam.update_camera_pose(pos=kf_pos, lookat=kf_lookat, up=(0, 0, 1))
            else:
                self._cam.set_pose(pos=kf_pos, lookat=kf_lookat, up=(0, 0, 1))

        start = args.start_frame if start_frame_override is None else start_frame_override
        end = args.end_frame if args.end_frame is not None else self._n_frames
        print(f"[render] {out.name}  frames {start}–{end - 1} of {self._n_frames - 1}")
        if start > 0:
            self._apply(start - 1)

        # Follow mode: camera tracks the robot with a fixed offset
        follow = args.follow
        if follow:
            cam_offset = np.array(self.cam_pos) - np.array(self.cam_lookat)

        stride = max(1, args.stride)
        n_written = 0
        interrupted = False
        writer = imageio.get_writer(str(out), fps=max(1, self.fps // stride),
                                    codec="libx264", macro_block_size=1)
        try:
            for i in range(start, end):
                if (i - start) % stride != 0:
                    # Still advance scene state for correct FK/cloth, but skip render
                    self._apply(i)
                    continue
                self._apply(i)

                # Per-frame camera update — skipped for fixed keyframe shots
                if kf_idx is None:
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
                writer.append_data(rgb)
                n_written += 1
                if i % 100 == 0 and i > start:
                    print(f"[render] Frame {i}/{end}")
        except KeyboardInterrupt:
            interrupted = True
        finally:
            writer.close()

        if interrupted:
            print(f"[render] Interrupted — partial video saved: {out} ({n_written} frames, {max(1, self.fps // stride)} fps)")
        else:
            print(f"[render] Saved {out} ({n_written} frames, {max(1, self.fps // stride)} fps)")
        if frames_dir is not None:
            print(f"[render] Individual frames in {frames_dir}/")

    # Distinct bright colors per light index (RGB 0-1), cycling if more than 4
    _LIGHT_MARKER_COLORS = [
        (1.0, 0.4, 0.1),   # 0: warm orange  — key
        (0.1, 0.8, 0.3),   # 1: green        — fill
        (0.2, 0.4, 1.0),   # 2: blue         — rim
        (1.0, 0.9, 0.1),   # 3: yellow       — back-ambient
        (1.0, 0.1, 0.8),   # 4+: magenta fallback
    ]

    def _add_light_markers(self, scene) -> None:
        """Add a small sphere entity in the viewer for each Luisa raytracer light.

        Each marker sits at the light's position with its actual radius, so you can
        see both location and size. Markers are non-physics rigid bodies (gravity=0,
        collisions disabled) so they stay in place. Their positions are updated by
        _register_light_keybinds when you move lights with 1-6 / TAB.
        """
        raytracer = scene._visualizer.raytracer
        if raytracer is None or not raytracer.lights:
            return

        colors = self._LIGHT_MARKER_COLORS
        for i, light in enumerate(raytracer.lights):
            color = colors[min(i, len(colors) - 1)]
            marker = scene.add_entity(
                gs.morphs.Sphere(
                    pos=tuple(float(v) for v in light.pos),
                    radius=float(light.radius),
                    fixed=False,
                ),
                surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ColorTexture(color=color)),
            )
            self._light_markers.append(marker)
            print(f"[lights] marker {i}: pos={tuple(round(float(v),3) for v in light.pos)}  "
                  f"radius={light.radius:.3f}  color={color}")

    def _register_light_keybinds(self) -> None:
        """Register keybinds to move lights interactively.

        TAB                = cycle active light (0 → 1 → 0 → ...)
        1/2/3/4/5/6        = move active light: +Y / -Y / -X / +X / +Z / -Z

        Only registers if the scene has a raytracer with at least one light.
        Prints updated positions after each move so they can be copied back into code.
        """
        from genesis.vis.keybindings import Key, KeyAction, Keybind

        raytracer = self._scene._visualizer.raytracer
        if raytracer is None or not raytracer.lights:
            return

        STEP = 0.05  # metres per key event
        n_lights = len(raytracer.lights)
        active = [0]  # mutable index of the currently selected light

        def _cycle_light() -> None:
            active[0] = (active[0] + 1) % n_lights
            print(f"[lights] active: light {active[0]}")

        BINDINGS = [
            (Key._1, np.array([ 0,  1,  0])),
            (Key._2, np.array([ 0, -1,  0])),
            (Key._3, np.array([-1,  0,  0])),
            (Key._4, np.array([ 1,  0,  0])),
            (Key._5, np.array([ 0,  0,  1])),
            (Key._6, np.array([ 0,  0, -1])),
        ]

        def _move(axis: np.ndarray) -> None:
            idx = active[0]
            if idx >= len(raytracer.lights):
                return
            light = raytracer.lights[idx]
            light.pos = light.pos + axis * STEP
            raytracer.update_rigid(light.name, gs.trans_to_T(light.pos))
            # Keep debug marker in sync
            if self._light_markers and idx < len(self._light_markers):
                self._light_markers[idx].set_pos(light.pos)
            positions = [tuple(np.round(l.pos, 3).tolist()) for l in raytracer.lights]
            print(f"[lights] light {idx} → {positions}")

        binds = [
            Keybind("light_cycle", Key.TAB, KeyAction.RELEASE, callback=_cycle_light),
        ]
        for key, axis in BINDINGS:
            binds.append(Keybind(f"light_{key.name}", key, KeyAction.HOLD,
                                 callback=_move, args=(axis,)))

        self._scene.viewer.register_keybinds(*binds)
        n = len(raytracer.lights)
        print(f"[lights] TAB=cycle active light  1-6=move  ({n} light(s), active: light 0)")

    def _run_interactive(self) -> None:
        from genesis.vis.keybindings import Key, KeyAction, Keybind

        args = self.args
        start = args.start_frame
        end = args.end_frame if args.end_frame is not None else self._n_frames
        print(f"[replay] frames {start}–{end - 1} of {self._n_frames - 1}  speed={args.speed}x")
        use_preview = self._preview_cam is not None or self._preview_nyx_cam is not None

        # ── State flags (mutated from keybind callbacks) ──────────────────────
        _pause = [False]
        _reset = [False]
        _speed = [args.speed]
        _current_frame = [start]

        # Keyframe browsing: [ / ] jump to prev/next custom keyframe
        _custom_kfs = self.custom_camera_keyframes()
        _kf_idx = [0]
        _kf_jump = [False]  # pending jump flag — index is always read from _kf_idx

        def _on_pause():
            _pause[0] = not _pause[0]

        def _on_reset():
            _reset[0] = True
            _pause[0] = False

        def _on_speed_up():
            _speed[0] *= 2.0
            print(f"[replay] speed={_speed[0]:.2f}x")

        def _on_speed_down():
            _speed[0] /= 2.0
            print(f"[replay] speed={_speed[0]:.2f}x")

        def _on_keyframe():
            pos    = tuple(round(float(v), 4) for v in self._scene.viewer.camera_pos)
            lookat = tuple(round(float(v), 4) for v in self._scene.viewer.camera_lookat)
            up     = tuple(round(float(v), 4) for v in self._scene.viewer.camera_up)
            print(f"[keyframe] ({_current_frame[0]}, {pos}, {lookat}, {up}),")

        def _on_prev_kf() -> None:
            if not _custom_kfs:
                return
            _kf_idx[0] = (_kf_idx[0] - 1) % len(_custom_kfs)
            _kf_jump[0] = True
            _pause[0] = True

        def _on_next_kf() -> None:
            if not _custom_kfs:
                return
            _kf_idx[0] = (_kf_idx[0] + 1) % len(_custom_kfs)
            _kf_jump[0] = True
            _pause[0] = True

        def _do_kf_jump() -> None:
            """Apply a pending keyframe camera jump. Must be called from the main loop thread.

            Reads _kf_idx at execution time so rapid [ / ] presses never lose a jump —
            the index is always correct even if multiple key events fired before this ran.
            Only moves the camera — does not seek the scene to the keyframe's frame.
            """
            if not _kf_jump[0]:
                return
            _kf_jump[0] = False
            idx = _kf_idx[0]
            entry = _custom_kfs[idx]
            kf_frame  = int(entry[0])
            kf_pos    = np.array(entry[1], dtype=float)
            kf_lookat = np.array(entry[2], dtype=float)
            raw_up    = entry[3] if len(entry) > 3 and entry[3] is not None else (0.0, 0.0, 1.0)
            kf_up     = np.array(raw_up, dtype=float)
            self._scene.viewer._camera_up = kf_up
            self._scene.viewer.set_camera_pose(pos=kf_pos, lookat=kf_lookat)
            n = len(_custom_kfs)
            print(f"[keyframe] {idx + 1}/{n}  frame={kf_frame}  "
                  f"pos={tuple(round(float(v), 4) for v in kf_pos)}  "
                  f"lookat={tuple(round(float(v), 4) for v in kf_lookat)}")

        self._scene.viewer.register_keybinds(
            Keybind("replay_pause",      Key.SPACE,        KeyAction.RELEASE, callback=_on_pause),
            Keybind("replay_reset",      Key.BACKSPACE,    KeyAction.RELEASE, callback=_on_reset),
            Keybind("replay_speed_up",   Key.PERIOD,       KeyAction.RELEASE, callback=_on_speed_up),
            Keybind("replay_speed_down", Key.COMMA,        KeyAction.RELEASE, callback=_on_speed_down),
            Keybind("replay_keyframe",   Key.K,            KeyAction.RELEASE, callback=_on_keyframe),
            Keybind("replay_prev_kf",    Key.BRACKETLEFT,  KeyAction.RELEASE, callback=_on_prev_kf),
            Keybind("replay_next_kf",    Key.BRACKETRIGHT, KeyAction.RELEASE, callback=_on_next_kf),
        )
        self._register_light_keybinds()

        if use_preview:
            import cv2
            w, h = args.res
            if self._preview_cam is not None:
                cv2.namedWindow("preview (luisa)", cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("preview (luisa)", w, h)
            if self._preview_nyx_cam is not None:
                cv2.namedWindow("preview (nyx)", cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("preview (nyx)", w, h)

        def _update_preview(frame_idx: int) -> None:
            if not use_preview:
                return
            if self._camera_traj is not None:
                cam_pos, cam_lookat = self._camera_traj.get_pose(frame_idx, self._n_frames)
            else:
                cam_pos = tuple(self._scene.viewer.camera_pos)
                cam_lookat = tuple(self._scene.viewer.camera_lookat)

            # Luisa window (skipped when --no-raytracer)
            if self._preview_cam is not None:
                self._preview_cam.set_pose(pos=cam_pos, lookat=cam_lookat, up=(0, 0, 1))
                rgb_result = self._preview_cam.render(rgb=True, force_render=True)
                rgb = rgb_result[0]
                rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
                cv2.imshow("preview (luisa)", rgb[..., ::-1].copy())

            # Nyx window (only when --nyx)
            if self._preview_nyx_cam is not None:
                self._preview_nyx_cam.update_camera_pose(pos=cam_pos, lookat=cam_lookat, up=(0, 0, 1))
                self._preview_nyx_cam._render_current_state()
                rgb_nyx = self._preview_nyx_cam._get_image_cache_entry()[0]
                rgb_nyx = rgb_nyx.cpu().numpy() if hasattr(rgb_nyx, "cpu") else np.array(rgb_nyx)
                cv2.imshow("preview (nyx)", rgb_nyx[..., ::-1].copy())

            cv2.waitKey(1)

        # ── Camera-settle tracker ─────────────────────────────────────────────
        _last_cam_pos    = [None]
        _last_cam_lookat = [None]
        _settled_frames  = [0]
        SETTLE_FRAMES    = 30  # ~0.5 s at 60 Hz

        def _check_camera_settled() -> None:
            """Print cam_pos / cam_lookat once the viewer camera stops moving."""
            if self._camera_traj is not None:
                return  # camera-traj mode: viewer isn't user-controlled
            pos    = tuple(round(float(v), 4) for v in self._scene.viewer.camera_pos)
            lookat = tuple(round(float(v), 4) for v in self._scene.viewer.camera_lookat)
            if pos == _last_cam_pos[0] and lookat == _last_cam_lookat[0]:
                _settled_frames[0] += 1
                if _settled_frames[0] == SETTLE_FRAMES:
                    print(f"[camera] cam_pos    = {pos}")
                    print(f"[camera] cam_lookat = {lookat}")
            else:
                _last_cam_pos[0]    = pos
                _last_cam_lookat[0] = lookat
                _settled_frames[0]  = 0

        def _wait_while_paused(frame_idx: int) -> None:
            while _pause[0] and not _reset[0]:
                _do_kf_jump()
                self._scene._visualizer.update(force=True)
                _check_camera_settled()
                _update_preview(frame_idx)
                time.sleep(1 / 60)

        def _run_pass() -> None:
            """Play through start..end frames once. Returns early if reset is requested."""
            i = start
            while i < end:
                if _reset[0]:
                    return
                _current_frame[0] = i
                _wait_while_paused(i)
                if _reset[0]:
                    return
                t0 = time.perf_counter()
                self._apply(i)
                if self._camera_traj is not None:
                    cam_pos, cam_lookat = self._camera_traj.get_pose(i, self._n_frames)
                    self._scene.viewer._camera_up = np.array(self._camera_traj.get_up(i))
                    self._scene.viewer.set_camera_pose(pos=np.array(cam_pos), lookat=np.array(cam_lookat))
                self._scene._visualizer.update(force=True)
                _check_camera_settled()
                _update_preview(i)
                elapsed = time.perf_counter() - t0
                # At speed > 1: skip frames so display stays ~60 fps
                step = max(1, round(_speed[0]))
                remaining = 1.0 / 60 - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                i += step
                if i % 200 < step:
                    print(f"Frame {i}/{end}")
            print("Trajectory complete. Press BACKSPACE to replay, SPACE to pause.")

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            _reset[0] = False
            _run_pass()
            if not args.loop and not _reset[0]:
                # Trajectory done — keep viewer alive, wait for BACKSPACE to replay
                while not _reset[0]:
                    self._scene._visualizer.update(force=True)
                    _check_camera_settled()
                    if use_preview:
                        cv2.waitKey(1)
                    time.sleep(1 / 60)
