"""
Replay the v4 yoyo trajectory with full preview and render support.

Usage
-----
    python examples/IPC_Solver/replay_yoyo_v4_traj.py [OPTIONS]

Interactive viewer (default)
    --loop                  Loop replay continuously
    --speed FLOAT           Playback speed multiplier (default: 1.0)
                            Keyboard: , = half speed  . = double speed
    --start-frame N         Start from frame N (BACKSPACE also resets here)
    --end-frame N           Stop at frame N exclusive (default: last frame)
    --camera-traj MODE      Drive camera automatically: surround | full | orbit | custom
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
    --trajectory KEY        Trajectory key (default: v4; see _yoyo_common.TRAJECTORIES)
    --seq-dir PATH          Override sequence directory
    --subsample N           Take every Nth frame (default: 1 = all frames)
    --smooth-joints SIGMA   Gaussian smoothing sigma on joint angles (default: 0 = off)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Default to v4 sequence
_V4_SEQ = Path(__file__).resolve().parents[2] / "DemoAssets" / "yoyo" / "v4" / "seq"
if "--seq-dir" not in sys.argv and "--trajectory" not in sys.argv:
    sys.argv += ["--seq-dir", str(_V4_SEQ), "--trajectory", "v4"]

from _replay_common import CameraTrajectory, FullViewCamera, SurroundCamera  # noqa: E402
from _yoyo_common import (  # noqa: E402
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
)
from replay_yoyo_traj import (  # noqa: E402
    ORBIT_ANGLE_END,
    ORBIT_ANGLE_START,
    ORBIT_CENTER,
    ORBIT_HEIGHT,
    ORBIT_RADIUS,
    YoyoReplay,
    _smooth,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Duration of the camera blend from the pulled-back pose into the orbit path
TRANSITION_SECONDS = 1.0


# ── Camera ─────────────────────────────────────────────────────────────────────


def _orbit_pose(frame_idx, n_frames):
    frac = frame_idx / max(n_frames - 1, 1)
    angle = ORBIT_ANGLE_START + (ORBIT_ANGLE_END - ORBIT_ANGLE_START) * _smooth(frac)
    cam_pos = np.array(
        [
            ORBIT_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
            ORBIT_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
            ORBIT_HEIGHT,
        ]
    )
    return cam_pos, np.array(ORBIT_CENTER, dtype=np.float64)


class OrbitWithIntroCamera(CameraTrajectory):
    """Smooth blend from REPLAY_CAMERA_POS into the half-circle orbit over TRANSITION_SECONDS."""

    def __init__(self, transition_n: int):
        self._transition_n = max(1, transition_n)
        self._pull_pos = np.array(REPLAY_CAMERA_POS, dtype=np.float64)
        self._pull_lookat = np.array(REPLAY_CAMERA_LOOKAT, dtype=np.float64)

    def get_pose(self, frame_idx, n_frames):
        orbit_pos, orbit_lookat = _orbit_pose(frame_idx, n_frames)
        if frame_idx < self._transition_n:
            t = _smooth(frame_idx / max(self._transition_n - 1, 1))
            cam_pos = self._pull_pos * (1 - t) + orbit_pos * t
            cam_lookat = self._pull_lookat * (1 - t) + orbit_lookat * t
        else:
            cam_pos, cam_lookat = orbit_pos, orbit_lookat
        return tuple(cam_pos), tuple(cam_lookat)


# ── Replay class ───────────────────────────────────────────────────────────────


class YoyoV4Replay(YoyoReplay):
    name = "yoyo_v4"
    cam_pos = REPLAY_CAMERA_POS
    cam_lookat = REPLAY_CAMERA_LOOKAT
    cam_fov = REPLAY_CAMERA_FOV

    # Shell parts that receive the logo texture in build_scene
    _SHELL_PARTS = {"yoyo-top_shell", "yoyo-bottom_shell", "yoyo-top_ring", "yoyo-bottom_ring"}
    _YOYO_ASSETS = _REPO_ROOT / "DemoAssets" / "yoyo" / "v3"
    _LOGO_IMG = _REPO_ROOT / "DemoAssets" / "yoyo" / "logo_centered.png"

    # ── Renderer ───────────────────────────────────────────────────────────────

    def make_renderer(self):
        # Configure Luisa sphere lights here to match nyx_lights().
        # Falls back to _yoyo_common.make_raytracer() until overridden.
        return super().make_renderer()

    def nyx_lights(self):
        # Configure Nyx point lights here.
        return []

    def nyx_light_field(self):
        return None

    # ── Camera ─────────────────────────────────────────────────────────────────

    def make_camera_traj(self, name):
        presets = {
            "surround": lambda: SurroundCamera(
                center=tuple(ORBIT_CENTER),
                radius=ORBIT_RADIUS,
                height=ORBIT_HEIGHT,
                angle_start=int(math.degrees(ORBIT_ANGLE_START)),
                angle_end=int(math.degrees(ORBIT_ANGLE_END)),
            ),
            "full": lambda: FullViewCamera(
                pos=self.cam_pos,
                lookat=self.cam_lookat,
            ),
            "_yoyo_default": lambda: OrbitWithIntroCamera(
                transition_n=int(TRANSITION_SECONDS * self.fps),
            ),
        }
        if name in presets:
            return presets[name]()
        return super().make_camera_traj(name)

    def custom_camera_keyframes(self):
        # Capture poses with K in interactive mode, paste here.
        # Each entry: (frame, pos, lookat[, up[, ease_in[, ease_out]]])
        return []

    # ── Scene ──────────────────────────────────────────────────────────────────

    def build_scene(self, scene):
        import genesis as gs

        original_add = scene.add_entity

        def patched_add(*args, **kwargs):
            name = kwargs.get("name", "")
            if name in self._SHELL_PARTS:
                logo_glb = self._YOYO_ASSETS / f"{name}_logo.glb"
                if logo_glb.exists():
                    if args:
                        args = (gs.morphs.Mesh(file=str(logo_glb), fixed=True, collision=False),) + args[1:]
                    else:
                        kwargs["morph"] = gs.morphs.Mesh(file=str(logo_glb), fixed=True, collision=False)
                    kwargs["surface"] = gs.surfaces.BSDF(
                        diffuse_texture=gs.textures.ImageTexture(image_path=str(self._LOGO_IMG)),
                        metallic=0.3,
                        roughness=0.4,
                    )
            return original_add(*args, **kwargs)

        scene.add_entity = patched_add
        try:
            super().build_scene(scene)
        finally:
            scene.add_entity = original_add

    # ── Args / trajectory ──────────────────────────────────────────────────────

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            "--subsample",
            type=int,
            default=1,
            help="Take every Nth frame from the trajectory data (default: 1 = all frames)",
        )
        parser.add_argument(
            "--smooth-joints",
            type=float,
            default=0,
            help="Gaussian smoothing sigma (in frames) for joint angles. 0 = off.",
        )

    def load_trajectory(self):
        n_frames = super().load_trajectory()
        stride = self.args.subsample
        sigma = self.args.smooth_joints

        # Smooth at full temporal resolution, then subsample — no aliasing.
        if sigma > 0 and self._raw_joint_data is not None:
            from scipy.ndimage import gaussian_filter1d

            print(f"[smooth-joints] Gaussian sigma={sigma} on {self._raw_joint_data.shape}")
            self._raw_joint_data = gaussian_filter1d(
                self._raw_joint_data.astype(np.float64), sigma=sigma, axis=0
            ).astype(self._raw_joint_data.dtype)

        if stride > 1:
            for name in self._rigid_data:
                self._rigid_data[name] = self._rigid_data[name][::stride]
            for name in self._fem_data:
                self._fem_data[name] = self._fem_data[name][::stride]
            if self._raw_joint_data is not None:
                self._raw_joint_data = self._raw_joint_data[::stride]
            n_frames = min(len(v) for v in self._rigid_data.values()) if self._rigid_data else n_frames // stride
            print(f"[subsample] stride={stride} → {n_frames} frames")
        return n_frames


if __name__ == "__main__":
    YoyoV4Replay().run()
