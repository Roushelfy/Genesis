"""
Render the full v4 yoyo trajectory with a camera that smoothly transitions
from the showcase end pose into the standard orbit path.

Phase naming for the final composed video:
    1. closeup          — static close-up of assembled yoyo (from showcase)
    2. explode          — yoyo explodes to the right
    3. hold             — exploded state held
    4. merge            — yoyo merges back into a whole
    5. pullback         — camera pulls back, robot fades in  (last but two)
    6. orbit_transition — 0.5s smooth camera transfer          (last but one)
    7. orbit_replay     — yoyo released, camera orbits         (last)

Phases 1–5 live in ``render_yoyo_v4_frame0.py`` (the showcase passes).
Phases 6–7 are produced here: the first 0.5s (30 frames at 60fps) interpolates
from REPLAY_CAMERA_POS/LOOKAT to the first YoyoOrbitCamera pose, then the
standard orbit continues for the rest of the trajectory.

Usage:
    python examples/IPC_Solver/render_yoyo_v4_replay.py --render --nyx
"""

import math
import sys
from pathlib import Path

import numpy as np

# Default trajectory is v4. Pass --trajectory long_sleep (or any other key in
# _yoyo_common.TRAJECTORIES) on the command line to override. The default
# renderer is Luisa; pass --nyx to switch.
_V4_SEQ = Path(__file__).resolve().parents[2] / "DemoAssets" / "yoyo" / "v4" / "seq"
if "--seq-dir" not in sys.argv and "--trajectory" not in sys.argv:
    sys.argv += ["--seq-dir", str(_V4_SEQ), "--trajectory", "v4"]

from _replay_common import CameraTrajectory  # noqa: E402
from _yoyo_common import (  # noqa: E402
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


# 1.0 s transition at the replay's native fps (replay runs at 60fps)
TRANSITION_SECONDS = 1.0


def _orbit_pose(frame_idx, n_frames):
    """YoyoOrbitCamera pose formula copied from replay_yoyo_traj.py."""
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
    """
    Replay camera used for phases 6–7:
      - phase 6 ``orbit_transition`` (1.0s): smooth weighted blend between
        the *pulled-away* pose (REPLAY_CAMERA_POS, held from the end of the
        showcase pullback) and the *original circular motion* (orbit pose
        at the current frame index). The orbit position is NOT frozen at
        frame 0 — the circular motion is already happening during the
        blend, so the two are cross-faded in time and space.
      - phase 7 ``orbit_replay``: the normal half-circle orbit for the rest
        of the trajectory.
    """

    def __init__(self, transition_n: int):
        self._transition_n = max(1, transition_n)
        self._pull_pos = np.array(REPLAY_CAMERA_POS, dtype=np.float64)
        self._pull_lookat = np.array(REPLAY_CAMERA_LOOKAT, dtype=np.float64)

    def get_pose(self, frame_idx, n_frames):
        # "Original circular motion" pose at this frame
        orbit_pos, orbit_lookat = _orbit_pose(frame_idx, n_frames)
        if frame_idx < self._transition_n:
            # Weighted blend — 0 at start (pure pull-away), 1 at end (pure orbit)
            t = _smooth(frame_idx / max(self._transition_n - 1, 1))
            cam_pos = self._pull_pos * (1 - t) + orbit_pos * t
            cam_lookat = self._pull_lookat * (1 - t) + orbit_lookat * t
        else:
            cam_pos, cam_lookat = orbit_pos, orbit_lookat
        return tuple(cam_pos), tuple(cam_lookat)


_REPO_ROOT = Path(__file__).resolve().parents[2]


class YoyoV4StaticReplay(YoyoReplay):
    name = "yoyo_v4_replay"

    # Luisa renderer lighting from replay_trashbag_sharpa_traj.py, applied to Nyx.
    # 3 sphere lights (key / fill / rim) + dark grey (0.01) background.
    def nyx_lights(self):
        return [
            # Key light: above-left, warm, casting shadows
            {"type": "point", "pos": (0.85, 1.25, 2.45), "color": (1.0, 0.97, 0.92), "intensity": 50.0, "shadow": True},
            # Fill light: right side, cooler, softer
            {"type": "point", "pos": (0.6, -1.7, 4.3), "color": (0.48, 0.52, 0.6), "intensity": 1.0, "shadow": False},
            # Rim light: behind the scene, cool, hard
            {"type": "point", "pos": (-0.8, -3.16, 0.5), "color": (0.8, 0.88, 1.0), "intensity": 150.0, "shadow": True},
        ]

    def nyx_light_field(self):
        return None

    # Shell logo texture (same as showcase)
    _SHELL_PARTS = {"yoyo-top_shell", "yoyo-bottom_shell", "yoyo-top_ring", "yoyo-bottom_ring"}
    _YOYO_ASSETS = _REPO_ROOT / "DemoAssets" / "yoyo" / "v3"
    _LOGO_IMG = _REPO_ROOT / "DemoAssets" / "yoyo" / "logo_centered.png"

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
            elif name == "robot":
                kwargs["surface"] = {
                    "paint_white_glossy": gs.surfaces.BSDF(
                        color=(0.74, 0.74, 0.74),
                        roughness=0.25,
                        metallic=0.25,
                    ),
                    "plastic_black_rough": gs.surfaces.BSDF(
                        color=(0.02, 0.02, 0.03),
                        roughness=0.35,
                        metallic=0.0,
                        ior=1.45,
                    ),
                }
            return original_add(*args, **kwargs)

        scene.add_entity = patched_add
        try:
            super().build_scene(scene)
        finally:
            scene.add_entity = original_add

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

        # Smooth joint angles BEFORE subsampling (smooth at full temporal res,
        # then subsample the smooth curve → no aliasing).
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
    YoyoV4StaticReplay().run()
