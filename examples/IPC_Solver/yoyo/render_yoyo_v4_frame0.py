"""
Multi-phase showcase:
  1. Close-up of assembled yoyo on a pure black background (robot hidden).
  2. Yoyo explodes to the right (camera static).
  3. Yoyo merges back to a whole.
  4. Camera pulls back to the wide replay pose (robot still hidden).
  5. Robot gradually appears at its v4 frame-0 pose.

Usage:
    python examples/IPC_Solver/render_yoyo_v4_frame0.py
    python examples/IPC_Solver/render_yoyo_v4_frame0.py --render --nyx
"""

import sys
from pathlib import Path

# Add the IPC_Solver parent dir to sys.path so _replay_common / _yoyo_common /
# replay_yoyo_traj are importable when this script is invoked from yoyo/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.transform import Rotation

# Default to the v4 sequence + black background
_V4_SEQ = Path(__file__).resolve().parents[3] / "DemoAssets" / "yoyo" / "v4" / "seq"
if "--seq-dir" not in sys.argv and "--trajectory" not in sys.argv:
    sys.argv += ["--seq-dir", str(_V4_SEQ), "--trajectory", "v4"]

import math  # noqa: E402

from _replay_common import CameraTrajectory, TrajectoryReplay  # noqa: E402
from _yoyo_common import (  # noqa: E402
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
)
from replay_yoyo_traj import (  # noqa: E402
    ORBIT_ANGLE_START,
    ORBIT_CENTER,
    ORBIT_HEIGHT,
    ORBIT_RADIUS,
    YoyoReplay,
)

# Orbit camera pose at its first frame (target for the showcase pullback,
# so the pullback ends exactly where the replay's orbit begins → continuous motion).
_ORBIT_START_POS = np.array([
    ORBIT_CENTER[0] + ORBIT_RADIUS * math.cos(ORBIT_ANGLE_START),
    ORBIT_CENTER[1] + ORBIT_RADIUS * math.sin(ORBIT_ANGLE_START),
    ORBIT_HEIGHT,
], dtype=np.float64)
_ORBIT_START_LOOKAT = np.array(ORBIT_CENTER, dtype=np.float64)


# ── Explode + reveal animation timing ──

FPS = 30
PHASE_HOLD_EXPLODED = 2.0     # start already decomposed, hold for viewing
PHASE_MERGE_PULLBACK = 3.5    # merge + camera pullback + robot reveal simultaneously

EXPLODE_TOP = 0.035
EXPLODE_BEARING_OUTER = 0.018
EXPLODE_BEARING_SPHERE = 0.028

# Close-up camera: 1/3 further than before (0.15 * 4/3)
CLOSEUP_DIST = 0.20

# Robot's "behind" offset: along the camera's view direction (into the screen)
# computed from the replay camera at module import time below.

# Hide the robot far below the world so it never appears in frame.
HIDE_OFFSET = np.array([0.0, 0.0, -50.0])


# Order entities are appended inside replay_yoyo_traj._add_ball_part
_BALL_PART_ORDER = [
    "yoyo-top_shell",
    "yoyo-bottom_shell",
    "yoyo-top_ring",
    "yoyo-bottom_ring",
    "yoyo-axle",
    "yoyo-hub",
]


def _smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _ease_out_explosive(t: float) -> float:
    """Fast initial burst, slow deceleration — feels like an explosion."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 4


def _ease_in_snap(t: float) -> float:
    """Slow start, fast settle — feels like a sudden snap-back."""
    t = max(0.0, min(1.0, t))
    return t ** 4


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _quat_wxyz_to_rotmat(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


class V4ExplodeRevealCamera(CameraTrajectory):
    """Static close-up → pull-back to wide replay pose."""

    def __init__(self, replay: "YoyoV4Showcase"):
        self._r = replay

    def get_pose(self, frame_idx, n_frames):
        r = self._r

        # Close-up pose (rotated 30° CCW from the replay camera direction)
        replay_pos = np.array(REPLAY_CAMERA_POS, dtype=np.float64)
        replay_lookat = np.array(REPLAY_CAMERA_LOOKAT, dtype=np.float64)
        view_dir = replay_pos - replay_lookat
        view_dir /= np.linalg.norm(view_dir)
        rot_left = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        closeup_dir = rot_left @ view_dir
        closeup_pos = r._yoyo_center + closeup_dir * CLOSEUP_DIST
        closeup_lookat = r._yoyo_center

        # Pullback target = the orbit's frame-0 pose, so the pullback leads
        # directly into the replay's circular motion without any intermediate
        # waypoint (no pause at REPLAY_CAMERA_POS).
        pull_target_pos = _ORBIT_START_POS
        pull_target_lookat = _ORBIT_START_LOOKAT

        # merge_pullback starts simultaneously with merge
        mp_start = r._phase_start["merge_pullback"]
        mp_end = mp_start + r._phase_counts["merge_pullback"]
        if frame_idx < mp_start:
            cam_pos, cam_lookat = closeup_pos, closeup_lookat
        elif frame_idx < mp_end:
            f = frame_idx - mp_start
            n = r._phase_counts["merge_pullback"]
            t = _smooth(f / max(n - 1, 1))
            cam_pos = _lerp(closeup_pos, pull_target_pos, t)
            cam_lookat = _lerp(closeup_lookat, pull_target_lookat, t)
        else:
            cam_pos, cam_lookat = pull_target_pos, pull_target_lookat

        return tuple(cam_pos), tuple(cam_lookat)


# G-Warm light gray 03 from DemoAssets/yoyo/color.png
# sRGB ≈ (195, 190, 186) → linear ≈ (0.55, 0.52, 0.50)
SHELL_COLOR = (0.55, 0.52, 0.50, 1.0)
SHELL_PARTS = {"yoyo-top_shell", "yoyo-bottom_shell", "yoyo-top_ring", "yoyo-bottom_ring"}

_REPO_ROOT_SHOWCASE = Path(__file__).resolve().parents[3]


class YoyoV4Showcase(YoyoReplay):
    _base_name = "yoyo_v4_showcase"

    @property
    def name(self):
        mode = getattr(self.args, "robot_pass", "auto") if hasattr(self, "args") else "auto"
        if mode in ("hidden", "visible"):
            return f"{self._base_name}_{mode}"
        return self._base_name

    # Same lighting as orbit BulletTimeRenderer
    def nyx_lights(self):
        return [
            {"type": "point", "pos": (0.85, 1.25, 2.45),
             "color": (1.0, 0.97, 0.92), "intensity": 20.0, "shadow": True},
            {"type": "point", "pos": (0.6, -1.7, 4.3),
             "color": (0.48, 0.52, 0.6), "intensity": 1.0, "shadow": False},
            {"type": "point", "pos": (-0.8, -3.16, 0.5),
             "color": (0.8, 0.88, 1.0), "intensity": 100.0, "shadow": True},
            {"type": "point", "pos": (0.85, 1.25, 0.0),
             "color": (1.0, 0.97, 0.92), "intensity": 20.0, "shadow": True},
        ]

    def nyx_light_field(self):
        return None

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            "--robot-pass",
            type=str,
            default="auto",
            choices=["auto", "hidden", "visible"],
            help="Robot visibility mode: auto (teleport at pullback), hidden (always), visible (always).",
        )
    # Class-level values used when the sensor is first created; the camera
    # trajectory overrides these every frame.
    cam_pos = tuple(np.array(REPLAY_CAMERA_POS))
    cam_lookat = tuple(np.array(REPLAY_CAMERA_LOOKAT))
    cam_fov = REPLAY_CAMERA_FOV

    def load_trajectory(self):
        super().load_trajectory()

        # The "hidden" pass is the explode-view pass (no robot). Add extra
        # directional lights from top / back / left / front so the yoyo
        # is dramatically lit for the explosion. The "visible" pass keeps
        # the normal lighting (which is what gets composited into the
        # final video during the pullback phase onwards).
        if getattr(self.args, "robot_pass", "auto") == "hidden":
            self._extra_nyx_lights = [
                # Top
                {"type": "directional", "dir": (0.0, 0.0, -1.0),
                 "color": (1.0, 1.0, 1.0), "intensity": 3.0, "shadow": False},
                # Back (away from camera → facing camera)
                {"type": "directional", "dir": (0.0, 1.0, 0.0),
                 "color": (1.0, 1.0, 1.0), "intensity": 2.0, "shadow": False},
                # Left
                {"type": "directional", "dir": (1.0, 0.0, 0.0),
                 "color": (1.0, 1.0, 1.0), "intensity": 2.0, "shadow": False},
                # Front (behind camera → facing into scene)
                {"type": "directional", "dir": (0.0, -1.0, 0.0),
                 "color": (1.0, 1.0, 1.0), "intensity": 2.0, "shadow": False},
            ]

        # Phase frame counts (in output frames).
        # Starts DECOMPOSED → merge+pullback simultaneously.
        phases = ["hold_exploded", "merge_pullback"]
        durations = [PHASE_HOLD_EXPLODED, PHASE_MERGE_PULLBACK]
        self._phase_counts = {p: int(d * FPS) for p, d in zip(phases, durations)}
        self._phase_start = {}
        acc = 0
        for p in phases:
            self._phase_start[p] = acc
            acc += self._phase_counts[p]
        n_total = acc

        # Explode parameter per frame: 1.0 during hold, merges to 0.0
        # during the first ~40% of merge_pullback phase.
        self._explode_t = np.zeros(n_total, dtype=np.float32)
        mp_start = self._phase_start["merge_pullback"]
        mp_n = self._phase_counts["merge_pullback"]
        merge_fraction = 1.0  # merge spans full phase, synced with camera pullback
        merge_frames = int(mp_n * merge_fraction)
        for f in range(n_total):
            if f < mp_start:
                self._explode_t[f] = 1.0
            elif f < mp_start + merge_frames:
                self._explode_t[f] = 1.0 - _smooth((f - mp_start) / max(merge_frames, 1))
            else:
                self._explode_t[f] = 0.0

        # Yoyo frame-0 center + spin axis (for explode direction and camera target)
        pose0 = self._rigid_data["yoyo_ball"][0]
        self._yoyo_center = pose0[:3].astype(np.float64)
        R = _quat_wxyz_to_rotmat(pose0[3:])
        self._spin_axis = R @ np.array([0.0, 0.0, 1.0])

        # Use our custom camera trajectory
        self.args.camera_traj = "_v4_showcase"
        self.fps = FPS
        return n_total

    def make_camera_traj(self, name):
        if name == "_v4_showcase":
            return V4ExplodeRevealCamera(self)
        return super().make_camera_traj(name)

    # Bearing sphere palette: alternating blue and cyan (metallic, opaque)
    _BEARING_PALETTE = [
        (0.40, 0.50, 1.00, 1.0),  # blue
        (0.30, 0.80, 1.00, 1.0),  # cyan
    ]

    def build_scene(self, scene):
        import genesis as gs

        # Temporarily wrap scene.add_entity to inject distinct colors for the
        # 8 bearing spheres as they are created by YoyoReplay.build_scene.
        original_add_entity = scene.add_entity

        _YOYO_ASSETS = _REPO_ROOT_SHOWCASE / "DemoAssets" / "yoyo" / "v3"

        def colored_add_entity(*args, **kwargs):
            name = kwargs.get("name", "")
            # Shells use logo-mesh GLB but a plain color BSDF (no logo PNG);
            # rings use the smoothed GLB and keep their built-in GLB material.
            if name in SHELL_PARTS:
                is_ring = name in ("yoyo-top_ring", "yoyo-bottom_ring")
                if is_ring:
                    glb_file = _YOYO_ASSETS / f"{name}_smooth.glb"
                else:
                    glb_file = _YOYO_ASSETS / f"{name}_logo.glb"
                if glb_file.exists():
                    if args:
                        args = (gs.morphs.Mesh(file=str(glb_file), fixed=True, collision=False),) + args[1:]
                    else:
                        kwargs["morph"] = gs.morphs.Mesh(file=str(glb_file), fixed=True, collision=False)
                    if not is_ring:
                        # G-Warm light gray 03 (sRGB ~195 → linear ~0.55)
                        kwargs["surface"] = gs.surfaces.BSDF(
                            color=(0.55, 0.52, 0.50, 1.0),
                            metallic=0.3,
                            roughness=0.4,
                        )
            # Bearing spheres: alternating blue/cyan, opaque
            if name.startswith("bearing_sphere_"):
                try:
                    idx = int(name.split("_")[-1])
                    color = self._BEARING_PALETTE[idx % len(self._BEARING_PALETTE)]
                    kwargs["surface"] = gs.surfaces.BSDF(
                        color=color, metallic=0.1, roughness=0.05,
                    )
                except (ValueError, IndexError):
                    pass
            return original_add_entity(*args, **kwargs)

        scene.add_entity = colored_add_entity
        try:
            super().build_scene(scene)
        finally:
            scene.add_entity = original_add_entity

    def apply_frame(self, scene, frame_idx):
        # Freeze everything at v4 frame 0
        TrajectoryReplay.apply_frame(self, scene, 0)

        # Capture the robot's frame-0 base pos once (before we move it away)
        if not hasattr(self, "_robot_base_pos"):
            base_pos = self._robot.get_pos()
            if base_pos.ndim > 1:
                base_pos = base_pos[0]
            self._robot_base_pos = base_pos.cpu().numpy().astype(np.float64).copy()

        # Robot pass mode
        mode = getattr(self.args, "robot_pass", "auto")
        pullback_start = self._phase_start["merge_pullback"]
        hidden_pos = self._robot_base_pos + HIDE_OFFSET
        if mode == "hidden":
            self._robot.set_pos(hidden_pos)
        elif mode == "visible":
            self._robot.set_pos(self._robot_base_pos)
        else:
            if frame_idx < pullback_start:
                self._robot.set_pos(hidden_pos)
            else:
                self._robot.set_pos(self._robot_base_pos)

        # Apply the explode animation on top of the frame-0 yoyo positions
        t_explode = float(self._explode_t[frame_idx])
        if t_explode <= 0.0:
            return

        pose0 = self._rigid_data["yoyo_ball"][0]
        ball_center = pose0[:3].astype(np.float64)
        spin_axis = self._spin_axis

        # Slow rotation around spin_axis: one revolution every 2 seconds.
        # Every component's quaternion is composed with this rotation so the
        # mesh offsets orbit the axis together (true rigid rotation).
        rotation_period_sec = 2.0
        sphere_angle = (frame_idx / FPS) * (2.0 * math.pi / rotation_period_sec)
        spin_rot = Rotation.from_rotvec(sphere_angle * spin_axis)

        def compose_spin(q_wxyz):
            q0 = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            xyzw = (spin_rot * q0).as_quat()
            return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        # Explode from CENTER: axle/hub stay static, shells move outward both sides.
        d = EXPLODE_TOP * t_explode  # positive offset magnitude
        ball_offsets = {
            "yoyo-top_shell": 1.5 * d,       # rightmost
            "yoyo-top_ring": 0.7 * d,         # right
            "yoyo-axle": 0.0,                  # center (static)
            "yoyo-hub": 0.0,                   # center (static)
            "yoyo-bottom_ring": -0.7 * d,     # left
            "yoyo-bottom_shell": -1.5 * d,    # leftmost
        }
        new_ball_quat = compose_spin(pose0[3:])
        ball_entities = self._rigid_entities.get("yoyo_ball", [])
        for idx, ent in enumerate(ball_entities):
            if idx >= len(_BALL_PART_ORDER):
                break
            part_name = _BALL_PART_ORDER[idx]
            offset = ball_offsets.get(part_name, 0.0)
            ent.set_pos(ball_center + spin_axis * offset)
            ent.set_quat(new_ball_quat)

        # Bearing outer + spheres: offset to one side (positive spin_axis)
        bearing_outer_step = 0.5 * d
        bearing_sphere_step = 1.2 * d  # ~3cm closer to center (was 2.0)

        bo_entities = self._rigid_entities.get("bearing_outer", [])
        bo_pose = self._rigid_data.get("bearing_outer")
        if bo_entities and bo_pose is not None:
            bo_p0 = bo_pose[0]
            new_pos = bo_p0[:3].astype(np.float64) + spin_axis * bearing_outer_step
            for ent in bo_entities:
                ent.set_pos(new_pos)
                ent.set_quat(compose_spin(bo_p0[3:]))

        # All 8 bearing spheres share the same origin pose at frame 0 — each
        # ball's distinct geometry is baked as an offset in its mesh file.
        for i in range(8):
            name = f"bearing_sphere_{i}"
            bs_entities = self._rigid_entities.get(name, [])
            bs_pose = self._rigid_data.get(name)
            if not bs_entities or bs_pose is None:
                continue
            bs_p0 = bs_pose[0]
            new_pos = bs_p0[:3].astype(np.float64) + spin_axis * bearing_sphere_step
            new_quat = compose_spin(bs_p0[3:])
            for ent in bs_entities:
                ent.set_pos(new_pos)
                ent.set_quat(new_quat)


if __name__ == "__main__":
    YoyoV4Showcase().run()
