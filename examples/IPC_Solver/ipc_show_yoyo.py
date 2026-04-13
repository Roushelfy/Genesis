"""
Yoyo showcase: exploded view with orbiting camera.

Renders a close-up of the yoyo with transparent shells, animates the parts
splitting apart (exploded view) and merging back, while the camera orbits
360° around the yoyo and ends at the ipc_robot_yoyo replay camera pose.

Usage:
    python ipc_show_yoyo.py --render
    python ipc_show_yoyo.py --render --nyx
    python ipc_show_yoyo.py           # interactive viewer
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from _replay_common import CameraTrajectory, TrajectoryReplay

from _yoyo_common import (
    REPLAY_CAMERA_FOV,
    REPLAY_CAMERA_LOOKAT,
    REPLAY_CAMERA_POS,
    REPLAY_CAMERA_UP,
    YOYO_ASSETS_DIR,
)

# ── Yoyo pose (matches frame-0 of IPC simulation) ──

YOYO_CENTER = np.array([0.256, 0.008, -0.056])
YOYO_QUAT_WXYZ = np.array([0.7071068, 0.0, -0.7071068, 0.0])

# ── Camera ──

ORBIT_RADIUS = 0.12
ORBIT_ELEVATION = math.radians(25)

_REPLAY_POS = np.array(REPLAY_CAMERA_POS)
_REPLAY_LOOKAT = np.array(REPLAY_CAMERA_LOOKAT)
_REPLAY_UP = np.array(REPLAY_CAMERA_UP)

# ── Animation timeline (seconds at FPS) ──

FPS = 30
PHASE_ASSEMBLED = 2.0
PHASE_EXPLODE = 3.0
PHASE_HOLD = 2.5
PHASE_MERGE = 3.0
PHASE_PULLBACK = 3.5

# ── Explosion offsets along spin axis ──

from scipy.spatial.transform import Rotation as _Rot


def _quat_to_R(quat_wxyz):
    w, x, y, z = quat_wxyz
    return _Rot.from_quat([x, y, z, w]).as_matrix()


_SPIN_AXIS = _quat_to_R(YOYO_QUAT_WXYZ) @ np.array([0.0, 0.0, 1.0])

EXPLODE_TOP = 0.035
EXPLODE_BOTTOM = -0.035
EXPLODE_BEARING_OUTER = 0.018
EXPLODE_BEARING_SPHERE = 0.028


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _orbit_pos(angle_rad, radius, center, elevation):
    r_horiz = radius * math.cos(elevation)
    z_off = radius * math.sin(elevation)
    return center + np.array(
        [
            r_horiz * math.cos(angle_rad),
            r_horiz * math.sin(angle_rad),
            z_off,
        ]
    )


def _find_asset(name):
    p = YOYO_ASSETS_DIR / f"{name}.glb"
    if p.exists():
        return str(p)
    p = YOYO_ASSETS_DIR / f"{name}.obj"
    if p.exists():
        return str(p)
    return None


# ── Camera trajectory ──


class ShowCamera(CameraTrajectory):
    """Orbit close-up, then pull back to the replay camera pose."""

    def __init__(self, n_orbit, n_total, start_angle):
        self._n_orbit = n_orbit
        self._n_total = n_total
        self._start_angle = start_angle

    def get_pose(self, frame_idx, n_frames):
        if frame_idx < self._n_orbit:
            frac = frame_idx / self._n_orbit
            angle = self._start_angle + frac * 2 * math.pi
            cam_pos = _orbit_pos(angle, ORBIT_RADIUS, YOYO_CENTER, ORBIT_ELEVATION)
            cam_lookat = tuple(YOYO_CENTER)
        else:
            pullback_frame = frame_idx - self._n_orbit
            n_pullback = self._n_total - self._n_orbit
            t = _smooth(pullback_frame / max(n_pullback - 1, 1))
            orbit_end_pos = _orbit_pos(self._start_angle, ORBIT_RADIUS, YOYO_CENTER, ORBIT_ELEVATION)
            cam_pos = _lerp(orbit_end_pos, _REPLAY_POS, t)
            cam_lookat = tuple(_lerp(YOYO_CENTER, _REPLAY_LOOKAT, t))
        return tuple(cam_pos), cam_lookat


# ── Replay class ──


class YoyoSeparateReplay(TrajectoryReplay):
    name = "yoyo_show"
    cam_pos = tuple(
        _orbit_pos(
            math.atan2(_REPLAY_POS[1] - YOYO_CENTER[1], _REPLAY_POS[0] - YOYO_CENTER[0]) + math.pi,
            ORBIT_RADIUS,
            YOYO_CENTER,
            ORBIT_ELEVATION,
        )
    )
    cam_lookat = tuple(YOYO_CENTER)
    cam_fov = REPLAY_CAMERA_FOV

    def load_trajectory(self):
        n_assembled = int(PHASE_ASSEMBLED * FPS)
        n_explode = int(PHASE_EXPLODE * FPS)
        n_hold = int(PHASE_HOLD * FPS)
        n_merge = int(PHASE_MERGE * FPS)
        n_pullback = int(PHASE_PULLBACK * FPS)
        self._n_orbit = n_assembled + n_explode + n_hold + n_merge
        n_total = self._n_orbit + n_pullback

        # Precompute explosion parameter per frame
        self._explode_t = np.zeros(n_total, dtype=np.float32)
        for frame in range(n_total):
            if frame < n_assembled:
                self._explode_t[frame] = 0.0
            elif frame < n_assembled + n_explode:
                self._explode_t[frame] = _smooth((frame - n_assembled) / n_explode)
            elif frame < n_assembled + n_explode + n_hold:
                self._explode_t[frame] = 1.0
            elif frame < self._n_orbit:
                self._explode_t[frame] = 1.0 - _smooth((frame - n_assembled - n_explode - n_hold) / n_merge)
            else:
                self._explode_t[frame] = 0.0

        self.fps = FPS

        # Camera trajectory
        replay_angle = math.atan2(
            _REPLAY_POS[1] - YOYO_CENTER[1],
            _REPLAY_POS[0] - YOYO_CENTER[0],
        )
        self._start_angle = replay_angle + math.pi
        if not self.args.camera_traj:
            self.args.camera_traj = "_show_default"

        return n_total

    def build_scene(self, scene):
        import genesis as gs

        def _add(name, surface, asset_name=None):
            return scene.add_entity(
                morph=gs.morphs.Mesh(file=_find_asset(asset_name or name), fixed=True, collision=False),
                material=gs.materials.Rigid(),
                surface=surface,
                name=name,
            )

        self._top_shell = _add("top_shell", None, "yoyo-top_shell")
        self._bottom_shell = _add("bottom_shell", None, "yoyo-bottom_shell")
        self._top_ring = _add("top_ring", None, "yoyo-top_ring")
        self._bottom_ring = _add("bottom_ring", None, "yoyo-bottom_ring")
        self._axle = _add("axle", None, "yoyo-axle")
        self._hub = _add("hub", None, "yoyo-hub")
        self._bearing_outer = _add("bearing_outer", None)
        self._bearing_spheres = []
        for i in range(8):
            if _find_asset(f"bearing_sphere_{i}") is not None:
                self._bearing_spheres.append(_add(f"bearing_sphere_{i}", None))

        # No robot, no trajectory data
        self._robot = None
        self._joint_qpos = None
        self._rigid_entities = {}
        self._rigid_data = {}
        self._fem_entities = {}
        self._fem_data = {}

    def apply_frame(self, scene, frame_idx):
        t = self._explode_t[frame_idx]

        def _set_part(ent, offset):
            ent.set_pos(YOYO_CENTER + _SPIN_AXIS * offset)
            ent.set_quat(YOYO_QUAT_WXYZ)

        _set_part(self._top_shell, EXPLODE_TOP * t)
        _set_part(self._bottom_shell, EXPLODE_BOTTOM * t)
        _set_part(self._top_ring, EXPLODE_TOP * 0.5 * t)
        _set_part(self._bottom_ring, EXPLODE_BOTTOM * 0.5 * t)
        _set_part(self._axle, 0.0)
        _set_part(self._hub, 0.0)
        _set_part(self._bearing_outer, EXPLODE_BEARING_OUTER * t)
        for ent in self._bearing_spheres:
            _set_part(ent, EXPLODE_BEARING_SPHERE * t)

    def make_camera_traj(self, name):
        if name == "_show_default":
            return ShowCamera(self._n_orbit, self._n_frames, self._start_angle)
        return super().make_camera_traj(name)


if __name__ == "__main__":
    YoyoSeparateReplay().run()
