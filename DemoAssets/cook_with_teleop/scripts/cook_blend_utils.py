"""Cooking-specific blend utilities.

Wraps the core :class:`BlendKeyframe` / :class:`BlendSchedule` from the yoyo
``blend_utils`` module and adds helpers for the cooking trajectory format
(JSON with per-frame ``pan`` / ``spatula`` pos+quat).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

# Import core blend classes from yoyo blend_utils (format-agnostic)
_YOYO_SCRIPTS = str(Path(__file__).resolve().parents[2] / "yoyo" / "scripts")
if _YOYO_SCRIPTS not in sys.path:
    sys.path.insert(0, _YOYO_SCRIPTS)

from blend_utils import (  # noqa: E402
    BlendKeyframe,
    BlendSchedule,
    load_blend_keyframes,
    save_blend_keyframes,
)

__all__ = [
    "BlendKeyframe",
    "BlendSchedule",
    "save_blend_keyframes",
    "load_blend_keyframes",
    "ENTITIES",
    "CHANNEL_NAMES",
    "frame_to_channels",
    "channels_to_frame",
    "renormalize_quat_channels",
    "bake_cook_trajectory",
    "load_cook_trajectory",
    "save_cook_trajectory",
    "quat_to_euler_deg",
    "euler_deg_to_quat",
]

ENTITIES = ("pan", "spatula")

POS_AXES = ("x", "y", "z")
QUAT_AXES = ("w", "x", "y", "z")

CHANNEL_NAMES: list[str] = []
for _ent in ENTITIES:
    for _ax in POS_AXES:
        CHANNEL_NAMES.append(f"{_ent}.pos.{_ax}")
    for _ax in QUAT_AXES:
        CHANNEL_NAMES.append(f"{_ent}.quat.{_ax}")


# ------------------------------------------------------------------
# Euler <-> Quaternion (intrinsic XYZ, degrees)
# ------------------------------------------------------------------

def quat_to_euler_deg(quat: list[float] | tuple[float, ...]) -> list[float]:
    """Convert [w, x, y, z] quaternion to [rx, ry, rz] Euler XYZ in degrees."""
    w, x, y, z = quat
    # Roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    ry = math.asin(sinp)
    # Yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def euler_deg_to_quat(euler_deg: list[float] | tuple[float, ...]) -> list[float]:
    """Convert [rx, ry, rz] Euler XYZ in degrees to [w, x, y, z] quaternion."""
    rx, ry, rz = [math.radians(a) for a in euler_deg]
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return [w, x, y, z]


# ------------------------------------------------------------------
# Frame <-> channel conversion
# ------------------------------------------------------------------

def frame_to_channels(frame: dict) -> dict[str, float]:
    """Convert a trajectory frame's pan/spatula data to a flat channel dict.

    Returns up to 14 channels (7 per entity).  Entities missing from the
    frame are silently skipped.
    """
    channels: dict[str, float] = {}
    for ent in ENTITIES:
        data = frame.get(ent)
        if data is None:
            continue
        pos = data.get("pos", [0.0, 0.0, 0.0])
        quat = data.get("quat", [1.0, 0.0, 0.0, 0.0])
        for i, ax in enumerate(POS_AXES):
            channels[f"{ent}.pos.{ax}"] = float(pos[i])
        for i, ax in enumerate(QUAT_AXES):
            channels[f"{ent}.quat.{ax}"] = float(quat[i])
    return channels


def channels_to_frame(channels: dict[str, float], template: dict) -> dict:
    """Merge channel values back into a full frame dict.

    *template* is deep-copied; only channels present in *channels* are
    overwritten.
    """
    out = copy.deepcopy(template)
    for ent in ENTITIES:
        if ent not in out:
            out[ent] = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
        for i, ax in enumerate(POS_AXES):
            key = f"{ent}.pos.{ax}"
            if key in channels:
                out[ent]["pos"][i] = channels[key]
        for i, ax in enumerate(QUAT_AXES):
            key = f"{ent}.quat.{ax}"
            if key in channels:
                out[ent]["quat"][i] = channels[key]
    return out


def renormalize_quat_channels(channels: dict[str, float], entity: str) -> None:
    """In-place normalize the quaternion channels for *entity* (NLerp)."""
    keys = [f"{entity}.quat.{ax}" for ax in QUAT_AXES]
    vals = [channels.get(k) for k in keys]
    if any(v is None for v in vals):
        return
    norm = math.sqrt(sum(v * v for v in vals))  # type: ignore[union-attr]
    if norm < 1e-12:
        return
    for k, v in zip(keys, vals):
        channels[k] = v / norm  # type: ignore[operator]


# ------------------------------------------------------------------
# Trajectory I/O
# ------------------------------------------------------------------

def load_cook_trajectory(path: str | Path) -> dict:
    """Load a cooking trajectory JSON.  Returns the full dict (with ``frames``)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_cook_trajectory(path: str | Path, traj_data: dict) -> None:
    """Write a cooking trajectory JSON."""
    Path(path).write_text(
        json.dumps(traj_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ------------------------------------------------------------------
# Bake
# ------------------------------------------------------------------

def bake_cook_trajectory(
    frames: list[dict],
    blend_schedule: BlendSchedule,
) -> list[dict]:
    """Return new frames list with blend keyframes applied.

    Position channels are blended linearly.  After blending, quaternion
    channels are renormalized (NLerp) so the result stays on the unit
    sphere.
    """
    out: list[dict] = []
    for f_idx, frame in enumerate(frames):
        teleop = frame_to_channels(frame)
        blended = blend_schedule.blend_joints(teleop, float(f_idx))
        for ent in ENTITIES:
            renormalize_quat_channels(blended, ent)
        out.append(channels_to_frame(blended, frame))
    return out
