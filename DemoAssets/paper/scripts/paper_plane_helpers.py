"""
Auxiliary helpers for the paper-plane fold sequence (boundary mode only).

Extracted from ``paper_plane_fold_sequence.py`` with all fan-mode code removed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from uipc import Logger, Matrix4x4, Engine, World, Scene, Animation, view, builtin
except ImportError as exc:
    raise SystemExit(
        "This example requires the libuipc Python bindings (`uipc._native.pyuipc`). "
        "Build/install the Python package before running it."
    ) from exc

_FOLD_DIR = os.path.dirname(__file__)
if _FOLD_DIR not in sys.path:
    sys.path.insert(0, _FOLD_DIR)

_TESTS_DIR = os.path.join(_FOLD_DIR, "..", "..", "tests")
if _TESTS_DIR not in sys.path:
    sys.path.append(_TESTS_DIR)

from asset_dir import AssetDir  # noqa: E402
import paper_plane_2_boundary_crease_debug_demo as boundary_debug  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MESH_PARTITION_SIZE = 16

SHELL_THICKNESS = 5e-5
SHELL_DENSITY = 1200.0
SHELL_YOUNG = 1.5e9
SHELL_POISSON = 0.3
SHELL_BENDING_STIFFNESS = 1.2e6
# strain-based plasticity parameters
SHELL_STRAIN_YIELD_THRESHOLD = 0.05
SHELL_STRAIN_HARDENING_MODULUS = 0.0
# stress-based plasticity parameters
SHELL_STRESS_YIELD_STRESS = 4.2e5
SHELL_STRESS_HARDENING_MODULUS = 0.0

STRONG_SPC_STRENGTH = 1200
WEAK_SPC_STRENGTH = 100.0

GROUND_Y = -0.01
PLANE_LIFT_Y = 0.001
DT = 0.01
CONTACT_D_HAT = 3e-3
NEWTON_TOL_RATE = 1.0e-2
STRONG_FIX_CURRENT_Y = 0.01

CUBE_SCALE = 0.4
CUBE_HOVER_Y = 0.50
CUBE_PRESS_Y = 0.21
CUBE_HOME_MARGIN_X = 0.35
CUBE_HOME_Z_RATIO = 0.20
CUBE_STC_STRENGTH = np.array([100.0, 100.0], dtype=np.float64)
OVERHEAD_CUBE_SCALE = 10
OVERHEAD_CUBE_HOVER_Y = 13
OVERHEAD_CUBE_PRESS_Y = 5.02

ROUNDED_CUBE_MESH_PATH = os.path.join(AssetDir.asset_path(), "rounded_cube.obj")
PLAIN_CUBE_MESH_PATH = None  # cube.obj removed; use _make_unit_cube_trimesh() inline

FOLD_MOVE_FRAMES = 220
FOLD_HOLD_FRAMES = 60
PRESS_TRANSFER_FRAMES = 140
PRESS_LOWER_FRAMES = 100
PRESS_TO_LEFT_FRAMES = 140
PRESS_TO_RIGHT_FRAMES = 300
PRESS_STAMP_MOVE_FRAMES = 10
PRESS_LIFT_FRAMES = 30
PRESS_RETURN_FRAMES = 120
INSPECT_FRAMES = 160
STAMP_LIFT_DELTA_Y = 0.2

TARGET_SNAP_TOL = 1.0e-3
GEOMETRY_MATCH_TOL = 2.0e-4
RESUME_MANIFEST_FILENAME = "resume_manifest.json"
DEFAULT_ASSET_FILENAME = "paper_plane_2_coarse.obj"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NamedVertex:
    label: str
    vertex_id: int
    xyz: np.ndarray


@dataclass(frozen=True)
class AnchorAction:
    label: str
    vertex_id: int
    xyz: np.ndarray
    lock_current_pose: bool = False
    current_pose_y: float | None = None


@dataclass(frozen=True)
class MoverAction:
    label: str
    vertex_id: int
    start_xyz: np.ndarray
    target_label: str
    target_vertex_id: int
    target_xyz: np.ndarray
    center_label: str
    center_xyz: np.ndarray
    axis_dir: np.ndarray
    axial_offset_start: float
    axial_offset_end: float
    radius_start: float
    radius_end: float
    basis_u: np.ndarray
    basis_v: np.ndarray
    signed_angle: float


@dataclass(frozen=True)
class PressSpec:
    mode: str = "sweep"
    path_labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PressAction:
    mode: str
    crease_name: str
    home_xyz: np.ndarray
    hover_y: float
    press_y: float
    vertical_only: bool
    path_labels: tuple[str, ...]
    path_vertex_ids: tuple[int, ...]
    path_xyzs: tuple[np.ndarray, ...]
    cube_yaw_radians: float


@dataclass(frozen=True)
class GlobalFlipSpec:
    axis_name: str
    lift_y: float = 0.3
    angle_degrees: float = 180.0
    phase: str = "full"
    group_name: str | None = None
    mover_labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GlobalFlipAction:
    group_name: str
    phase: str
    axis_name: str
    axis_start_label: str
    axis_end_label: str
    axis_start_vertex_id: int
    axis_end_vertex_id: int
    axis_start_xyz: np.ndarray
    axis_end_xyz: np.ndarray
    axis_dir: np.ndarray
    lift_y: float
    angle_radians: float
    mover_vertex_ids: tuple[int, ...]


@dataclass(frozen=True)
class MotionState:
    step_index: int
    step_name: str
    step_frame: int
    fold_alpha: float
    phase: str
    cube_center_xyz: np.ndarray


@dataclass(frozen=True)
class StepSpec:
    name: str
    crease_name: str
    mover_labels: tuple[str, ...]
    strong_fix_labels: tuple[str, ...]
    press: PressSpec | None
    free_run_frames: int = 0
    global_flip: GlobalFlipSpec | None = None
    current_pose_fix_labels: tuple[str, ...] = ()
    strong_fix_crease_names: tuple[str, ...] = ()
    strong_fix_segment_labels: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FoldStep:
    name: str
    crease_name: str
    weak_anchors: tuple[AnchorAction, ...]
    strong_fixes: tuple[AnchorAction, ...]
    movers: tuple[MoverAction, ...]
    targets: tuple[NamedVertex, ...]
    press: PressAction | None
    free_run_frames: int
    global_flip: GlobalFlipAction | None
    crease_vertex_ids: tuple[int, ...]


@dataclass(frozen=True)
class AssetContext:
    mode: str
    asset_filename: str
    cube_mesh_path: str
    plane: object
    diagnostics: dict[str, object]
    named_vertices: dict[str, NamedVertex]


@dataclass(frozen=True)
class SyntheticCreaseChain:
    crease_name: str
    vertex_ids: tuple[int, ...]
    endpoint_a_name: str
    endpoint_b_name: str


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-12:
        raise AssertionError("encountered a near-zero vector")
    return vec / norm


def smooth_lerp(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * math.cos(math.pi * t)
    return a + (b - a) * s


def lerp_point(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * smooth_lerp(0.0, 1.0, t)


def rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


# ---------------------------------------------------------------------------
# Cube transform
# ---------------------------------------------------------------------------


def cube_transform(center_xyz: np.ndarray, yaw_radians: float = 0.0) -> Matrix4x4:
    transform = Matrix4x4.Identity()
    c = math.cos(yaw_radians)
    s = math.sin(yaw_radians)
    transform[0, 0] = c
    transform[0, 2] = s
    transform[2, 0] = -s
    transform[2, 2] = c
    transform[0:3, 3] = np.asarray(center_xyz, dtype=np.float64)
    return transform


# ---------------------------------------------------------------------------
# Timing / scheduling
# ---------------------------------------------------------------------------


def press_phase_duration(press: PressSpec | PressAction | None) -> int:
    if press is None:
        return 0
    if isinstance(press, PressAction) and press.vertical_only:
        return PRESS_LOWER_FRAMES + PRESS_LIFT_FRAMES
    if isinstance(press, PressSpec):
        point_count = len(press.path_labels) if press.path_labels is not None else 3
    else:
        point_count = len(press.path_labels)
    if point_count < 2:
        raise AssertionError("press path must contain at least two points")
    if press.mode == "stamped":
        return (
            PRESS_TRANSFER_FRAMES
            + point_count * (PRESS_LOWER_FRAMES + PRESS_LIFT_FRAMES)
            + max(point_count - 1, 0) * PRESS_STAMP_MOVE_FRAMES
            + PRESS_RETURN_FRAMES
        )
    duration = PRESS_TRANSFER_FRAMES + PRESS_LOWER_FRAMES + PRESS_LIFT_FRAMES + PRESS_RETURN_FRAMES
    for segment_index in range(point_count - 1):
        duration += PRESS_TO_LEFT_FRAMES if point_count >= 3 and segment_index == 0 else PRESS_TO_RIGHT_FRAMES
    return duration


def step_duration(step: StepSpec | FoldStep) -> int:
    if step.free_run_frames > 0:
        return int(step.free_run_frames)
    if step.global_flip is not None:
        lift_frames, rotate_frames, lower_frames = global_flip_phase_frames()
        phase = step.global_flip.phase
        if phase == "lift":
            return lift_frames
        if phase == "rotate":
            return rotate_frames
        if phase == "lower":
            return lower_frames
        return lift_frames + rotate_frames + lower_frames
    return FOLD_MOVE_FRAMES + FOLD_HOLD_FRAMES + press_phase_duration(step.press)


def total_sequence_frames(steps: tuple[StepSpec, ...] | tuple[FoldStep, ...]) -> int:
    return sum(step_duration(step) for step in steps) + INSPECT_FRAMES


def completed_step_end_frames(steps: tuple[StepSpec, ...] | tuple[FoldStep, ...]) -> tuple[int, ...]:
    frames = []
    frame = 0
    for step in steps:
        frame += step_duration(step)
        frames.append(frame)
    return tuple(frames)


def next_step_index_for_frame(
    frame: int,
    steps: tuple[StepSpec, ...] | tuple[FoldStep, ...],
) -> int | None:
    remaining = max(frame, 0)
    for index, step in enumerate(steps):
        duration = step_duration(step)
        if remaining < duration:
            return index
        remaining -= duration
    return None


def sequence_schedule(frame: int, steps: tuple[StepSpec, ...] | tuple[FoldStep, ...]) -> tuple[int, int]:
    if not steps:
        raise AssertionError("fold sequence must contain at least one step")
    remaining = max(frame, 0)
    for index, step in enumerate(steps):
        duration = step_duration(step)
        if remaining < duration:
            return index, remaining
        remaining -= duration
    last_index = len(steps) - 1
    return last_index, step_duration(steps[last_index]) + remaining


def global_flip_phase_frames() -> tuple[int, int, int]:
    lift_frames = max(FOLD_MOVE_FRAMES // 4, 1)
    lower_frames = max(FOLD_MOVE_FRAMES // 4, 1)
    rotate_frames = max(FOLD_MOVE_FRAMES - lift_frames - lower_frames, 1)
    return lift_frames, rotate_frames, lower_frames


# ---------------------------------------------------------------------------
# Motion schedule
# ---------------------------------------------------------------------------


def _xz_points(xyzs: tuple[np.ndarray, ...], y: float) -> tuple[np.ndarray, ...]:
    return tuple(np.array([xyz[0], y, xyz[2]], dtype=np.float64) for xyz in xyzs)


def motion_schedule(
    frame: int,
    steps: tuple[StepSpec, ...] | tuple[FoldStep, ...],
    step_index: int,
    press: PressAction | None,
    default_cube_home: np.ndarray | None = None,
) -> MotionState:
    local_frame = max(frame, 0)
    step = steps[step_index]
    if press is not None:
        cube_home = np.array(press.home_xyz, copy=True)
    elif default_cube_home is not None:
        cube_home = np.array(default_cube_home, copy=True)
    else:
        cube_home = np.array([0.0, CUBE_HOVER_Y, 0.0], dtype=np.float64)
    step_name = step.name
    if step.free_run_frames > 0:
        return MotionState(step_index, step_name, frame, 0.0, "free-run", cube_home)
    if step.global_flip is not None:
        lift_frames, rotate_frames, lower_frames = global_flip_phase_frames()
        phase = step.global_flip.phase
        if phase == "lift":
            alpha = smooth_lerp(0.0, 1.0, local_frame / max(lift_frames - 1, 1))
            return MotionState(step_index, step_name, frame, alpha, "flip-lift", cube_home)
        if phase == "rotate":
            alpha = smooth_lerp(0.0, 1.0, local_frame / max(rotate_frames - 1, 1))
            return MotionState(step_index, step_name, frame, alpha, "flip-rotate", cube_home)
        if phase == "lower":
            alpha = smooth_lerp(0.0, 1.0, local_frame / max(lower_frames - 1, 1))
            return MotionState(step_index, step_name, frame, alpha, "flip-lower", cube_home)
        if local_frame < lift_frames:
            alpha = smooth_lerp(0.0, 1.0, local_frame / max(lift_frames - 1, 1))
            return MotionState(step_index, step_name, frame, alpha, "flip-lift", cube_home)
        local_frame -= lift_frames
        if local_frame < rotate_frames:
            alpha = smooth_lerp(0.0, 1.0, local_frame / max(rotate_frames - 1, 1))
            return MotionState(step_index, step_name, frame, alpha, "flip-rotate", cube_home)
        local_frame -= rotate_frames
        if local_frame < lower_frames:
            alpha = smooth_lerp(0.0, 1.0, local_frame / max(lower_frames - 1, 1))
            return MotionState(step_index, step_name, frame, alpha, "flip-lower", cube_home)
        return MotionState(step_index, step_name, frame, 1.0, "flip-hold", cube_home)
    if local_frame < FOLD_MOVE_FRAMES:
        alpha = smooth_lerp(0.0, 1.0, local_frame / max(FOLD_MOVE_FRAMES - 1, 1))
        return MotionState(step_index, step_name, frame, alpha, "fold-move", cube_home)
    local_frame -= FOLD_MOVE_FRAMES
    if local_frame < FOLD_HOLD_FRAMES:
        return MotionState(step_index, step_name, frame, 1.0, "fold-hold", cube_home)
    if press is None:
        return MotionState(step_index, step_name, frame, 1.0, "inspect", cube_home)
    if press.vertical_only:
        press_xyz = np.array([press.home_xyz[0], press.press_y, press.home_xyz[2]], dtype=np.float64)
        local_frame -= FOLD_HOLD_FRAMES
        if local_frame < PRESS_LOWER_FRAMES:
            cube_xyz = lerp_point(cube_home, press_xyz, local_frame / max(PRESS_LOWER_FRAMES - 1, 1))
            return MotionState(step_index, step_name, frame, 1.0, "cube-lower", cube_xyz)
        local_frame -= PRESS_LOWER_FRAMES
        if local_frame < PRESS_LIFT_FRAMES:
            cube_xyz = lerp_point(press_xyz, cube_home, local_frame / max(PRESS_LIFT_FRAMES - 1, 1))
            return MotionState(step_index, step_name, frame, 1.0, "cube-lift", cube_xyz)
        return MotionState(step_index, step_name, frame, 1.0, "inspect", cube_home)
    hover_points = _xz_points(press.path_xyzs, press.hover_y)
    press_points = _xz_points(press.path_xyzs, press.press_y)
    local_frame -= FOLD_HOLD_FRAMES
    stamp_hover_points = _xz_points(press.path_xyzs, press.press_y + STAMP_LIFT_DELTA_Y)
    if local_frame < PRESS_TRANSFER_FRAMES:
        transfer_points = stamp_hover_points if press.mode == "stamped" else hover_points
        cube_xyz = lerp_point(cube_home, transfer_points[0], local_frame / max(PRESS_TRANSFER_FRAMES - 1, 1))
        return MotionState(step_index, step_name, frame, 1.0, "cube-transfer", cube_xyz)
    local_frame -= PRESS_TRANSFER_FRAMES
    if press.mode == "stamped":
        for stamp_index in range(len(press_points)):
            if local_frame < PRESS_LOWER_FRAMES:
                cube_xyz = lerp_point(
                    stamp_hover_points[stamp_index],
                    press_points[stamp_index],
                    local_frame / max(PRESS_LOWER_FRAMES - 1, 1),
                )
                return MotionState(step_index, step_name, frame, 1.0, f"cube-lower-{stamp_index}", cube_xyz)
            local_frame -= PRESS_LOWER_FRAMES
            if local_frame < PRESS_LIFT_FRAMES:
                cube_xyz = lerp_point(
                    press_points[stamp_index],
                    stamp_hover_points[stamp_index],
                    local_frame / max(PRESS_LIFT_FRAMES - 1, 1),
                )
                return MotionState(step_index, step_name, frame, 1.0, f"cube-lift-{stamp_index}", cube_xyz)
            local_frame -= PRESS_LIFT_FRAMES
            if stamp_index + 1 < len(press_points):
                if local_frame < PRESS_STAMP_MOVE_FRAMES:
                    cube_xyz = lerp_point(
                        stamp_hover_points[stamp_index],
                        stamp_hover_points[stamp_index + 1],
                        local_frame / max(PRESS_STAMP_MOVE_FRAMES - 1, 1),
                    )
                    return MotionState(
                        step_index,
                        step_name,
                        frame,
                        1.0,
                        f"cube-hop-{stamp_index + 1}",
                        cube_xyz,
                    )
                local_frame -= PRESS_STAMP_MOVE_FRAMES
        if local_frame < PRESS_RETURN_FRAMES:
            cube_xyz = lerp_point(
                stamp_hover_points[-1],
                cube_home,
                local_frame / max(PRESS_RETURN_FRAMES - 1, 1),
            )
            return MotionState(step_index, step_name, frame, 1.0, "cube-return", cube_xyz)
        return MotionState(step_index, step_name, frame, 1.0, "inspect", cube_home)
    if local_frame < PRESS_LOWER_FRAMES:
        cube_xyz = lerp_point(hover_points[0], press_points[0], local_frame / max(PRESS_LOWER_FRAMES - 1, 1))
        return MotionState(step_index, step_name, frame, 1.0, "cube-lower", cube_xyz)
    local_frame -= PRESS_LOWER_FRAMES
    for segment_index in range(len(press_points) - 1):
        segment_frames = (
            PRESS_TO_LEFT_FRAMES if len(press_points) >= 3 and segment_index == 0 else PRESS_TO_RIGHT_FRAMES
        )
        if local_frame < segment_frames:
            cube_xyz = lerp_point(
                press_points[segment_index],
                press_points[segment_index + 1],
                local_frame / max(segment_frames - 1, 1),
            )
            return MotionState(step_index, step_name, frame, 1.0, f"cube-to-path-{segment_index + 1}", cube_xyz)
        local_frame -= segment_frames
    if local_frame < PRESS_LIFT_FRAMES:
        cube_xyz = lerp_point(press_points[-1], hover_points[-1], local_frame / max(PRESS_LIFT_FRAMES - 1, 1))
        return MotionState(step_index, step_name, frame, 1.0, "cube-lift", cube_xyz)
    local_frame -= PRESS_LIFT_FRAMES
    if local_frame < PRESS_RETURN_FRAMES:
        cube_xyz = lerp_point(hover_points[-1], cube_home, local_frame / max(PRESS_RETURN_FRAMES - 1, 1))
        return MotionState(step_index, step_name, frame, 1.0, "cube-return", cube_xyz)
    return MotionState(step_index, step_name, frame, 1.0, "inspect", cube_home)


# ---------------------------------------------------------------------------
# Mover / flip position helpers
# ---------------------------------------------------------------------------


def mover_position(action: MoverAction, alpha: float) -> np.ndarray:
    phi = action.signed_angle * alpha
    axial_offset = (1.0 - alpha) * action.axial_offset_start + alpha * action.axial_offset_end
    radius = (1.0 - alpha) * action.radius_start + alpha * action.radius_end
    radial = math.cos(phi) * action.basis_u + math.sin(phi) * action.basis_v
    return action.center_xyz + axial_offset * action.axis_dir + radius * radial


def rotate_point_about_axis(
    point_xyz: np.ndarray,
    axis_start_xyz: np.ndarray,
    axis_dir: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    relative = np.asarray(point_xyz, dtype=np.float64) - np.asarray(axis_start_xyz, dtype=np.float64)
    return np.asarray(axis_start_xyz, dtype=np.float64) + rodrigues_rotate(relative, axis_dir, angle_radians)


def global_flip_position(
    start_xyz: np.ndarray,
    action: GlobalFlipAction,
    motion: MotionState,
) -> np.ndarray:
    lift_offset = np.array([0.0, action.lift_y, 0.0], dtype=np.float64)
    if motion.phase == "flip-lift":
        return np.asarray(start_xyz, dtype=np.float64) + motion.fold_alpha * lift_offset

    lifted_xyz = np.asarray(start_xyz, dtype=np.float64) + lift_offset
    lifted_axis_start = action.axis_start_xyz + lift_offset
    if motion.phase == "flip-rotate":
        return rotate_point_about_axis(
            lifted_xyz,
            lifted_axis_start,
            action.axis_dir,
            motion.fold_alpha * action.angle_radians,
        )

    fully_rotated_xyz = rotate_point_about_axis(
        lifted_xyz,
        lifted_axis_start,
        action.axis_dir,
        action.angle_radians,
    )
    if motion.phase == "flip-lower":
        return fully_rotated_xyz - motion.fold_alpha * lift_offset
    return fully_rotated_xyz - lift_offset


def recover_global_flip_start_positions(
    current_xyz: np.ndarray,
    action: GlobalFlipAction,
) -> np.ndarray:
    start_xyz = np.array(current_xyz, copy=True)
    lift_offset = np.array([0.0, action.lift_y, 0.0], dtype=np.float64)
    mover_ids = tuple(int(vid) for vid in action.mover_vertex_ids)
    if action.phase == "rotate":
        for vertex_id in mover_ids:
            start_xyz[vertex_id] = current_xyz[vertex_id] - lift_offset
        return start_xyz
    if action.phase == "lower":
        lifted_axis_start = np.array(current_xyz[action.axis_start_vertex_id], copy=True)
        for vertex_id in mover_ids:
            rotated_lifted_xyz = np.array(current_xyz[vertex_id], copy=True)
            lifted_xyz = rotate_point_about_axis(
                rotated_lifted_xyz,
                lifted_axis_start,
                action.axis_dir,
                -action.angle_radians,
            )
            start_xyz[vertex_id] = lifted_xyz - lift_offset
        return start_xyz
    return start_xyz


# ---------------------------------------------------------------------------
# Manifest / resume helpers
# ---------------------------------------------------------------------------


def timing_manifest_data() -> dict[str, int]:
    return {
        "FOLD_MOVE_FRAMES": FOLD_MOVE_FRAMES,
        "FOLD_HOLD_FRAMES": FOLD_HOLD_FRAMES,
        "PRESS_TRANSFER_FRAMES": PRESS_TRANSFER_FRAMES,
        "PRESS_LOWER_FRAMES": PRESS_LOWER_FRAMES,
        "PRESS_TO_LEFT_FRAMES": PRESS_TO_LEFT_FRAMES,
        "PRESS_TO_RIGHT_FRAMES": PRESS_TO_RIGHT_FRAMES,
        "PRESS_STAMP_MOVE_FRAMES": PRESS_STAMP_MOVE_FRAMES,
        "PRESS_LIFT_FRAMES": PRESS_LIFT_FRAMES,
        "PRESS_RETURN_FRAMES": PRESS_RETURN_FRAMES,
        "INSPECT_FRAMES": INSPECT_FRAMES,
    }


def step_manifest_entries(step_specs: tuple[StepSpec, ...]) -> list[dict[str, object]]:
    return [
        {
            "name": step.name,
            "crease_name": step.crease_name,
            "mover_labels": list(step.mover_labels),
            "strong_fix_labels": list(step.strong_fix_labels),
            "current_pose_fix_labels": list(step.current_pose_fix_labels),
            "strong_fix_crease_names": list(step.strong_fix_crease_names),
            "strong_fix_segment_labels": [list(pair) for pair in step.strong_fix_segment_labels],
            "free_run_frames": step.free_run_frames,
            "press": (
                None
                if step.press is None
                else {
                    "mode": step.press.mode,
                    "path_labels": None if step.press.path_labels is None else list(step.press.path_labels),
                }
            ),
            "global_flip": (
                None
                if step.global_flip is None
                else {
                    "axis_name": step.global_flip.axis_name,
                    "lift_y": step.global_flip.lift_y,
                    "angle_degrees": step.global_flip.angle_degrees,
                    "phase": step.global_flip.phase,
                    "group_name": step.global_flip.group_name,
                    "mover_labels": (
                        None if step.global_flip.mover_labels is None else list(step.global_flip.mover_labels)
                    ),
                }
            ),
        }
        for step in step_specs
    ]


def timing_signature(
    step_specs: tuple[StepSpec, ...],
    schedule_steps: tuple[FoldStep, ...],
) -> str:
    payload = {
        "step_defs": step_manifest_entries(step_specs),
        "step_end_frames": list(completed_step_end_frames(schedule_steps)),
        "timing": timing_manifest_data(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resume_manifest_path(workspace: str) -> str:
    return os.path.join(workspace, RESUME_MANIFEST_FILENAME)


def load_resume_manifest(workspace: str) -> tuple[dict[str, object] | None, str | None]:
    manifest_path = resume_manifest_path(workspace)
    if not os.path.exists(manifest_path):
        return None, "manifest not found"
    try:
        with open(manifest_path, "r", encoding="utf-8") as infile:
            manifest = json.load(infile)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"manifest unreadable: {exc}"
    required = (
        "asset_filename",
        "asset_stem",
        "step_defs",
        "step_end_frames",
        "total_frames",
        "last_completed_step_index",
        "last_completed_step_name",
        "last_dump_frame",
        "timing_signature",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        return None, f"manifest missing keys: {', '.join(missing)}"
    return manifest, None


def save_resume_manifest(workspace: str, manifest: dict[str, object]) -> None:
    with open(resume_manifest_path(workspace), "w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)


def resolved_asset_filename(asset_filename: str | None) -> str:
    asset_filename = asset_filename or DEFAULT_ASSET_FILENAME
    return os.path.abspath(boundary_debug.resolve_origami_asset_path(asset_filename))


def asset_identity(asset_filename: str | None) -> tuple[str, str, str]:
    resolved = resolved_asset_filename(asset_filename)
    basename = os.path.basename(resolved)
    return resolved, basename, Path(basename).stem


def demo_workspace(
    asset_filename: str | None,
    sequence: str = "full",
    bending_model: str = "strain",
) -> str:
    _, _, asset_stem = asset_identity(asset_filename)
    base_output = Path(AssetDir.output_path(__file__)).resolve()
    workspace = base_output.parent / Path(__file__).stem / asset_stem
    if bending_model != "strain":
        workspace = workspace / bending_model
    if sequence != "full":
        workspace = workspace / sequence
    workspace.mkdir(parents=True, exist_ok=True)
    return str(workspace)


def resume_manifest_template(
    asset_filename: str | None,
    step_specs: tuple[StepSpec, ...],
    schedule_steps: tuple[FoldStep, ...],
    total_frames: int,
    sequence: str = "full",
    bending_model: str = "strain",
) -> dict[str, object]:
    resolved_filename, _, asset_stem = asset_identity(asset_filename)
    return {
        "asset_filename": resolved_filename,
        "asset_stem": asset_stem,
        "sequence": sequence,
        "bending_model": bending_model,
        "step_defs": step_manifest_entries(step_specs),
        "step_end_frames": list(completed_step_end_frames(schedule_steps)),
        "total_frames": total_frames,
        "timing": timing_manifest_data(),
        "last_completed_step_index": None,
        "last_completed_step_name": None,
        "last_dump_frame": None,
        "timing_signature": timing_signature(step_specs, schedule_steps),
    }


def completed_prefix_compatible(
    manifest: dict[str, object],
    manifest_template: dict[str, object],
    completed_index: int,
) -> str | None:
    prefix_count = completed_index + 1
    for field_name in ("step_defs", "step_end_frames"):
        actual = manifest.get(field_name)
        expected = manifest_template.get(field_name)
        if not isinstance(actual, list) or actual[:prefix_count] != expected[:prefix_count]:
            return f"{field_name} mismatch through completed step"
    actual_bending_model = manifest.get("bending_model") or "strain"
    expected_bending_model = manifest_template.get("bending_model") or "strain"
    if actual_bending_model != expected_bending_model:
        return "bending_model mismatch"
    if manifest.get("last_completed_step_name") != manifest_template["step_defs"][completed_index]["name"]:
        return "last_completed_step_name mismatch"
    return None


def maybe_recover_world(
    world: World,
    workspace: str,
    manifest_template: dict[str, object],
    schedule_steps: tuple[FoldStep, ...],
    from_start: bool,
    start_after_step: int | None,
) -> dict[str, object]:
    resume_info: dict[str, object] = {
        "mode": "from-start" if from_start else "fresh",
        "manifest_path": resume_manifest_path(workspace),
        "recovered": False,
        "recovered_frame": None,
        "last_completed_step_index": None,
        "last_completed_step_name": None,
        "next_step_index": 0 if schedule_steps else None,
        "next_step_name": schedule_steps[0].name if schedule_steps else "inspect",
        "skip_reason": "disabled by --from-start" if from_start else "no valid manifest",
    }
    if start_after_step is not None:
        if not schedule_steps:
            raise SystemExit("--start-after-step requires at least one step in the sequence.")
        if start_after_step < 0 or start_after_step > len(schedule_steps):
            raise SystemExit(f"--start-after-step must be between 0 and {len(schedule_steps)}, got {start_after_step}.")
        if start_after_step == 0:
            print("[paper_plane_resume] resume skipped: explicit start-after-step 0")
            resume_info.update(
                {
                    "mode": "start-after-step",
                    "skip_reason": "explicit start-after-step 0",
                    "next_step_index": 0,
                    "next_step_name": schedule_steps[0].name,
                }
            )
            return resume_info
        dump_frame = completed_step_end_frames(schedule_steps)[start_after_step - 1]
        if not world.recover(dump_frame):
            raise SystemExit(
                f"Requested --start-after-step {start_after_step}, but recover({dump_frame}) failed. "
                "Make sure that checkpoint dump exists."
            )
        world.retrieve()
        completed_index = start_after_step - 1
        next_index = next_step_index_for_frame(dump_frame, schedule_steps)
        next_name = schedule_steps[next_index].name if next_index is not None else "inspect"
        print(
            f"[paper_plane_resume] recovered requested step boundary {dump_frame} "
            f"after step {schedule_steps[completed_index].name}"
        )
        resume_info.update(
            {
                "mode": "start-after-step",
                "recovered": True,
                "recovered_frame": dump_frame,
                "last_completed_step_index": completed_index,
                "last_completed_step_name": schedule_steps[completed_index].name,
                "next_step_index": next_index,
                "next_step_name": next_name,
                "skip_reason": "",
            }
        )
        return resume_info
    if from_start:
        print("[paper_plane_resume] resume skipped: disabled by --from-start")
        return resume_info
    manifest, reason = load_resume_manifest(workspace)
    if manifest is None:
        print(f"[paper_plane_resume] resume skipped: {reason}")
        resume_info["skip_reason"] = reason
        return resume_info
    manifest_bending_model = manifest.get("bending_model") or "strain"
    expected_bending_model = manifest_template.get("bending_model") or "strain"
    if manifest_bending_model != expected_bending_model:
        message = f"bending model mismatch: manifest={manifest_bending_model}, expected={expected_bending_model}"
        print(f"[paper_plane_resume] resume skipped: {message}")
        resume_info["skip_reason"] = message
        return resume_info
    try:
        dump_frame = int(manifest["last_dump_frame"])
    except (TypeError, ValueError):
        message = "manifest has invalid dump frame"
        print(f"[paper_plane_resume] resume skipped: {message}")
        resume_info["skip_reason"] = message
        return resume_info
    if not world.recover(dump_frame):
        message = f"world.recover({dump_frame}) failed"
        print(f"[paper_plane_resume] resume skipped: {message}")
        resume_info["skip_reason"] = message
        return resume_info
    world.retrieve()
    completed_index_raw = manifest.get("last_completed_step_index")
    try:
        completed_index = int(completed_index_raw)
    except (TypeError, ValueError):
        completed_index = None
    if completed_index is not None and not (0 <= completed_index < len(schedule_steps)):
        completed_index = None
    next_index = next_step_index_for_frame(dump_frame, schedule_steps)
    next_name = schedule_steps[next_index].name if next_index is not None else "inspect"
    print(
        f"[paper_plane_resume] recovered from frame {dump_frame} after step {manifest.get('last_completed_step_name')}"
    )
    resume_info.update(
        {
            "mode": "recovered",
            "recovered": True,
            "recovered_frame": dump_frame,
            "last_completed_step_index": completed_index,
            "last_completed_step_name": manifest.get("last_completed_step_name"),
            "next_step_index": next_index,
            "next_step_name": next_name,
            "skip_reason": "",
        }
    )
    return resume_info


def maybe_dump_completed_step(
    world: World,
    frame: int,
    workspace: str,
    step_specs: tuple[StepSpec, ...],
    step_end_frames: tuple[int, ...],
    manifest_template: dict[str, object],
    runtime: dict[str, object],
    resume_info: dict[str, object],
    schedule_steps: tuple[FoldStep, ...],
) -> None:
    if frame not in step_end_frames:
        return
    step_index = step_end_frames.index(frame)
    dumped_step_indices = runtime["dumped_step_indices"]
    if step_index in dumped_step_indices:
        return
    if not world.dump():
        print(f"[paper_plane_resume] dump skipped: world.dump() failed at frame {frame}")
        return
    dumped_step_indices.add(step_index)
    next_index = next_step_index_for_frame(frame, schedule_steps)
    next_name = schedule_steps[next_index].name if next_index is not None else "inspect"
    manifest = dict(manifest_template)
    manifest.update(
        {
            "last_completed_step_index": step_index,
            "last_completed_step_name": step_specs[step_index].name,
            "last_dump_frame": frame,
        }
    )
    save_resume_manifest(workspace, manifest)
    resume_info.update(
        {
            "last_completed_step_index": step_index,
            "last_completed_step_name": step_specs[step_index].name,
            "next_step_index": next_index,
            "next_step_name": next_name,
        }
    )
    print(f"[paper_plane_resume] dumped frame {frame} after step {step_specs[step_index].name}")


# ---------------------------------------------------------------------------
# Cube mesh path
# ---------------------------------------------------------------------------


def default_cube_mesh_path() -> str:
    # PLAIN_CUBE_MESH_PATH (cube.obj) has been removed; fall back to rounded_cube.
    if PLAIN_CUBE_MESH_PATH is not None and os.path.exists(PLAIN_CUBE_MESH_PATH):
        return PLAIN_CUBE_MESH_PATH
    return ROUNDED_CUBE_MESH_PATH


# ---------------------------------------------------------------------------
# Mesh processing
# ---------------------------------------------------------------------------


def orient_mesh_for_ground(mesh) -> None:
    positions_view = view(mesh.positions())
    positions = np.array(positions_view, copy=True).reshape(-1, 3)
    spans = positions.max(axis=0) - positions.min(axis=0)
    normal_axis = int(np.argmin(spans))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    plane_axes.sort(key=lambda axis: spans[axis])
    aligned = np.zeros_like(positions)
    aligned[:, 0] = positions[:, plane_axes[0]]
    aligned[:, 1] = 0.0
    aligned[:, 2] = positions[:, plane_axes[1]]
    aligned[:, 1] += PLANE_LIFT_Y - float(aligned[:, 1].min())
    positions = aligned
    positions_view[:] = positions.reshape(positions_view.shape)


def process_closed_surface(
    sc,
    label_surface_fn,
    label_triangle_orient_fn,
    flip_inward_triangles_fn,
):
    label_surface_fn(sc)
    if sc.dim() != 3:
        return sc
    label_triangle_orient_fn(sc)
    return flip_inward_triangles_fn(sc)


# ---------------------------------------------------------------------------
# Nearest-vertex helpers
# ---------------------------------------------------------------------------


def nearest_boundary_vertex(
    positions: np.ndarray,
    boundary_loop: list[int],
    plane_axes: tuple[int, int],
    u_target: float,
    v_target: float,
) -> int:
    target = np.array([u_target, v_target], dtype=np.float64)
    return min(
        boundary_loop,
        key=lambda vid: float(np.linalg.norm(positions[vid, list(plane_axes)] - target)),
    )


def nearest_vertex(positions: np.ndarray, target_xyz: np.ndarray) -> tuple[int, float]:
    distances = np.linalg.norm(positions - target_xyz.reshape(1, 3), axis=1)
    vertex_id = int(np.argmin(distances))
    return vertex_id, float(distances[vertex_id])


def nearest_vertex_between_point_and_crease(
    positions: np.ndarray,
    diagnostics: dict[str, object],
    named_ids: dict[str, int],
    point_label: str,
    crease_name: str,
) -> int:
    point_xyz = positions[named_ids[point_label]]
    endpoint_labels = crease_endpoint_labels(crease_name, diagnostics)
    axis_start_xyz = positions[named_ids[endpoint_labels[0]]]
    axis_end_xyz = positions[named_ids[endpoint_labels[1]]]
    axis_dir = normalize(axis_end_xyz - axis_start_xyz)
    projected_xyz = axis_start_xyz + np.dot(point_xyz - axis_start_xyz, axis_dir) * axis_dir
    midpoint_xyz = 0.5 * (point_xyz + projected_xyz)
    vertex_id, _ = nearest_vertex(positions, midpoint_xyz)
    return vertex_id


def nearest_vertex_between_points(
    positions: np.ndarray,
    named_ids: dict[str, int],
    point_a_label: str,
    point_b_label: str,
) -> int:
    midpoint_xyz = 0.5 * (positions[named_ids[point_a_label]] + positions[named_ids[point_b_label]])
    vertex_id, _ = nearest_vertex(positions, midpoint_xyz)
    return vertex_id


# ---------------------------------------------------------------------------
# Segment / crease sampling
# ---------------------------------------------------------------------------


def sample_segment_vertex_ids(
    positions: np.ndarray,
    named_vertices: dict[str, NamedVertex],
    start_label: str,
    end_label: str,
    spacing: float,
) -> tuple[int, ...]:
    start_xyz = np.array(named_vertices[start_label].xyz, copy=True)
    end_xyz = np.array(named_vertices[end_label].xyz, copy=True)
    segment = end_xyz - start_xyz
    length = float(np.linalg.norm(segment))
    if length < 1.0e-12:
        return (int(named_vertices[start_label].vertex_id),)
    sample_count = max(int(math.ceil(length / max(spacing, 1.0e-6))), 1) + 1
    vertex_ids: list[int] = []
    seen: set[int] = set()
    for sample_index in range(sample_count):
        t = sample_index / max(sample_count - 1, 1)
        xyz = start_xyz + t * segment
        vertex_id, _ = nearest_vertex(positions, xyz)
        if vertex_id in seen:
            continue
        seen.add(vertex_id)
        vertex_ids.append(vertex_id)
    return tuple(vertex_ids)


def resolve_named_vertices(diagnostics: dict[str, object]) -> dict[str, NamedVertex]:
    positions = diagnostics["positions"]
    boundary_loop = diagnostics["boundary_loop"]
    plane_axes = tuple(int(axis) for axis in diagnostics["plane_axes"])
    xmin, xmax, ymin, ymax = diagnostics["bbox"]
    named_ids: dict[str, int] = {}
    for label, vertex_id in diagnostics["point_lookup"].items():
        named_ids[label] = int(vertex_id)
    named_ids.update(
        {
            "LEFT_TOP": nearest_boundary_vertex(positions, boundary_loop, plane_axes, xmin, ymax),
            "RIGHT_TOP": nearest_boundary_vertex(positions, boundary_loop, plane_axes, xmax, ymax),
            "LEFT_BOTTOM": nearest_boundary_vertex(positions, boundary_loop, plane_axes, xmin, ymin),
            "RIGHT_BOTTOM": nearest_boundary_vertex(positions, boundary_loop, plane_axes, xmax, ymin),
            "TOP": nearest_boundary_vertex(positions, boundary_loop, plane_axes, 0.5 * (xmin + xmax), ymax),
            "BOTTOM": nearest_boundary_vertex(positions, boundary_loop, plane_axes, 0.5 * (xmin + xmax), ymin),
        }
    )
    named_ids["TOP_LEFT"] = named_ids["LEFT_TOP"]
    named_ids["TOP_RIGHT"] = named_ids["RIGHT_TOP"]
    named_ids["BOTTOM_LEFT"] = named_ids["LEFT_BOTTOM"]
    named_ids["BOTTOM_RIGHT"] = named_ids["RIGHT_BOTTOM"]
    named_ids["P08A_TO_LR_H3_MID"] = nearest_vertex_between_point_and_crease(
        positions,
        diagnostics,
        named_ids,
        "P08A",
        "LR_H3",
    )
    named_ids["P06B_TO_P09_C02_TO_B_MID"] = nearest_vertex_between_point_and_crease(
        positions,
        diagnostics,
        named_ids,
        "P06B",
        "P09_C02_TO_B",
    )
    named_ids["P09_C02_X_TO_P08_P13_X_MID"] = nearest_vertex_between_points(
        positions,
        named_ids,
        "P09_C02_X",
        "P08_P13_X",
    )
    named_ids["P06A_TO_P13_A_TO_C02_MID"] = nearest_vertex_between_point_and_crease(
        positions,
        diagnostics,
        named_ids,
        "P06A",
        "P13_A_TO_C02",
    )
    named_ids["P13_C02_X_TO_P09_P12_X_MID"] = nearest_vertex_between_points(
        positions,
        named_ids,
        "P13_C02_X",
        "P09_P12_X",
    )
    named_vertices = {
        label: NamedVertex(label=label, vertex_id=vertex_id, xyz=np.array(positions[vertex_id], copy=True))
        for label, vertex_id in named_ids.items()
    }
    return named_vertices


# ---------------------------------------------------------------------------
# Boundary registration helpers
# ---------------------------------------------------------------------------


def register_boundary_point(diagnostics: dict[str, object], label: str, vertex_id: int) -> None:
    diagnostics["point_lookup"][label] = int(vertex_id)
    diagnostics["endpoint_lookup"][label] = int(vertex_id)


def register_boundary_segment_crease(
    diagnostics: dict[str, object],
    crease_name: str,
    vertex_ids: tuple[int, ...],
    endpoint_a_name: str,
    endpoint_b_name: str,
) -> None:
    diagnostics["crease_lookup"][crease_name] = SyntheticCreaseChain(
        crease_name=crease_name,
        vertex_ids=tuple(int(vid) for vid in vertex_ids),
        endpoint_a_name=endpoint_a_name,
        endpoint_b_name=endpoint_b_name,
    )
    register_boundary_point(diagnostics, endpoint_a_name, int(vertex_ids[0]))
    register_boundary_point(diagnostics, endpoint_b_name, int(vertex_ids[-1]))


def register_internal_row_crease(
    diagnostics: dict[str, object],
    crease_name: str,
    endpoint_a_name: str,
    endpoint_b_name: str,
    *,
    row_coord: float,
    axis_min: float,
    axis_max: float,
    row_tol: float = 1.0e-6,
) -> None:
    positions = diagnostics["positions"]
    adjacency = diagnostics["adjacency"]
    axis_u, axis_v = (int(axis) for axis in diagnostics["plane_axes"])
    row_ids = [
        int(vid)
        for vid, xyz in enumerate(positions)
        if abs(float(xyz[axis_v]) - row_coord) <= row_tol
        and axis_min - row_tol <= float(xyz[axis_u]) <= axis_max + row_tol
    ]
    if len(row_ids) < 2:
        raise AssertionError(f"failed to find row candidates for {crease_name}")

    row_set = set(row_ids)
    seen: set[int] = set()
    best_component: list[int] = []
    for start in row_ids:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for nb in adjacency[current]:
                if nb in row_set and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(component) > len(best_component):
            best_component = component
    if len(best_component) < 2:
        raise AssertionError(f"failed to find a connected row for {crease_name}")

    vertex_ids = tuple(
        sorted(best_component, key=lambda vid: (float(positions[vid, axis_u]), float(positions[vid, axis_v])))
    )
    register_boundary_segment_crease(
        diagnostics,
        crease_name,
        vertex_ids,
        endpoint_a_name,
        endpoint_b_name,
    )


def register_boundary_loop_segment_crease(
    diagnostics: dict[str, object],
    crease_name: str,
    start_label: str,
    end_label: str,
    *,
    prefer: str = "shortest",
) -> None:
    boundary_loop = [int(vid) for vid in diagnostics["boundary_loop"]]
    positions = diagnostics["positions"]
    axis_u, axis_v = (int(axis) for axis in diagnostics["plane_axes"])
    start_id = int(diagnostics["point_lookup"][start_label])
    end_id = int(diagnostics["point_lookup"][end_label])
    start_index = boundary_loop.index(start_id)
    end_index = boundary_loop.index(end_id)
    if start_index <= end_index:
        segment_a = boundary_loop[start_index : end_index + 1]
        segment_b = boundary_loop[end_index:] + boundary_loop[: start_index + 1]
    else:
        segment_a = boundary_loop[start_index:] + boundary_loop[: end_index + 1]
        segment_b = boundary_loop[end_index : start_index + 1]

    def polyline_length(vertex_ids: list[int]) -> float:
        if len(vertex_ids) < 2:
            return 0.0
        xyz = positions[np.asarray(vertex_ids, dtype=np.int32)]
        return float(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1).sum())

    def mean_v(vertex_ids: list[int]) -> float:
        xyz = positions[np.asarray(vertex_ids, dtype=np.int32)]
        return float(np.mean(xyz[:, axis_v])) if len(vertex_ids) else 0.0

    if prefer == "lower":
        segment_ids = segment_a if mean_v(segment_a) <= mean_v(segment_b) else list(reversed(segment_b))
    elif prefer == "upper":
        segment_ids = segment_a if mean_v(segment_a) >= mean_v(segment_b) else list(reversed(segment_b))
    else:
        segment_ids = (
            segment_a if polyline_length(segment_a) <= polyline_length(segment_b) else list(reversed(segment_b))
        )
    if segment_ids[0] != start_id:
        segment_ids = list(reversed(segment_ids))
    register_boundary_segment_crease(
        diagnostics,
        crease_name,
        tuple(int(vid) for vid in segment_ids),
        start_label,
        end_label,
    )


def resolve_boundary_curve_ref(diagnostics: dict[str, object], ref: str) -> str:
    crease_lookup = diagnostics["crease_lookup"]
    if ref in crease_lookup:
        return ref
    if ref.startswith("P") and len(ref) == 3 and ref[1:].isdigit():
        crease_id = f"C{ref[1:]}"
        if crease_id in crease_lookup:
            return crease_id
    raise AssertionError(f"failed to resolve boundary curve reference {ref}")


def register_boundary_intersection(
    diagnostics: dict[str, object],
    label: str,
    curve_ref_a: str,
    curve_ref_b: str,
) -> None:
    crease_lookup = diagnostics["crease_lookup"]
    curve_a = crease_lookup[resolve_boundary_curve_ref(diagnostics, curve_ref_a)]
    curve_b = crease_lookup[resolve_boundary_curve_ref(diagnostics, curve_ref_b)]
    intersections = sorted(
        set(int(vid) for vid in curve_a.vertex_ids).intersection(int(vid) for vid in curve_b.vertex_ids)
    )
    if intersections:
        register_boundary_point(diagnostics, label, int(intersections[0]))
        return

    positions = diagnostics["positions"]
    plane_axes = tuple(int(axis) for axis in diagnostics["plane_axes"])
    a0 = np.array(positions[int(curve_a.vertex_ids[0]), list(plane_axes)], copy=True)
    a1 = np.array(positions[int(curve_a.vertex_ids[-1]), list(plane_axes)], copy=True)
    b0 = np.array(positions[int(curve_b.vertex_ids[0]), list(plane_axes)], copy=True)
    b1 = np.array(positions[int(curve_b.vertex_ids[-1]), list(plane_axes)], copy=True)
    da = a1 - a0
    db = b1 - b0
    det = float(da[0] * db[1] - da[1] * db[0])
    if abs(det) < 1.0e-12:
        raise AssertionError(f"failed to find intersection {label} from {curve_ref_a} and {curve_ref_b}")
    rhs = b0 - a0
    t = float((rhs[0] * db[1] - rhs[1] * db[0]) / det)
    uv = a0 + t * da
    xyz = np.zeros(3, dtype=np.float64)
    xyz[list(plane_axes)] = uv
    normal_axis = int(diagnostics["normal_axis"])
    xyz[normal_axis] = float(positions[int(curve_a.vertex_ids[0]), normal_axis])
    vertex_id, _ = nearest_vertex(positions, xyz)
    register_boundary_point(diagnostics, label, int(vertex_id))


def register_crease_prefix_to_intersection(
    diagnostics: dict[str, object],
    chains: list[object],
    crease_id: str,
    intersection_key: str,
    segment_name: str,
    endpoint_name: str,
) -> None:
    chain = next((c for c in chains if c.crease_id == crease_id), None)
    if chain is None:
        return
    ids = tuple(int(vid) for vid in chain.vertex_ids)
    vid = int(diagnostics["point_lookup"][intersection_key])
    if vid in ids:
        idx = ids.index(vid)
        register_boundary_segment_crease(diagnostics, segment_name, ids[: idx + 1], endpoint_name, intersection_key)


def augment_boundary_diagnostics(diagnostics: dict[str, object]) -> None:
    positions = diagnostics["positions"]
    crease_lookup = diagnostics["crease_lookup"]
    chains = diagnostics["chains"]
    asset_stem = Path(str(diagnostics["asset_filename"])).stem.lower()

    lr_dn1 = crease_lookup["LR_DN1"]
    lr_dn1_ids = tuple(int(vid) for vid in lr_dn1.vertex_ids)
    lr_dn1_midpoint_vid = int(lr_dn1_ids[len(lr_dn1_ids) // 2])
    register_boundary_point(diagnostics, "P09_MIDPOINT", lr_dn1_midpoint_vid)
    register_boundary_segment_crease(
        diagnostics,
        "P09_MID_TO_B",
        lr_dn1_ids[len(lr_dn1_ids) // 2 :],
        "P09_MIDPOINT",
        "P09B",
    )

    c02_chain = next((chain for chain in chains if chain.crease_id == "C02"), None)
    if c02_chain is None:
        raise AssertionError("failed to find crease C02 for boundary diagnostics")
    c02_ids = set(int(vid) for vid in c02_chain.vertex_ids)
    p09_c02_intersections = sorted(set(lr_dn1_ids).intersection(c02_ids))
    if not p09_c02_intersections:
        raise AssertionError("failed to find the P09/C02 intersection")
    p09_c02_vid = int(p09_c02_intersections[0])
    register_boundary_point(diagnostics, "P09_C02_X", p09_c02_vid)
    p09_c02_start = lr_dn1_ids.index(p09_c02_vid)
    register_boundary_segment_crease(
        diagnostics,
        "P09_C02_TO_B",
        lr_dn1_ids[p09_c02_start:],
        "P09_C02_X",
        "P09B",
    )

    lr_dp1 = crease_lookup["LR_DP1"]
    lr_dp1_ids = tuple(int(vid) for vid in lr_dp1.vertex_ids)
    p13_c02_intersections = sorted(set(lr_dp1_ids).intersection(c02_ids))
    if not p13_c02_intersections:
        raise AssertionError("failed to find the P13/C02 intersection")
    p13_c02_vid = int(p13_c02_intersections[0])
    register_boundary_point(diagnostics, "P13_C02_X", p13_c02_vid)
    p13_c02_end = lr_dp1_ids.index(p13_c02_vid)
    register_boundary_segment_crease(
        diagnostics,
        "P13_A_TO_C02",
        lr_dp1_ids[: p13_c02_end + 1],
        "P13A",
        "P13_C02_X",
    )

    tr_dn1_ids = set(int(vid) for vid in crease_lookup["TR_DN1"].vertex_ids)
    intersections = sorted(tr_dn1_ids.intersection(set(lr_dp1_ids)))
    if not intersections:
        raise AssertionError("failed to find the P08/P13 intersection")
    if len(intersections) != 1:
        target = 0.5 * (
            np.array(positions[crease_lookup["TR_DN1"].endpoint_b], copy=True)
            + np.array(positions[crease_lookup["LR_DP1"].endpoint_b], copy=True)
        )
        vertex_id, _ = nearest_vertex(positions, target)
    else:
        vertex_id = int(intersections[0])
    register_boundary_point(diagnostics, "P08_P13_X", vertex_id)

    lt_dp2_ids = set(int(vid) for vid in crease_lookup["LT_DP2"].vertex_ids)
    p09_p12_intersections = sorted(set(lr_dn1_ids).intersection(lt_dp2_ids))
    if not p09_p12_intersections:
        raise AssertionError("failed to find the P09/P12 intersection")
    register_boundary_point(diagnostics, "P09_P12_X", int(p09_p12_intersections[0]))

    register_boundary_intersection(diagnostics, "C01_C05_X", "C01", "C05")
    register_boundary_intersection(diagnostics, "C01_P07_X", "C01", "P07")
    register_boundary_intersection(diagnostics, "C01_C04_X", "C01", "C04")
    register_boundary_intersection(diagnostics, "C04_C02_X", "C04", "C02")
    register_boundary_intersection(diagnostics, "P03_C05_X", "P03", "C05")
    register_boundary_intersection(diagnostics, "P03_P07_X", "P03", "P07")
    register_boundary_intersection(diagnostics, "C03_C04_X", "C03", "C04")
    register_boundary_intersection(diagnostics, "P08_C04_X", "P08", "C04")
    register_boundary_intersection(diagnostics, "C11_C04_X", "C11", "C04")
    boundary_loop = diagnostics["boundary_loop"]
    plane_axes = tuple(int(axis) for axis in diagnostics["plane_axes"])
    xmin, xmax, ymin, _ = diagnostics["bbox"]
    register_boundary_point(
        diagnostics,
        "LEFT_BOTTOM",
        nearest_boundary_vertex(positions, boundary_loop, plane_axes, xmin, ymin),
    )
    register_boundary_point(
        diagnostics,
        "RIGHT_BOTTOM",
        nearest_boundary_vertex(positions, boundary_loop, plane_axes, xmax, ymin),
    )

    c04_chain = crease_lookup["C04"]
    c04_ids = tuple(int(vid) for vid in c04_chain.vertex_ids)
    c01_c04_vid = int(diagnostics["point_lookup"]["C01_C04_X"])
    c03_c04_vid = int(diagnostics["point_lookup"]["C03_C04_X"])
    c04_mid_start = c04_ids.index(c01_c04_vid)
    c04_mid_end = c04_ids.index(c03_c04_vid)
    if c04_mid_start <= c04_mid_end:
        segment_ids = c04_ids[c04_mid_start : c04_mid_end + 1]
    else:
        segment_ids = tuple(reversed(c04_ids[c04_mid_end : c04_mid_start + 1]))
    register_boundary_segment_crease(
        diagnostics,
        "C01_C04_X_TO_C03_C04_X",
        segment_ids,
        "C01_C04_X",
        "C03_C04_X",
    )

    p08_c04_vid = int(diagnostics["point_lookup"]["P08_C04_X"])
    c11_c04_vid = int(diagnostics["point_lookup"]["C11_C04_X"])
    c04_start = c04_ids.index(p08_c04_vid)
    c04_end = c04_ids.index(c11_c04_vid)
    if c04_start <= c04_end:
        segment_ids = c04_ids[c04_start : c04_end + 1]
    else:
        segment_ids = tuple(reversed(c04_ids[c04_end : c04_start + 1]))
    register_boundary_segment_crease(
        diagnostics,
        "P08_C04_X_TO_C11_C04_X",
        segment_ids,
        "P08_C04_X",
        "C11_C04_X",
    )
    register_boundary_loop_segment_crease(
        diagnostics,
        "BOTTOM_EDGE",
        "LEFT_BOTTOM",
        "RIGHT_BOTTOM",
        prefer="lower",
    )

    register_crease_prefix_to_intersection(diagnostics, chains, "C01", "C01_C04_X", "C01_C04_X_TO_P01A", "P01A")
    register_crease_prefix_to_intersection(diagnostics, chains, "C03", "C03_C04_X", "C03_C04_X_TO_P03A", "P03A")

    if "paper_plane_5" in asset_stem:
        register_internal_row_crease(
            diagnostics,
            "C04_UPPER_SHORT",
            "C04_UPPER_SHORT_A",
            "C04_UPPER_SHORT_B",
            row_coord=1.18993,
            axis_min=-0.224075,
            axis_max=0.224075,
        )


# ---------------------------------------------------------------------------
# Asset loading (boundary only)
# ---------------------------------------------------------------------------


def load_asset_context(asset_filename: str | None = None) -> AssetContext:
    asset_filename = resolved_asset_filename(asset_filename)
    plane = boundary_debug.load_paper_plane_mesh(asset_filename)
    orient_mesh_for_ground(plane)
    diagnostics = boundary_debug.detect_boundary_creases(plane, asset_filename)
    augment_boundary_diagnostics(diagnostics)
    named_vertices = resolve_named_vertices(diagnostics)
    return AssetContext(
        mode="boundary",
        asset_filename=asset_filename,
        cube_mesh_path=default_cube_mesh_path(),
        plane=plane,
        diagnostics=diagnostics,
        named_vertices=named_vertices,
    )


# ---------------------------------------------------------------------------
# Cube setup
# ---------------------------------------------------------------------------


def compute_cube_home_xyz(diagnostics: dict[str, object]) -> np.ndarray:
    xmin, xmax, zmin, zmax = diagnostics["bbox"]
    return np.array(
        [
            xmax + CUBE_SCALE + CUBE_HOME_MARGIN_X,
            CUBE_HOVER_Y,
            zmin + CUBE_HOME_Z_RATIO * (zmax - zmin),
        ],
        dtype=np.float64,
    )


def compute_cube_setup(
    diagnostics: dict[str, object],
    overhead_big_cube: bool,
) -> tuple[float, np.ndarray, float, float, bool]:
    if overhead_big_cube:
        xmin, xmax, zmin, zmax = diagnostics["bbox"]
        home_xyz = np.array(
            [
                0.5 * (xmin + xmax),
                OVERHEAD_CUBE_HOVER_Y,
                0.5 * (zmin + zmax),
            ],
            dtype=np.float64,
        )
        return (
            OVERHEAD_CUBE_SCALE,
            home_xyz,
            OVERHEAD_CUBE_HOVER_Y,
            OVERHEAD_CUBE_PRESS_Y,
            True,
        )
    return (
        CUBE_SCALE,
        compute_cube_home_xyz(diagnostics),
        CUBE_HOVER_Y,
        CUBE_PRESS_Y,
        False,
    )


# ---------------------------------------------------------------------------
# Crease helpers
# ---------------------------------------------------------------------------


def crease_endpoint_labels(crease_name: str, diagnostics: dict[str, object]) -> tuple[str, str]:
    chain = diagnostics["crease_lookup"][crease_name]
    return str(chain.endpoint_a_name), str(chain.endpoint_b_name)


def crease_vertex_ids(crease_name: str, diagnostics: dict[str, object]) -> tuple[int, ...]:
    return tuple(int(vid) for vid in diagnostics["crease_lookup"][crease_name].vertex_ids)


def crease_midpoint_marker(
    crease_name: str,
    positions: np.ndarray,
    diagnostics: dict[str, object],
) -> NamedVertex:
    chain_ids = crease_vertex_ids(crease_name, diagnostics)
    midpoint_vid = chain_ids[len(chain_ids) // 2]
    label = f"{crease_name}_MIDPOINT"
    return NamedVertex(label=label, vertex_id=midpoint_vid, xyz=np.array(positions[midpoint_vid], copy=True))


# ---------------------------------------------------------------------------
# Named vertex / reflection helpers
# ---------------------------------------------------------------------------


def named_vertex_labels_by_id(named_vertices: dict[str, NamedVertex]) -> dict[int, str]:
    def label_rank(label: str) -> tuple[int, int, str]:
        if label in {
            "SRC",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_LEFT",
            "BOTTOM_RIGHT",
            "LEFT_TOP",
            "RIGHT_TOP",
            "LEFT_BOTTOM",
            "RIGHT_BOTTOM",
            "TOP",
            "BOTTOM",
        }:
            return (0, len(label), label)
        if label.endswith("_END") or label.endswith("_A") or label.endswith("_B"):
            return (1, len(label), label)
        if label.startswith("P") and len(label) >= 3 and label[1:3].isdigit():
            return (3, len(label), label)
        return (2, len(label), label)

    labels_by_id: dict[int, str] = {}
    for label, vertex in named_vertices.items():
        existing = labels_by_id.get(vertex.vertex_id)
        if existing is None or label_rank(label) < label_rank(existing):
            labels_by_id[vertex.vertex_id] = label
    return labels_by_id


def reflect_point_across_crease(
    point_xyz: np.ndarray,
    axis_start_xyz: np.ndarray,
    axis_end_xyz: np.ndarray,
) -> np.ndarray:
    axis_dir = normalize(axis_end_xyz - axis_start_xyz)
    relative = point_xyz - axis_start_xyz
    axial = float(np.dot(relative, axis_dir)) * axis_dir
    radial = relative - axial
    return axis_start_xyz + axial - radial


# ---------------------------------------------------------------------------
# Mover / angle helpers
# ---------------------------------------------------------------------------


def choose_upper_signed_angle(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    target_dir: np.ndarray,
    center_xyz: np.ndarray,
    axis_dir: np.ndarray,
    axial_offset_start: float,
    axial_offset_end: float,
    radius_start: float,
    radius_end: float,
) -> float:
    signed_angle = math.atan2(
        float(np.dot(target_dir, basis_v)),
        float(np.dot(target_dir, basis_u)),
    )
    if math.isclose(abs(signed_angle), math.pi, abs_tol=1.0e-6):
        probe_plus = (
            center_xyz
            + 0.5 * (axial_offset_start + axial_offset_end) * axis_dir
            + 0.5
            * (radius_start + radius_end)
            * (math.cos(0.5 * math.pi) * basis_u + math.sin(0.5 * math.pi) * basis_v)
        )
        return math.pi if probe_plus[1] >= center_xyz[1] else -math.pi
    midpoint_axis = center_xyz + 0.5 * (axial_offset_start + axial_offset_end) * axis_dir
    midpoint_radius = 0.5 * (radius_start + radius_end)
    midpoint_probe = midpoint_axis + midpoint_radius * (
        math.cos(0.5 * signed_angle) * basis_u + math.sin(0.5 * signed_angle) * basis_v
    )
    if midpoint_probe[1] < midpoint_axis[1]:
        return signed_angle - math.tau if signed_angle >= 0.0 else signed_angle + math.tau
    return signed_angle


def build_crease_line_mover_action(
    start: NamedVertex,
    target: NamedVertex,
    crease_name: str,
    axis_start_xyz: np.ndarray,
    axis_end_xyz: np.ndarray,
) -> MoverAction:
    axis_dir = normalize(axis_end_xyz - axis_start_xyz)
    start_rel = start.xyz - axis_start_xyz
    target_rel = target.xyz - axis_start_xyz
    axial_offset_start = float(np.dot(start_rel, axis_dir))
    axial_offset_end = float(np.dot(target_rel, axis_dir))
    start_radial = start_rel - axial_offset_start * axis_dir
    target_radial = target_rel - axial_offset_end * axis_dir
    radius_start = float(np.linalg.norm(start_radial))
    radius_end = float(np.linalg.norm(target_radial))
    if radius_start < 1.0e-8 or radius_end < 1.0e-8:
        raise AssertionError(f"{start.label}->{target.label} is too close to the {crease_name} axis")
    basis_u = normalize(start_radial)
    basis_v = normalize(np.cross(axis_dir, basis_u))
    target_dir = normalize(target_radial)
    signed_angle = choose_upper_signed_angle(
        basis_u,
        basis_v,
        target_dir,
        axis_start_xyz,
        axis_dir,
        axial_offset_start,
        axial_offset_end,
        radius_start,
        radius_end,
    )
    return MoverAction(
        label=start.label,
        vertex_id=start.vertex_id,
        start_xyz=np.array(start.xyz, copy=True),
        target_label=target.label,
        target_vertex_id=target.vertex_id,
        target_xyz=np.array(target.xyz, copy=True),
        center_label=f"{crease_name}_AXIS",
        center_xyz=np.array(axis_start_xyz, copy=True),
        axis_dir=np.array(axis_dir, copy=True),
        axial_offset_start=axial_offset_start,
        axial_offset_end=axial_offset_end,
        radius_start=radius_start,
        radius_end=radius_end,
        basis_u=np.array(basis_u, copy=True),
        basis_v=np.array(basis_v, copy=True),
        signed_angle=signed_angle,
    )


# ---------------------------------------------------------------------------
# Press helpers
# ---------------------------------------------------------------------------


def press_endpoint_sort_key(vertex: NamedVertex) -> tuple[float, float, str]:
    return (float(vertex.xyz[0]), float(vertex.xyz[2]), vertex.label)


def press_cube_yaw_radians(left_xyz: np.ndarray, right_xyz: np.ndarray) -> float:
    crease_dir = np.asarray(right_xyz, dtype=np.float64) - np.asarray(left_xyz, dtype=np.float64)
    crease_dir[1] = 0.0
    norm = float(np.linalg.norm(crease_dir))
    if norm < 1.0e-12:
        return 0.0
    crease_dir /= norm
    return math.atan2(float(crease_dir[0]), float(crease_dir[2]))


def build_stamped_press_path(
    crease_name: str,
    positions: np.ndarray,
    diagnostics: dict[str, object],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[np.ndarray, ...]]:
    chain_ids = crease_vertex_ids(crease_name, diagnostics)
    endpoint_labels = crease_endpoint_labels(crease_name, diagnostics)
    if len(chain_ids) < 2:
        vertex_id = int(chain_ids[0])
        xyz = np.array(positions[vertex_id], copy=True)
        return ((endpoint_labels[0],), (vertex_id,), (xyz,))

    chain_xyz = positions[np.asarray(chain_ids, dtype=np.int32)]
    edge_vectors = chain_xyz[1:] - chain_xyz[:-1]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    total_length = float(np.sum(edge_lengths))
    if total_length < 1.0e-12:
        start_id = int(chain_ids[0])
        end_id = int(chain_ids[-1])
        return (
            (endpoint_labels[0], endpoint_labels[1]),
            (start_id, end_id),
            (np.array(positions[start_id], copy=True), np.array(positions[end_id], copy=True)),
        )

    stride = CUBE_SCALE
    distances = [0.0]
    distance = stride
    while distance < total_length - 1.0e-8:
        distances.append(distance)
        distance += stride
    if not math.isclose(distances[-1], total_length, abs_tol=1.0e-8):
        distances.append(total_length)

    labels: list[str] = []
    vertex_ids: list[int] = []
    xyzs: list[np.ndarray] = []
    segment_index = 0
    segment_start_length = 0.0
    for index, target_distance in enumerate(distances):
        while (
            segment_index < len(edge_lengths) - 1
            and segment_start_length + edge_lengths[segment_index] < target_distance - 1.0e-12
        ):
            segment_start_length += float(edge_lengths[segment_index])
            segment_index += 1
        local_length = float(target_distance - segment_start_length)
        segment_length = float(edge_lengths[segment_index])
        if segment_length < 1.0e-12:
            xyz = np.array(chain_xyz[segment_index], copy=True)
        else:
            t = np.clip(local_length / segment_length, 0.0, 1.0)
            xyz = chain_xyz[segment_index] + t * edge_vectors[segment_index]
        if index == 0:
            labels.append(endpoint_labels[0])
            vertex_ids.append(int(chain_ids[0]))
        elif index == len(distances) - 1:
            labels.append(endpoint_labels[1])
            vertex_ids.append(int(chain_ids[-1]))
        else:
            labels.append(f"{crease_name}_STAMP_{index:02d}")
            vertex_ids.append(-1)
        xyzs.append(np.array(xyz, copy=True))
    return tuple(labels), tuple(vertex_ids), tuple(xyzs)


def build_press_action(
    step_spec: StepSpec,
    positions: np.ndarray,
    named_vertices: dict[str, NamedVertex],
    diagnostics: dict[str, object],
    home_xyz: np.ndarray,
    hover_y: float,
    press_y: float,
    vertical_only: bool,
) -> PressAction | None:
    if step_spec.press is None:
        return None
    if vertical_only:
        center_xyz = np.array([home_xyz[0], press_y, home_xyz[2]], dtype=np.float64)
        return PressAction(
            mode=step_spec.press.mode,
            crease_name=step_spec.crease_name,
            home_xyz=np.array(home_xyz, copy=True),
            hover_y=hover_y,
            press_y=press_y,
            vertical_only=True,
            path_labels=("OVERHEAD_PRESS",),
            path_vertex_ids=(-1,),
            path_xyzs=(center_xyz,),
            cube_yaw_radians=0.0,
        )
    endpoint_labels = crease_endpoint_labels(step_spec.crease_name, diagnostics)
    endpoint_vertices = [named_vertices[label] for label in endpoint_labels]
    left_vertex, right_vertex = sorted(endpoint_vertices, key=press_endpoint_sort_key)
    midpoint_vertex = crease_midpoint_marker(step_spec.crease_name, positions, diagnostics)
    cube_yaw = press_cube_yaw_radians(left_vertex.xyz, right_vertex.xyz)
    if step_spec.press.mode == "stamped":
        path_labels, path_vertex_ids, path_xyzs = build_stamped_press_path(
            step_spec.crease_name,
            positions,
            diagnostics,
        )
    elif step_spec.press.path_labels is None:
        path_vertices = (midpoint_vertex, left_vertex, right_vertex)
        path_labels = tuple(vertex.label for vertex in path_vertices)
        path_vertex_ids = tuple(vertex.vertex_id for vertex in path_vertices)
        path_xyzs = tuple(np.array(vertex.xyz, copy=True) for vertex in path_vertices)
    else:
        resolved_path: list[NamedVertex] = []
        for label in step_spec.press.path_labels:
            if label == f"{step_spec.crease_name}_MIDPOINT":
                resolved_path.append(midpoint_vertex)
            else:
                resolved_path.append(named_vertices[label])
        path_vertices = tuple(resolved_path)
        path_labels = tuple(vertex.label for vertex in path_vertices)
        path_vertex_ids = tuple(vertex.vertex_id for vertex in path_vertices)
        path_xyzs = tuple(np.array(vertex.xyz, copy=True) for vertex in path_vertices)
    return PressAction(
        mode=step_spec.press.mode,
        crease_name=step_spec.crease_name,
        home_xyz=np.array(home_xyz, copy=True),
        hover_y=hover_y,
        press_y=press_y,
        vertical_only=False,
        path_labels=path_labels,
        path_vertex_ids=path_vertex_ids,
        path_xyzs=path_xyzs,
        cube_yaw_radians=cube_yaw,
    )


def build_global_flip_action(
    step_spec: StepSpec,
    named_vertices: dict[str, NamedVertex],
    diagnostics: dict[str, object],
) -> GlobalFlipAction | None:
    if step_spec.global_flip is None:
        return None
    axis_name = step_spec.global_flip.axis_name
    endpoint_labels = crease_endpoint_labels(axis_name, diagnostics)
    axis_start = named_vertices[endpoint_labels[0]]
    axis_end = named_vertices[endpoint_labels[1]]
    axis_dir = normalize(axis_end.xyz - axis_start.xyz)
    mover_labels = step_spec.global_flip.mover_labels
    if mover_labels is None:
        mover_vertex_ids = tuple(sorted({int(vertex.vertex_id) for vertex in named_vertices.values()}))
    else:
        mover_vertex_ids = tuple(sorted({int(named_vertices[label].vertex_id) for label in mover_labels}))
    return GlobalFlipAction(
        group_name=step_spec.global_flip.group_name or step_spec.name,
        phase=step_spec.global_flip.phase,
        axis_name=axis_name,
        axis_start_label=axis_start.label,
        axis_end_label=axis_end.label,
        axis_start_vertex_id=int(axis_start.vertex_id),
        axis_end_vertex_id=int(axis_end.vertex_id),
        axis_start_xyz=np.array(axis_start.xyz, copy=True),
        axis_end_xyz=np.array(axis_end.xyz, copy=True),
        axis_dir=np.array(axis_dir, copy=True),
        lift_y=float(step_spec.global_flip.lift_y),
        angle_radians=math.radians(float(step_spec.global_flip.angle_degrees)),
        mover_vertex_ids=mover_vertex_ids,
    )


def _add_anchor(
    anchors: list[AnchorAction],
    ids_set: set[int],
    vertex_id: int,
    label: str,
    xyz: np.ndarray,
    lock_current_pose: bool = True,
    current_pose_y: float | None = None,
) -> None:
    if vertex_id in ids_set:
        return
    ids_set.add(vertex_id)
    anchors.append(AnchorAction(label, vertex_id, np.array(xyz, copy=True), lock_current_pose, current_pose_y))


def compile_step(
    step_spec: StepSpec,
    positions: np.ndarray,
    base_named_vertices: dict[str, NamedVertex],
    diagnostics: dict[str, object],
    home_xyz: np.ndarray,
    hover_y: float,
    press_y: float,
    vertical_only: bool,
) -> FoldStep:
    named_vertices = {
        label: NamedVertex(
            label=vertex.label,
            vertex_id=vertex.vertex_id,
            xyz=(
                np.array(positions[vertex.vertex_id], copy=True)
                if vertex.vertex_id >= 0
                else np.array(vertex.xyz, copy=True)
            ),
        )
        for label, vertex in base_named_vertices.items()
    }
    endpoint_labels = crease_endpoint_labels(step_spec.crease_name, diagnostics)
    axis_start = named_vertices[endpoint_labels[0]]
    axis_end = named_vertices[endpoint_labels[1]]
    midpoint_vertex = crease_midpoint_marker(step_spec.crease_name, positions, diagnostics)
    axis_start_xyz = np.array(axis_start.xyz, copy=True)
    axis_end_xyz = np.array(axis_end.xyz, copy=True)
    global_flip = build_global_flip_action(step_spec, named_vertices, diagnostics)
    passive_step = global_flip is not None or step_spec.free_run_frames > 0

    weak_anchor_ids: set[int] = set()
    weak_anchor_actions: list[AnchorAction] = []
    if not passive_step:
        for vertex in (axis_start, axis_end, midpoint_vertex):
            _add_anchor(weak_anchor_actions, weak_anchor_ids, vertex.vertex_id, vertex.label, vertex.xyz)
    weak_anchors = tuple(weak_anchor_actions)

    labels_by_id = named_vertex_labels_by_id(named_vertices)
    strong_fix_ids: set[int] = set()
    strong_fixes: list[AnchorAction] = []
    for label in step_spec.strong_fix_labels:
        vertex = named_vertices[label]
        _add_anchor(
            strong_fixes, strong_fix_ids, vertex.vertex_id, label, vertex.xyz, current_pose_y=STRONG_FIX_CURRENT_Y
        )
    for crease_name in step_spec.strong_fix_crease_names:
        for index, vertex_id in enumerate(crease_vertex_ids(crease_name, diagnostics)):
            label = labels_by_id.get(vertex_id, f"{crease_name}_FIX_{index:02d}")
            _add_anchor(strong_fixes, strong_fix_ids, vertex_id, label, positions[vertex_id])
    for start_label, end_label in step_spec.strong_fix_segment_labels:
        segment_ids = sample_segment_vertex_ids(
            positions,
            named_vertices,
            start_label,
            end_label,
            spacing=0.25 * CUBE_SCALE,
        )
        for index, vertex_id in enumerate(segment_ids):
            label = labels_by_id.get(vertex_id, f"{start_label}_TO_{end_label}_FIX_{index:02d}")
            _add_anchor(strong_fixes, strong_fix_ids, vertex_id, label, positions[vertex_id])

    targets: list[NamedVertex] = []
    movers: list[MoverAction] = []
    if not passive_step:
        for mover_label in step_spec.mover_labels:
            mover_vertex = named_vertices[mover_label]
            reflected_xyz = reflect_point_across_crease(mover_vertex.xyz, axis_start_xyz, axis_end_xyz)
            target_vertex_id, target_distance = nearest_vertex(positions, reflected_xyz)
            if target_distance <= TARGET_SNAP_TOL:
                target_label = labels_by_id.get(target_vertex_id, f"SYM_{mover_label}")
                target = NamedVertex(
                    label=target_label,
                    vertex_id=target_vertex_id,
                    xyz=np.array(positions[target_vertex_id], copy=True),
                )
                _add_anchor(
                    strong_fixes, strong_fix_ids, target.vertex_id, target.label, target.xyz, lock_current_pose=False
                )
            else:
                target = NamedVertex(
                    label=f"TARGET_{mover_label}",
                    vertex_id=-1,
                    xyz=np.array(reflected_xyz, copy=True),
                )
            targets.append(target)
            movers.append(
                build_crease_line_mover_action(
                    mover_vertex,
                    target,
                    step_spec.crease_name,
                    axis_start_xyz,
                    axis_end_xyz,
                )
            )

    press = build_press_action(
        step_spec,
        positions,
        named_vertices,
        diagnostics,
        home_xyz,
        hover_y,
        press_y,
        vertical_only,
    )
    return FoldStep(
        name=step_spec.name,
        crease_name=step_spec.crease_name,
        weak_anchors=weak_anchors,
        strong_fixes=tuple(strong_fixes),
        movers=tuple(movers),
        targets=tuple(targets),
        press=press,
        free_run_frames=int(step_spec.free_run_frames),
        global_flip=global_flip,
        crease_vertex_ids=crease_vertex_ids(step_spec.crease_name, diagnostics),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_asset_context(context: AssetContext) -> None:
    diagnostics = context.diagnostics
    positions = diagnostics["positions"]
    named = context.named_vertices
    xmin, xmax, ymin, ymax = diagnostics["bbox"]
    expected = {
        "LEFT_TOP": np.array([xmin, ymax]),
        "RIGHT_TOP": np.array([xmax, ymax]),
        "LEFT_BOTTOM": np.array([xmin, ymin]),
        "RIGHT_BOTTOM": np.array([xmax, ymin]),
        "TOP": np.array([0.5 * (xmin + xmax), ymax]),
        "BOTTOM": np.array([0.5 * (xmin + xmax), ymin]),
    }
    plane_axes = tuple(int(axis) for axis in diagnostics["plane_axes"])
    for label, expected_uv in expected.items():
        actual_uv = np.array(named[label].xyz[list(plane_axes)], copy=True)
        if np.linalg.norm(actual_uv - expected_uv) > GEOMETRY_MATCH_TOL:
            raise AssertionError(f"{label} resolved to an unexpected boundary vertex")
    for required in ("LT_DP2_A", "LT_DP2_B"):
        if required not in named:
            raise AssertionError(f"boundary asset is missing {required}")
    if not os.path.exists(context.cube_mesh_path):
        raise AssertionError(f"cube mesh not found: {context.cube_mesh_path}")
    if positions.shape[1] != 3:
        raise AssertionError("positions must be 3D")


# ---------------------------------------------------------------------------
# Overlay / visualization helpers
# ---------------------------------------------------------------------------


def build_curve_overlay(max_nodes: int) -> dict[str, np.ndarray]:
    edges = np.asarray([[i, i + 1] for i in range(max(max_nodes - 1, 0))], dtype=np.int32)
    return {
        "positions": np.zeros((max_nodes, 3), dtype=np.float64),
        "edges": edges,
    }


def build_pair_overlay(max_pairs: int) -> dict[str, np.ndarray]:
    nodes = np.zeros((2 * max_pairs, 3), dtype=np.float64)
    edges = np.asarray([[2 * i, 2 * i + 1] for i in range(max_pairs)], dtype=np.int32)
    return {
        "positions": nodes,
        "edges": edges,
    }


def fill_cloud_positions(items: list[np.ndarray], count: int) -> np.ndarray:
    points = np.zeros((count, 3), dtype=np.float64)
    if not items:
        return points
    points[: len(items)] = np.asarray(items, dtype=np.float64)
    for index in range(len(items), count):
        points[index] = items[0]
    return points


# ---------------------------------------------------------------------------
# OBJ frame export
# ---------------------------------------------------------------------------


def frame_obj_directory(workspace: str) -> str:
    path = os.path.join(workspace, "obj_frames")
    os.makedirs(path, exist_ok=True)
    return path


def write_obj_frame(
    output_path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    extra_meshes: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("# paper_plane_fold_sequence frame export\n")
        outfile.write("o paper\n")
        for vertex in vertices:
            outfile.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for tri in triangles:
            outfile.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")
        if extra_meshes:
            vertex_offset = len(vertices)
            for mesh_index, (extra_verts, extra_tris) in enumerate(extra_meshes):
                outfile.write(f"o extra_{mesh_index}\n")
                for vertex in extra_verts:
                    outfile.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                for tri in extra_tris:
                    outfile.write(
                        f"f {int(tri[0]) + 1 + vertex_offset} "
                        f"{int(tri[1]) + 1 + vertex_offset} "
                        f"{int(tri[2]) + 1 + vertex_offset}\n"
                    )
                vertex_offset += len(extra_verts)
