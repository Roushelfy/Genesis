"""Utilities for loading trajectory recordings and converting them for the URDF GUI.

Handles:
- Downloading ``marvin_sharpa`` URDF + meshes from HuggingFace
- Loading ``trajectory.npz`` data
- Building the Genesis-to-yourdfpy joint mapping
- Converting Genesis-order qpos arrays to URDFController-compatible dicts/schedules
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from urdf_controller import URDFController


# ---------------------------------------------------------------------------
# HuggingFace asset download
# ---------------------------------------------------------------------------


def get_sharpa_urdf() -> str:
    """Download the full ``marvin_sharpa_description`` folder and return the URDF path."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id="Genesis-Intelligence/internal_assets",
        repo_type="dataset",
        allow_patterns="marvin_sharpa_description/**",
    )
    return os.path.join(local_dir, "marvin_sharpa_description", "marvin_sharpa.urdf")


# ---------------------------------------------------------------------------
# Trajectory data container
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryData:
    """Container for data loaded from ``trajectory.npz``."""

    sim_times: np.ndarray  # (N,)
    robot_qpos: np.ndarray  # (N, D)
    yoyo_pos: np.ndarray | None = None  # (N, 3)
    yoyo_quat: np.ndarray | None = None  # (N, 4)
    string_particles: np.ndarray | None = None  # (N, V, 3)
    ball_mesh_path: str | None = None
    string_mesh_path: str | None = None
    n_frames: int = 0
    dt: float = 0.0
    traj_dir: Path = field(default_factory=Path)


def load_trajectory_npz(traj_dir: Path | str) -> TrajectoryData:
    """Load a ``trajectory.npz`` from *traj_dir* and return a :class:`TrajectoryData`."""
    traj_dir = Path(traj_dir)
    npz_path = traj_dir / "trajectory.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"trajectory.npz not found in {traj_dir}")
    data = np.load(npz_path)

    sim_times = data["sim_times"]
    robot_qpos = data["robot_qpos"]
    n = len(sim_times)
    dt = float(sim_times[1] - sim_times[0]) if n > 1 else 1.0 / 60.0

    ball_mesh = traj_dir / "yoyo_ball.obj"
    string_mesh = traj_dir / "yoyo_string.obj"

    return TrajectoryData(
        sim_times=sim_times,
        robot_qpos=robot_qpos,
        yoyo_pos=data.get("yoyo_pos"),
        yoyo_quat=data.get("yoyo_quat"),
        string_particles=data.get("yoyo_string_particles"),
        ball_mesh_path=str(ball_mesh) if ball_mesh.exists() else None,
        string_mesh_path=str(string_mesh) if string_mesh.exists() else None,
        n_frames=n,
        dt=dt,
        traj_dir=traj_dir,
    )


# ---------------------------------------------------------------------------
# Genesis joint-order mapping
# ---------------------------------------------------------------------------

_MAPPING_FILE_NAME = "genesis_joint_order.json"


def load_genesis_joint_order(search_dirs: list[Path | str]) -> list[str]:
    """Load the Genesis qpos joint ordering from a pre-generated JSON file.

    Searches *search_dirs* in order for ``genesis_joint_order.json``.
    Generate this file once by running::

        python build_joint_mapping.py

    Raises ``FileNotFoundError`` if no mapping file is found.
    """
    for d in search_dirs:
        candidate = Path(d) / _MAPPING_FILE_NAME
        if candidate.exists():
            names = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                print(f"[mapping] Loaded Genesis joint order from {candidate}")
                return names

    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"genesis_joint_order.json not found in [{searched}]. "
        f"Run 'python build_joint_mapping.py' first to generate it."
    )


# ---------------------------------------------------------------------------
# Joint mapping helpers
# ---------------------------------------------------------------------------


def qpos_to_joint_dict(
    qpos_row: np.ndarray,
    genesis_joint_names: list[str],
) -> dict[str, float]:
    """Convert a Genesis-ordered qpos vector to ``{joint_name: value}``.

    *genesis_joint_names* is the list returned by :func:`get_genesis_joint_order`
    so that ``qpos_row[i]`` maps to ``genesis_joint_names[i]``.
    """
    return {
        name: float(qpos_row[i])
        for i, name in enumerate(genesis_joint_names)
        if name  # skip empty slots (fixed joints etc.)
    }


def trajectory_to_frame_schedule(
    robot_qpos: np.ndarray,
    genesis_joint_names: list[str],
    traj_dt: float = 1.0 / 60.0,
    sim_dt: float = 0.001,
) -> list[dict]:
    """Interpolate ``(N, D)`` qpos at *sim_dt* resolution for the UIPC schedule.

    The trajectory was recorded at *traj_dt* intervals.  Between each pair of
    consecutive trajectory frames, this function generates linearly-interpolated
    sub-frames so the schedule ticks at *sim_dt*.

    Each entry is ``{"joints": {name: val, ...}}``.
    """
    n_traj = robot_qpos.shape[0]
    if n_traj == 0:
        return []

    steps_per_frame = max(1, int(np.round(traj_dt / sim_dt)))

    schedule: list[dict] = []
    schedule.append({"joints": qpos_to_joint_dict(robot_qpos[0], genesis_joint_names)})

    for i in range(1, n_traj):
        src = robot_qpos[i - 1]
        dst = robot_qpos[i]
        for s in range(1, steps_per_frame + 1):
            alpha = s / steps_per_frame
            interp = src + alpha * (dst - src)
            schedule.append({"joints": qpos_to_joint_dict(interp, genesis_joint_names)})

    return schedule


# ---------------------------------------------------------------------------
# Quaternion / transform helpers (for overview-mode yoyo display)
# ---------------------------------------------------------------------------


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion ``(w, x, y, z)`` to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
