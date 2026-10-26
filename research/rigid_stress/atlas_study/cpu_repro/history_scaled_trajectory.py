"""Shared preparation for the separately declared smooth-time control."""
from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np

from .history_speed_study import scaled_sources
from .history_study import Trajectory, current_force, prepare_trajectory
from .loads import assemble_sources
from .model import Mother


SPEED_DEFINITION = (
    "Smooth contact direction, force, friction and radius use time=speed*frame; "
    "release/reset/shock event frames remain fixed. Different speeds cover different "
    "total smooth paths in 80 frames, rather than the same path in different times."
)


def scaled_current_force(
    mother: Mother, frame: int, batch: int, order: int, speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if speed == 1.0:
        return current_force(mother, frame, batch, order)
    sources, active, resets = scaled_sources(frame, batch, speed)
    raw = assemble_sources(mother, sources, order=order, audit=False)
    full = np.zeros((mother.n_dof, batch))
    full[:, active] = raw[:, 0::2] + raw[:, 1::2]
    return mother.balance(full)[mother.free], active, resets


def prepare_scaled_trajectory(
    mother: Mother, speed: float, batch: int, frames: int = 80, order: int = 8,
) -> Trajectory:
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("Smooth-time speed must be finite and positive.")
    if speed == 1.0:
        return prepare_trajectory(mother, batch, frames=frames, order=order)
    start = perf_counter()
    forces = np.empty((frames, len(mother.free), batch))
    active = np.empty((frames, batch), dtype=bool)
    resets = []
    for frame in range(frames):
        forces[frame], active[frame], reset = scaled_current_force(mother, frame, batch, order, speed)
        resets.append(reset)
    return Trajectory(forces, active, tuple(resets), perf_counter() - start,
                      hashlib.sha256(forces.tobytes()).hexdigest(), order)
