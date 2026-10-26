"""Whole-surface feature libraries and continuous source-local selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from .fixtures import fibonacci_directions, unit


@dataclass(frozen=True)
class FeatureBank:
    centers: np.ndarray
    fill_angle: float
    support_angle: float
    core_angle: float
    spaces: tuple
    narrow_fraction: float = 0.12
    maximum_radius: float = 0.003

    def select(self, source, strategy: str = "continuous") -> tuple[np.ndarray, np.ndarray]:
        if source.radius > self.maximum_radius * (1.0 + 1e-12):
            raise ValueError("The prepared feature library covers radii through 3 nominal mm.")
        weights = selection_weights(self.centers, source.direction, self.support_angle, strategy, self.narrow_fraction)
        indices = np.flatnonzero(weights > 0.0)
        return indices, weights[indices]


def icosahedron_centers() -> np.ndarray:
    golden = 0.5 * (1.0 + np.sqrt(5.0))
    centers = []
    for a in (-1.0, 1.0):
        for b in (-golden, golden):
            centers.extend(((0.0, a, b), (a, b, 0.0), (b, 0.0, a)))
    return unit(np.array(sorted(centers)))


def covering_angle(centers: np.ndarray) -> float:
    hull = ConvexHull(centers)
    vertices = unit(hull.equations[:, :3])
    nearest = np.max(vertices @ centers.T, axis=1)
    return float(np.max(np.arccos(np.clip(nearest, -1.0, 1.0))))


def library_centers(count: int = 12) -> np.ndarray:
    if count == 12:
        return icosahedron_centers()
    if count < 4:
        raise ValueError("A spherical library needs at least four centers.")
    return fibonacci_directions(count)


def wendland(distance: np.ndarray) -> np.ndarray:
    t = np.maximum(1.0 - distance, 0.0)
    return t ** 4 * (1.0 + 4.0 * distance)


def selection_weights(
    centers: np.ndarray, direction: np.ndarray, support_angle: float,
    strategy: str = "continuous", narrow_fraction: float = 0.12,
) -> np.ndarray:
    direction = unit(direction)
    angles = np.arccos(np.clip(centers @ direction, -1.0, 1.0))
    if strategy == "nearest":
        weights = np.zeros(len(centers))
        weights[np.argmin(angles)] = 1.0
        return weights
    weights = wendland(angles / support_angle)
    if strategy == "narrow":
        gap = (angles - np.min(angles)) / (narrow_fraction * support_angle)
        weights *= wendland(gap)
    elif strategy == "top2":
        keep = np.argsort(angles, kind="stable")[:2]
        weights[np.setdiff1d(np.arange(len(centers)), keep)] = 0.0
    elif strategy != "continuous":
        raise ValueError("Selection strategy must be continuous, nearest, narrow or top2.")
    denominator = np.sum(weights)
    if denominator <= 0.0:
        raise ValueError("The current source is outside the prepared selection support.")
    return weights / denominator


def build_bank(mother, count: int = 12) -> FeatureBank:
    centers = library_centers(count)
    fill_angle = covering_angle(centers)
    support_angle = 1.2 * fill_angle
    core_angle = support_angle + 0.003 / 0.022
    spaces = tuple(mother.feature(center, core_angle, halo=1, name=f"feature-{i:02d}")
                   for i, center in enumerate(centers))
    return FeatureBank(centers, fill_angle, support_angle, core_angle, spaces)


def field_weights(mother) -> np.ndarray:
    point = np.tile(mother.model.gauss_weights.reshape(-1), 2)
    return np.sqrt(np.repeat(point, 3) * np.tile((1.0, 1.0, 2.0), len(point)))


def relative_field_error(value: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    tail = (None,) * (value.ndim - 1)
    scale = weights[(slice(None),) + tail]
    numerator = np.linalg.norm(scale * (value - reference), axis=0)
    denominator = np.maximum(np.linalg.norm(scale * reference, axis=0), 1e-30)
    return numerator / denominator


def sampled_maximum(value: np.ndarray) -> np.ndarray:
    local = value.reshape((-1, 3) + value.shape[1:])
    xx, yy, xy = local[:, 0], local[:, 1], local[:, 2]
    mises = np.sqrt(np.maximum(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy, 0.0))
    return np.max(mises, axis=0)
