"""Current material-cap loads with physical side-dependent nodal couples."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import roots_jacobi

from .geometry import normalize


@dataclass(frozen=True)
class Source:
    direction: np.ndarray
    radius: float
    force: np.ndarray | None = None
    normal_force: float = 1.0
    friction: float = 0.0
    tangent: np.ndarray | None = None
    side: int = 1
    name: str = ""
    cap_tangent: np.ndarray | None = None


@lru_cache(maxsize=8)
def cap_rule(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("Quadrature order must be positive.")
    nodes, weights = roots_jacobi(order, 0.5, 0.0)
    radius = np.sqrt(0.5 * (nodes + 1.0))
    angle = 2 * np.pi * (np.arange(4 * order) + 0.5) / (4 * order)
    radial = np.repeat(radius, 4 * order)
    angular = np.tile(angle, order)
    normalized_weights = np.repeat(weights / np.sum(weights) / (4 * order), 4 * order)
    return radial, angular, normalized_weights


def tangent_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    first = normalize(np.cross(direction, axis))
    return first, np.cross(direction, first)


def source_directions(source: Source, order: int = 8) -> tuple[np.ndarray, np.ndarray]:
    center = normalize(source.direction)
    if source.cap_tangent is None:
        first, second = tangent_basis(center)
    else:
        first = np.asarray(source.cap_tangent, dtype=float)
        first = normalize(first - np.dot(first, center) * center)
        second = np.cross(center, first)
    radial, angular, weights = cap_rule(order)
    theta = source.radius / 0.022 * radial
    tangent = np.cos(angular)[:, None] * first + np.sin(angular)[:, None] * second
    directions = np.cos(theta)[:, None] * center + np.sin(theta)[:, None] * tangent
    return directions, weights


def force_for_source(mother, source: Source) -> np.ndarray:
    if source.force is not None:
        force = np.asarray(source.force, dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("Explicit source force must be a finite three-vector.")
        return force
    normal = mother.geometry.locate(np.atleast_2d(source.direction)).normals[0] * source.side
    if source.tangent is None:
        first, _ = tangent_basis(normal)
    else:
        first = np.asarray(source.tangent, dtype=float)
        first = normalize(first - np.dot(first, normal) * normal)
    return source.normal_force * (-normal + source.friction * first)


def assemble_sources(mother, sources: list[Source] | tuple[Source, ...], order: int = 8,
                     audit: bool = False) -> np.ndarray:
    """Assemble raw load columns before applying the common mother relief."""
    result = np.zeros((mother.n_dof, len(sources)))
    for i_source, source in enumerate(sources):
        if source.radius < 0 or not np.isfinite(source.radius):
            raise ValueError("Source radius must be finite and nonnegative.")
        if source.side not in (-1, 1):
            raise ValueError("Surface side must be -1 or +1.")
        if source.normal_force < 0 or source.friction < 0:
            raise ValueError("Normal force and friction must be nonnegative.")
        force = force_for_source(mother, source)
        directions, weights = source_directions(source, order)
        location = mother.geometry.locate(directions)
        node_weights = location.weights * weights[:, None]
        nodal = node_weights[:, :, None] * force
        directors = mother.geometry.directors[location.nodes]
        couples = source.side * 0.5 * mother.model.h * np.cross(directors, nodal)
        for component in range(3):
            np.add.at(result[:, i_source], (6 * location.nodes + component).reshape(-1), nodal[:, :, component].reshape(-1))
            np.add.at(result[:, i_source], (6 * location.nodes + 3 + component).reshape(-1), couples[:, :, component].reshape(-1))
        if audit:
            recovered_force = result[:, i_source].reshape((-1, 6))[:, :3].sum(axis=0)
            physical_points = location.positions + source.side * 0.5 * mother.model.h * np.einsum(
                "si,sij->sj", location.weights, directors)
            expected_moment = np.sum(np.cross(physical_points, force) * weights[:, None], axis=0)
            node_loads = result[:, i_source].reshape((-1, 6))
            recovered_moment = (np.cross(mother.geometry.vertices, node_loads[:, :3]) + node_loads[:, 3:]).sum(axis=0)
            if not np.allclose(recovered_force, force, rtol=2e-13, atol=2e-15):
                raise ArithmeticError("Physical source resultant force was not preserved.")
            if not np.allclose(recovered_moment, expected_moment, rtol=2e-12, atol=2e-16):
                raise ArithmeticError("Physical source resultant moment was not preserved.")
    return result
