"""Deterministic moving material-cap loads for the reconstructed shell study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .loads import Source


@dataclass(frozen=True)
class Case:
    name: str
    sources: tuple[Source, ...]
    family: str


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)


def fibonacci_directions(count: int, phase: float = 0.0) -> np.ndarray:
    index = np.arange(count)
    z = 1.0 - 2.0 * (index + 0.5) / count
    angle = (index + phase) * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def make_cases() -> tuple[Case, ...]:
    """Build a frozen reconstruction corpus with 42 loaded cases and release."""
    cases = []
    for i, direction in enumerate(fibonacci_directions(24, phase=0.37)):
        source = Source(
            direction=direction,
            radius=0.001 * (1 + i % 3),
            normal_force=0.1 + 0.02 * np.sin(0.71 * i),
            friction=(0.0, 0.3, 0.6)[i % 3],
            name=f"surface-{i:02d}",
        )
        cases.append(Case(source.name, (source,), "independent_surface"))
    special = (
        ("north-pole", (0.0, 0.0, 1.0)),
        ("south-pole", (0.0, 0.0, -1.0)),
        ("equator-seam", (1.0, 1.0, 0.0)),
        ("lower-seam", (1.0, 0.0, -1.0)),
        ("upper-corner", (1.0, 1.0, 1.0)),
        ("lower-corner", (-1.0, 1.0, -1.0)),
    )
    for i, (name, direction) in enumerate(special):
        source = Source(
            direction=unit(direction), radius=0.001 * (1 + i % 3), normal_force=0.08 + 0.01 * i,
            friction=(0.0, 0.3, 0.6)[i % 3], name=name,
        )
        cases.append(Case(name, (source,), "pole_seam_corner"))
    far = (
        Source(unit((1.0, 0.23, 0.12)), 0.002, normal_force=0.12, friction=0.3, name="far-a"),
        Source(unit((-1.0, -0.18, 0.31)), 0.003, normal_force=0.085, friction=0.6, name="far-b"),
    )
    near = (
        Source(unit((0.18, 1.0, 0.24)), 0.001, normal_force=0.11, friction=0.6, name="near-a"),
        Source(unit((0.20, 1.0, 0.26)), 0.002, normal_force=0.07, friction=0.3, name="near-b"),
    )
    triple = (
        Source(unit((1.0, 0.1, 0.1)), 0.001, normal_force=0.12, friction=0.6, name="triple-a"),
        Source(unit((-0.2, 1.0, -0.15)), 0.003, normal_force=0.085, friction=0.3, name="triple-b"),
        Source(unit((-0.4, -0.3, 1.0)), 0.002, normal_force=0.095, friction=0.6, name="triple-c"),
    )
    cases.extend((Case("two-far-asymmetric", far, "multiple"), Case("two-near", near, "multiple"),
                  Case("three-far-asymmetric", triple, "multiple")))
    for i, t in enumerate(np.linspace(-0.35, 0.35, 9)):
        source = Source(
            direction=unit((1.0, t, 0.23 + 0.12 * np.sin(2.0 * t))),
            radius=0.002 * (1.0 + 0.08 * np.sin(3.0 * t)),
            normal_force=0.1 * (1.0 + 0.1 * np.cos(t)), friction=0.3,
            name=f"path-{i:02d}",
        )
        cases.append(Case(source.name, (source,), "smooth_path"))
    cases.append(Case("release", (), "release"))
    assert len(cases) == 43 and sum(bool(case.sources) for case in cases) == 42
    return tuple(cases)


def _manifest_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _manifest_value(item) for key, item in value.items()
                if key != "cap_tangent" or item is not None}
    if isinstance(value, (tuple, list)):
        return [_manifest_value(item) for item in value]
    return value


def cases_manifest(cases: tuple[Case, ...]) -> list[dict]:
    return [_manifest_value(asdict(case)) for case in cases]


def performance_cases() -> tuple[Case, ...]:
    """Select fifteen loaded controls and release for a fixed sixteen-case tile."""
    cases = make_cases()
    indices = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 24, 28, 30, 31, 32, 42)
    return tuple(cases[i] for i in indices)
