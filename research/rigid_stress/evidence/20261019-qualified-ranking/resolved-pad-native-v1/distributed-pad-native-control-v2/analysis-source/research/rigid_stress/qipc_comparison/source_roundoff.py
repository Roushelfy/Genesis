"""Bound cone-input reconciliation by the independent physical force scale.

Only sources rejected by the existing local sixteen-epsilon cone test are
changed. A finite aggregate correction beyond sixteen source epsilons of the
actual command/unsigned-force scale is refused, rather than a relaxed cone.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SourceRoundoff:
    forces: np.ndarray
    corrected_sources: int
    force_change_bound_n: float
    moment_change_bound_nm: float
    physical_force_budget_n: float


def reconcile_source_roundoff(points, forces, normals, friction, physical_force_scale):
    if not np.isfinite(physical_force_scale) or physical_force_scale < 0 or not np.isfinite(friction) or friction < 0:
        raise ValueError("Finite nonnegative physical force scale and friction are required")
    if not np.isfinite(forces).all() or not np.isfinite(normals).all() or not np.isfinite(points).all():
        raise ValueError("Finite source geometry, normal and force are required")
    assert points.shape == forces.shape == normals.shape and forces.shape[1] == 3
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-12)
    epsilon = np.finfo(forces.dtype).eps
    compression = np.einsum("ni,ni->n", forces, normals)
    tangent = forces - compression[:, None] * normals
    tangent_norm = np.linalg.norm(tangent, axis=1)
    magnitudes = np.linalg.norm(forces, axis=1)
    allowance = 16 * epsilon * magnitudes
    rejected = (compression < -allowance) | (tangent_norm > friction * compression + allowance)
    bound = friction * np.maximum(compression, 0)
    ratio = np.minimum(1, np.divide(bound, tangent_norm, out=np.ones_like(bound), where=tangent_norm > 0))
    corrected = forces.copy()
    corrected[rejected] = (np.maximum(compression[rejected], 0)[:, None] * normals[rejected]
                           + ratio[rejected, None] * tangent[rejected])
    delta = corrected - forces
    force_bound = np.linalg.norm(delta, axis=1).sum()
    budget = 16 * epsilon * max(physical_force_scale, magnitudes.sum())
    if force_bound > budget:
        raise ValueError(f"Cone conflict exceeds physical roundoff budget: {force_bound:.9g}N > {budget:.9g}N")
    p = np.einsum("ni,ni->n", corrected, normals)
    t = np.linalg.norm(corrected - p[:, None] * normals, axis=1)
    actual_allowance = 16 * epsilon * np.linalg.norm(corrected, axis=1)
    assert np.all(p >= -actual_allowance) and np.all(t <= friction * p + actual_allowance)
    moment_bound = np.sum(np.linalg.norm(points, axis=1) * np.linalg.norm(delta, axis=1))
    return SourceRoundoff(corrected, int(rejected.sum()), force_bound.item(), moment_bound.item(), budget.item())
