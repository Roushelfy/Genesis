"""Constraint generation for nonnegative quadratic surface gaps, without relaxing admission."""

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import numpy as np

from .p2_elastic import quadratic_triangle_minimum

if TYPE_CHECKING:
    from .plate_contact_fem import ContactMinimum


@dataclass
class QuadraticContactMinimum:
    minimum: "ContactMinimum"
    interpolation: np.ndarray
    rounds: int
    converged: bool
    minimum_scaled_gap: float


def quadratic_contact_minimum(matrix, gap, rigid, target, initial, triangles, solve, journal=None):
    """Add the exact most-penetrating point of each quadratic face to the same convex contact problem.

    Each new constraint is evaluated with the P2 shape functions. Its nonnegative
    multiplier is a compressive point force, spread by those same functions;
    consistent P2 nodal loads may have negative entries. The base compliance
    remains unchanged and its exact interpolation supplies every new row.
    """
    interpolation = np.eye(len(gap))
    force = initial.copy()
    converged = False
    for iteration in range(40):
        minimum = solve(
            interpolation @ matrix @ interpolation.T, interpolation @ gap, interpolation @ rigid, target, force
        )
        force = minimum.force
        nodal_force = interpolation.T @ force
        nodal_gap = matrix @ nodal_force + gap + rigid @ minimum.coordinates
        values, bary = quadratic_triangle_minimum(nodal_gap[triangles], return_points=True)
        minimum_gap = float(values.min())
        if journal is not None:
            journal.write(
                json.dumps(
                    {
                        "constraint_round": iteration + 1,
                        "constraint_count": len(force),
                        "minimum_scaled_gap": minimum_gap,
                        "active_kkt_rank": int(minimum.rank),
                        "active_kkt_dimension": int(minimum.active.sum() + 4),
                    }
                )
                + "\n"
            )
        if not minimum.converged:
            break
        if minimum_gap >= -1e-8:
            if minimum.rank == minimum.active.sum() + 4:
                converged = True
                break
            if iteration == 39:
                break
            # Near-coincident generated constraints can leave a redundant active
            # multiplier basis. Remove one, solve again, and recheck the exact
            # whole-face gap oracle; no physical acceptance threshold changes.
            active = np.flatnonzero(minimum.active)
            generated = active[active >= len(gap)]
            if not len(generated):
                break
            distance = np.linalg.norm(interpolation[generated, None] - interpolation[None, active], axis=2)
            distance[generated[:, None] == active[None]] = np.inf
            remove = generated[np.unravel_index(distance.argmin(), distance.shape)[0]]
            if journal is not None:
                journal.write(
                    json.dumps(
                        {
                            "removed_redundant_generated_constraint": int(remove),
                            "nearest_active_row_distance": float(distance.min()),
                        }
                    )
                    + "\n"
                )
            interpolation = np.delete(interpolation, remove, axis=0)
            force = np.delete(force, remove)
            continue
        if iteration == 39:
            break
        selected = np.flatnonzero(values < -1e-8)
        coordinates = bary[selected]
        weights = np.column_stack(
            (
                coordinates * (2 * coordinates - 1),
                4 * coordinates[:, 0] * coordinates[:, 1],
                4 * coordinates[:, 1] * coordinates[:, 2],
                4 * coordinates[:, 2] * coordinates[:, 0],
            )
        )
        added = np.zeros((len(selected), len(gap)))
        added[np.arange(len(selected))[:, None], triangles[selected]] = weights
        # Identical constraints cannot repair a rank-deficient or insufficiently
        # accurate solve. Retain that failure instead of silently accepting it.
        distance = np.max(np.abs(added[:, None] - interpolation[None]), axis=2).min(axis=1)
        if distance.min() < 1e-10:
            break
        interpolation = np.concatenate((interpolation, added))
        force = np.r_[force, np.zeros(len(added))]
    return QuadraticContactMinimum(minimum, interpolation, iteration + 1, converged, minimum_gap)
