"""Nonnegative source reduction preserving each connected contact wrench."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components


def compress(points, forces, adjacency, length):
    active = np.flatnonzero(np.linalg.norm(forces, axis=1) > 1e-12)
    if not len(active):
        return active, np.empty(0), np.empty(0, dtype=np.int64), 0.0
    count, labels = connected_components(adjacency[active][:, active], directed=False)
    selected, coefficients, groups = [], [], []
    error = 0.0
    for group in range(count):
        nodes = active[labels == group]
        magnitude = np.linalg.norm(forces[nodes], axis=1)
        directions = forces[nodes] / magnitude[:, None]
        center = points[nodes].mean(axis=0)
        matrix = np.vstack((directions.T, np.cross(points[nodes] - center, directions).T / length))
        target = matrix @ magnitude
        result = linprog(
            np.ones(len(nodes)),
            A_eq=matrix,
            b_eq=target,
            bounds=(0, None),
            method="highs",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        if not result.success:
            raise RuntimeError(f"Nonnegative contact-wrench compression failed: {result.message}")
        chosen = result.x > 1e-12
        defect = matrix[:, chosen] @ result.x[chosen] - target
        error = max(error, np.linalg.norm(defect).item())
        assert np.linalg.norm(defect) <= 1e-8 * max(np.linalg.norm(target), 1e-8) + 1e-10
        selected.append(nodes[chosen])
        coefficients.append(result.x[chosen] / magnitude[chosen])
        groups.append(np.full(chosen.sum(), group))
    return np.concatenate(selected), np.concatenate(coefficients), np.concatenate(groups), error
