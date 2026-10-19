"""Constraint generation for nonnegative quadratic surface gaps, without relaxing admission."""

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

import numpy as np

from .p2_elastic import quadratic_triangle_minimum, quadratic_triangle_weights

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
        if minimum.rank < minimum.active.sum() + 4:
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
        if not minimum.converged:
            break
        if minimum_gap >= -1e-8:
            converged = True
            break
        if iteration == 39:
            break
        selected = np.flatnonzero(values < -1e-8)
        coordinates = bary[selected]
        weights = quadratic_triangle_weights(coordinates)
        added = np.zeros((len(selected), len(gap)))
        added[np.arange(len(selected))[:, None], triangles[selected]] = weights
        # Neighboring faces can generate the same edge point in the same round.
        # Keep one identical interpolation row, then verify the whole gap again.
        unique = []
        for row in added:
            if not unique or np.max(np.abs(np.asarray(unique) - row), axis=1).min() > 1e-12:
                unique.append(row)
        added = np.array(unique)
        # Identical constraints cannot repair a rank-deficient or insufficiently
        # accurate solve. Retain that failure instead of silently accepting it.
        distance = np.max(np.abs(added[:, None] - interpolation[None]), axis=2).min(axis=1)
        if distance.min() < 1e-10:
            break
        interpolation = np.concatenate((interpolation, added))
        force = np.r_[force, np.zeros(len(added))]
    return QuadraticContactMinimum(minimum, interpolation, iteration + 1, converged, minimum_gap)


def main():
    """Replay only the small condensed contact problem before another full elasticity solve."""
    import argparse
    from pathlib import Path

    from scipy.optimize import minimize
    from skfem import Basis, ElementTetP2, MeshTet

    from .artifacts import file_id, runtime_identity, write_json
    from .plate_contact_fem import active_contact_minimum

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--field", type=Path, required=True)
    cli.add_argument("--load", type=float, default=0.1)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    operator = args.field.parent / "contact-operator.npz"
    start = json.loads((args.field.parent / "start.json").read_text())
    assert start["mesh"]["sha256"] == file_id(args.mesh)["sha256"] and start["element_order"] == 2
    with np.load(args.mesh) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
        faces = saved["mid_faces"]
    basis = Basis(mesh, ElementTetP2(), intorder=2)
    with np.load(args.field) as saved:
        np.testing.assert_array_equal(saved["vertices_m"], basis.doflocs.T)
        candidate = saved["outer_nodes"][saved["candidate_nodes"]]
    lookup = {tuple(edge): node for edge, node in zip(mesh.edges.T, basis.edge_dofs[0])}
    mids = np.array([[lookup[tuple(sorted((a, b)))] for a, b in zip(face, np.roll(face, -1))] for face in faces])
    mapping = np.full(basis.N, -1, dtype=int)
    mapping[candidate] = np.arange(len(candidate))
    triangles = mapping[np.column_stack((faces, mids))]
    triangles = triangles[(triangles >= 0).all(axis=1)]
    scale = args.load / (1e10 * 5e-5)
    with np.load(operator) as saved:
        matrix = saved["compliance_m_per_n"] * args.load / scale
        gap, rigid = saved["initial_gap_m"] / scale, saved["rigid"]
        top = saved["sign"] < 0
    target = np.array([0.0, 0.0, 0.0, 1.0])
    initial = np.where(top, 1 / top.sum(), 1 / (~top).sum())
    fitted = minimize(
        lambda x: (0.5 * x @ matrix @ x + gap @ x, matrix @ x + gap),
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0, None)] * len(initial),
        constraints={"type": "eq", "fun": lambda x: rigid.T @ x - target, "jac": lambda x: rigid.T},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        solved = quadratic_contact_minimum(
            matrix, gap, rigid, target, fitted.x, triangles, active_contact_minimum, journal
        )
    force = solved.interpolation.T @ solved.minimum.force
    np.savez_compressed(
        args.output / "contact-solution.npz",
        interpolation=solved.interpolation,
        multipliers=solved.minimum.force,
        nodal_alpha=force,
        coordinates_scaled=solved.minimum.coordinates,
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "operator": file_id(operator),
            "field_input": file_id(args.field),
            "constraint_rounds": solved.rounds,
            "converged": solved.converged,
            "minimum_scaled_gap": solved.minimum_scaled_gap,
            "active_kkt_rank": int(solved.minimum.rank),
            "active_kkt_dimension": int(solved.minimum.active.sum() + 4),
            "force_moment_relative_error": float(np.linalg.norm(rigid.T @ force - target)),
            "minimum_compressive_multiplier": float(solved.minimum.force.min()),
            "physical_reference_qualified": False,
            "scope": "Condensed contact-only replay using a verified retained operator; it does not recompute elasticity or check gaps outside the candidate faces. Full-field physical admission remains a separate required run.",
        },
    )


if __name__ == "__main__":
    main()
