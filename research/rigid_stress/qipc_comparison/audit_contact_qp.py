"""Independently solve a retained contact QP without admitting a physical reference."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from .artifacts import file_id, runtime_identity, write_json
from .p2_elastic import quadratic_triangle_minimum


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--field", type=Path, required=True)
    cli.add_argument("--solver-path", type=Path, required=True)
    cli.add_argument("--load", type=float, default=0.1)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    sys.path.insert(0, str(args.solver_path))
    import clarabel

    assert clarabel.__version__ == "0.11.1"
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.field.parent / "result.json").read_text())
    field_id = file_id(args.field)
    assert any(row["field"]["sha256"] == field_id["sha256"] and row["load_n"] == args.load for row in receipt["cases"])
    operator = args.field.parent / "contact-operator.npz"
    scale = args.load / (1e10 * 5e-5)
    with np.load(args.field) as saved:
        interpolation = saved["contact_constraint_interpolation"]
        original_alpha = saved["contact_alpha"]
        candidate = saved["candidate_nodes"]
        boundary = saved["outer_subtriangles"]
        outer_count = len(saved["outer_nodes"])
    with np.load(operator) as saved:
        np.testing.assert_array_equal(saved["nodes"], candidate)
        compliance = saved["compliance_m_per_n"] * args.load / scale
        gap = saved["initial_gap_m"] / scale
        rigid = saved["rigid"]
    matrix = interpolation @ compliance @ interpolation.T
    linear = interpolation @ gap
    equality = interpolation @ rigid
    target = np.array([0.0, 0.0, 0.0, 1.0])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = 1e-12
    settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-12
    settings.iterative_refinement_abstol = 1e-14
    settings.iterative_refinement_reltol = 1e-14
    solver = clarabel.DefaultSolver(
        sparse.triu(sparse.csc_matrix(matrix), format="csc"),
        linear,
        sparse.vstack((sparse.csc_matrix(equality.T), -sparse.eye(len(linear))), format="csc"),
        np.r_[target, np.zeros(len(linear))],
        [clarabel.ZeroConeT(4), clarabel.NonnegativeConeT(len(linear))],
        settings,
    )
    solution = solver.solve()
    multipliers = np.array(solution.x)
    coordinates = np.array(solution.z[:4])
    alpha = interpolation.T @ multipliers
    nodal_gap = compliance @ alpha + gap + rigid @ coordinates
    constraint_gap = interpolation @ nodal_gap
    wrench = float(np.linalg.norm(equality.T @ multipliers - target))
    stationarity = float(np.linalg.norm(constraint_gap - np.asarray(solution.z[4:])))
    complementarity = float(np.max(np.abs(multipliers * constraint_gap)))
    a, b, c, _ = np.split(boundary, 4)
    quadratic = np.column_stack((a[:, 0], b[:, 1], c[:, 2], a[:, 1], b[:, 2], a[:, 2]))
    mapping = np.full(outer_count, -1, dtype=int)
    mapping[candidate] = np.arange(len(candidate))
    triangles = mapping[quadratic]
    triangles = triangles[(triangles >= 0).all(axis=1)]
    minimum_gap = float(quadratic_triangle_minimum(nodal_gap[triangles]).min())
    np.savez_compressed(
        args.output / "solution.npz",
        interpolation=interpolation,
        multipliers=multipliers,
        nodal_alpha=alpha,
        coordinates_scaled=coordinates,
        constraint_gap_scaled=constraint_gap,
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "field_input": field_id,
            "operator": file_id(operator),
            "solver_version": clarabel.__version__,
            "solver_module": str(Path(clarabel.__file__).resolve()),
            "status": str(solution.status),
            "iterations": solution.iterations,
            "solve_seconds": solution.solve_time,
            "force_moment_relative_error": wrench,
            "stationarity_scaled_l2": stationarity,
            "complementarity_scaled_max": complementarity,
            "minimum_multiplier": float(multipliers.min()),
            "minimum_constraint_gap_scaled": float(constraint_gap.min()),
            "minimum_candidate_face_gap_scaled": minimum_gap,
            "nodal_force_relative_change": float(np.linalg.norm(alpha - original_alpha) / np.linalg.norm(alpha)),
            "fixed_qp_certificate_passed": bool(
                solution.status == clarabel.SolverStatus.Solved
                and wrench <= 1e-8
                and stationarity <= 1e-8
                and complementarity <= 1e-8
                and multipliers.min() >= -1e-10
                and constraint_gap.min() >= -1e-10
            ),
            "physical_reference_qualified": False,
            "scope": "Independent primal-dual certificate for the retained finite set of contact constraints. Nonunique multipliers are permitted by the convex QP; no full-field or continuum admission is made. Exact candidate-face gap is reported separately and may require additional constraints. No elasticity solve or outside-candidate boundary check is performed.",
            "solver_documentation": "https://clarabel.org/stable/python/getting_started_py/",
        },
    )


if __name__ == "__main__":
    main()
