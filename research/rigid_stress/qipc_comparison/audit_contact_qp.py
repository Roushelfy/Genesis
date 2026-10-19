"""Independently solve a retained contact QP without admitting a physical reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .contact_qp import solve_contact_qp


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--field", type=Path, required=True)
    cli.add_argument("--solver-path", type=Path, required=True)
    cli.add_argument("--load", type=float, default=0.1)
    cli.add_argument("--constraint-rounds", type=int, default=1)
    cli.add_argument("--form", choices=("dual", "whitened-primal"), default="dual")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert args.constraint_rounds > 0
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
    target = np.array([0.0, 0.0, 0.0, 1.0])
    a, b, c, _ = np.split(boundary, 4)
    quadratic = np.column_stack((a[:, 0], b[:, 1], c[:, 2], a[:, 1], b[:, 2], a[:, 2]))
    mapping = np.full(outer_count, -1, dtype=int)
    mapping[candidate] = np.arange(len(candidate))
    triangles = mapping[quadratic]
    triangles = triangles[(triangles >= 0).all(axis=1)]
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        solved = solve_contact_qp(
            compliance,
            gap,
            rigid,
            target,
            triangles,
            args.solver_path,
            interpolation=interpolation,
            round_limit=args.constraint_rounds,
            form=args.form,
            journal=journal,
        )
    np.savez_compressed(
        args.output / "solution.npz",
        interpolation=solved.interpolation,
        multipliers=solved.multipliers,
        nodal_alpha=solved.nodal_force,
        coordinates_scaled=solved.coordinates,
        constraint_gap_scaled=solved.constraint_gap,
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "field_input": field_id,
            "operator": file_id(operator),
            "solver_version": solved.solver_version,
            "solver_module": solved.solver_module,
            "status": solved.status,
            "form": args.form,
            "constraint_rounds": solved.rounds,
            "candidate_face_certificate_passed": solved.candidate_passed,
            "iterations": solved.iterations,
            "solve_seconds": solved.solve_seconds,
            "force_moment_relative_error": solved.force_error,
            "stationarity_scaled_l2": solved.stationarity,
            "complementarity_scaled_max": solved.complementarity,
            "minimum_multiplier": float(solved.multipliers.min()),
            "minimum_constraint_gap_scaled": float(solved.constraint_gap.min()),
            "minimum_candidate_face_gap_scaled": solved.minimum_gap,
            "nodal_force_relative_change": float(
                np.linalg.norm(solved.nodal_force - original_alpha) / np.linalg.norm(solved.nodal_force)
            ),
            "fixed_qp_certificate_passed": solved.fixed_passed,
            "physical_reference_qualified": False,
            "scope": "Independent primal-dual certificate for the retained finite set of contact constraints. Nonunique multipliers are permitted by the convex QP; no full-field or continuum admission is made. Exact candidate-face gap is reported separately and may require additional constraints. No elasticity solve or outside-candidate boundary check is performed.",
            "solver_documentation": "https://clarabel.org/stable/python/getting_started_py/",
        },
    )


if __name__ == "__main__":
    main()
