"""Offline numerical diagnosis of feasible near-apex pressure globalization."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import linalg

from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry
from research.rigid_stress.wrench import FinitePatchMapper, WrenchPatch


def solve(coordinates, weights, method, merit, extended):
    operator = coordinates.astype(np.longdouble if extended else np.float64)
    weights = weights.astype(operator.dtype)
    weights /= weights.sum()
    target = np.array([1, 0, 0], dtype=operator.dtype)
    gram = coordinates.T @ (np.asarray(weights, dtype=float)[:, None] * coordinates)
    coefficient = np.linalg.solve(gram, [1, 0, 0]).astype(operator.dtype)
    trace = []
    for iteration in range(80):
        profile = np.maximum(0, operator @ coefficient)
        gradient = operator.T @ (weights * profile) - target
        hessian = operator.T @ ((weights * (profile > 0))[:, None] * operator)
        norm = float(np.linalg.norm(gradient))
        trace.append({"iteration": iteration, "residual": norm})
        if norm <= 2e-12:
            break
        scale = 1 / np.sqrt(np.maximum(np.diag(hessian), 1e-300))
        balanced = np.asarray(scale[:, None] * hessian * scale[None, :], dtype=float)
        rhs = np.asarray(scale * gradient, dtype=float)
        if method == "inverse":
            local = np.linalg.inv(balanced) @ rhs
        elif method == "solve":
            local = np.linalg.solve(balanced, rhs)
        else:
            q, r = linalg.qr(balanced)
            local = linalg.solve_triangular(r, q.T @ rhs)
        step = scale * local
        objective = coefficient[0] - 0.5 * (weights @ profile**2)
        fraction = 1.0
        accepted = False
        for line in range(40):
            candidate = coefficient - fraction * step
            candidate_profile = np.maximum(0, operator @ candidate)
            value = candidate[0] - 0.5 * (weights @ candidate_profile**2)
            candidate_norm = float(np.linalg.norm(operator.T @ (weights * candidate_profile) - target))
            if value >= objective + 1e-4 * fraction * (gradient @ step) - 1e-15 or (
                merit and norm < 1e-4 and candidate_norm < 0.9 * norm
            ):
                coefficient = candidate
                accepted = True
                break
            fraction *= 0.5
        trace[-1].update({"line_evaluations": line + 1, "fraction": fraction, "line_accepted": accepted})
    profile = np.maximum(0, operator @ coefficient)
    error = operator.T @ (weights * profile) - target
    return {
        "method": method,
        "merit": merit,
        "extended_evaluation": extended,
        "accepted": float(np.linalg.norm(error)) <= 2e-12,
        "residual": float(np.linalg.norm(error)),
        "coefficient": np.asarray(coefficient, dtype=float).tolist(),
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.fixture.read_text())
    with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
        shell = P2Shell(
            asset["vertices"],
            asset["tetrahedra"],
            asset["surface_triangles"],
            1e10,
            0.3,
            2000,
            2,
            factor_backend="none",
        )
    mapper = FinitePatchMapper(SurfaceGeometry(shell, 10), anchor_to_surface=True, adaptive_integration=True)
    rows = []
    for case in source["cases"]:
        patch = WrenchPatch(
            np.array(case["position"]),
            np.array(case["force"]),
            case["radius"],
            case["friction"],
            np.zeros(3),
            np.array(case["normal"]),
        )
        anchor = mapper.surface_anchor(patch.center_m, patch.force_n)
        positions, weights, *_ = mapper.refined_quadrature(replace(patch, center_m=anchor))
        direction = patch.force_n / np.linalg.norm(patch.force_n)
        first = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
        first /= np.linalg.norm(first)
        coordinates = np.column_stack(
            (
                np.ones(len(positions)),
                (positions - anchor) @ first / patch.radius_m,
                (positions - anchor) @ np.cross(direction, first) / patch.radius_m,
            )
        )
        variants = []
        for method in ("inverse", "solve", "qr"):
            for merit in (False, True):
                for extended in (False, True):
                    result = solve(coordinates, weights, method, merit, extended)
                    variants.append(result)
                    print(
                        method,
                        merit,
                        extended,
                        result["accepted"],
                        result["residual"],
                        len(result["trace"]),
                        flush=True,
                    )
        rows.append({"case": case, "samples": len(positions), "variants": variants})
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
