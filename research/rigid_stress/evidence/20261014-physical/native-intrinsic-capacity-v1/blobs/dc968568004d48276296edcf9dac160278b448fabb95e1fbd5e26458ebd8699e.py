"""Separate sampled feasibility, conditioning and nonlinear fit failure on a captured contact."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from research.rigid_stress.mechanics import P2Shell, SurfaceGeometry
from research.rigid_stress.wrench import FinitePatchMapper, PadPressureFit, WrenchPatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.fixture.read_text())
    with np.load("examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
        oracle = P2Shell(
            asset["vertices"],
            asset["tetrahedra"],
            asset["surface_triangles"],
            1e10,
            0.3,
            2000,
            2,
            factor_backend="none",
        )
    mapper = FinitePatchMapper(SurfaceGeometry(oracle, 10), anchor_to_surface=True, adaptive_integration=True)
    reports = []
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
        direction = patch.force_n / np.linalg.norm(patch.force_n)
        first = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        report = {"case": case, "anchor": anchor.tolist(), "integrations": []}
        ids = np.array(mapper.tree.query_ball_point(anchor, patch.radius_m, return_sorted=True))
        distance = np.sum((mapper.positions[ids] - anchor) ** 2, axis=1) / patch.radius_m**2
        weights = mapper.weights[ids] * np.exp(-0.5 * distance / 0.45**2) * np.maximum(0, 1 - distance) ** 2
        integrations = [("fixed", mapper.positions[ids], weights)]
        positions, weights, *_ = mapper.refined_quadrature(replace(patch, center_m=anchor))
        integrations.append(("local", positions, weights))
        for name, positions, weights in integrations:
            coordinates = np.column_stack(
                (
                    np.ones(len(positions)),
                    (positions - anchor) @ first / patch.radius_m,
                    (positions - anchor) @ second / patch.radius_m,
                )
            )
            target = np.array([1.0, 0.0, 0.0])
            feasibility = linprog(
                np.zeros(len(positions)), A_eq=coordinates.T, b_eq=target, bounds=(0, None), method="highs"
            )
            gram = coordinates.T @ (weights[:, None] * coordinates) / weights.sum()
            row = {
                "name": name,
                "samples": len(positions),
                "lp_status": feasibility.status,
                "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
            }
            if feasibility.success:
                row["lp_residual"] = float(np.linalg.norm(coordinates.T @ feasibility.x - target))
            try:
                forces, diagnostic = PadPressureFit(positions, weights, patch).solve()
                row["cpu_fit"] = vars(diagnostic)
                row["force_norm_N"] = float(np.linalg.norm(forces.sum(axis=0)))
            except ValueError as error:
                row["cpu_fit_error"] = str(error)
            coefficient = np.linalg.solve(gram, target)
            trace = []
            normalized = weights / weights.sum()
            for iteration in range(80):
                profile = np.maximum(0.0, coordinates @ coefficient)
                gradient = coordinates.T @ (normalized * profile) - target
                hessian = coordinates.T @ ((normalized * (profile > 0))[:, None] * coordinates)
                scale = 1 / np.sqrt(np.maximum(np.diag(hessian), 1e-300))
                balanced = scale[:, None] * hessian * scale[None, :]
                trace.append(
                    {
                        "iteration": iteration,
                        "residual": float(np.linalg.norm(gradient)),
                        "determinant_ratio": float(np.linalg.det(hessian) / np.trace(hessian) ** 3),
                        "balanced_determinant_ratio": float(np.linalg.det(balanced) / np.trace(balanced) ** 3),
                    }
                )
                if np.linalg.norm(gradient) <= 2e-12:
                    break
                step = scale * np.linalg.lstsq(balanced, scale * gradient, rcond=1e-14)[0]
                objective = coefficient[0] - 0.5 * (normalized @ profile**2)
                fraction = 1.0
                for _ in range(40):
                    candidate = coefficient - fraction * step
                    value = candidate[0] - 0.5 * (normalized @ np.maximum(0.0, coordinates @ candidate) ** 2)
                    if value >= objective + 1e-4 * fraction * (gradient @ step) - 1e-15:
                        coefficient = candidate
                        break
                    fraction *= 0.5
            row["balanced_fit_trace"] = trace
            report["integrations"].append(row)
        reports.append(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
