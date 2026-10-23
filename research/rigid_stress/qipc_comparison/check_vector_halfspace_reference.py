"""Check vector cell compliance and unilateral contact against circular Hertz pressure."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import runtime_identity, write_json
from .study_box_corner_elasticity import galerkin_tensor, kernel_control, normal_contact


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--grid", type=int, default=12)
    cli.add_argument("--quadrature", type=int, default=4)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "grid": args.grid,
            "quadrature": args.quadrature,
            "scope": "Dimensionless Hertz control: reduced modulus and radius1, force4/3, exact radius and indentation1. The Galerkin normal-contact solve selects support and pressure from the quadratic gap. Target quadrature and spatial convergence are retained separately.",
        },
    )
    error = kernel_control()
    width = 3 / args.grid
    coordinates = (np.arange(args.grid) + 0.5) * width - 1.5
    points = np.array(np.meshgrid(coordinates, coordinates, indexing="ij")).reshape(2, -1).T
    widths = np.full_like(points, width)
    tensor, reciprocity = galerkin_tensor(points, widths, young=0.91, poisson=0.3, order=args.quadrature)
    matrix = tensor[:, :, 2, 2]
    np.linalg.cholesky(matrix)
    gap = np.sum(points**2 + widths**2 / 12, axis=1) / 2
    force, indentation, dual, iterations = normal_contact(
        matrix, gap, np.zeros(len(points), dtype=int), np.array([4 / 3])
    )
    knots, weights = np.polynomial.legendre.leggauss(48)
    offsets = np.array(np.meshgrid(knots / 2, knots / 2, indexing="ij")).reshape(2, -1).T * width
    weights = np.outer(weights, weights).reshape(-1) / 4
    radius2 = np.sum((points[:, None] + offsets) ** 2, axis=2)
    exact = 2 / np.pi * (np.sqrt(np.maximum(1 - radius2, 0)) @ weights) * width**2
    boundary = np.any(np.abs(points) > 1.5 - 0.51 * width, axis=1)
    assert force[boundary].sum() < 1e-12
    np.testing.assert_allclose(points.T @ force, 0, atol=1e-10, rtol=0)
    pressure_error = np.linalg.norm(force - exact) / np.linalg.norm(exact)
    depth_error = abs(indentation[0] - 1)
    np.savez_compressed(args.output / "fields.npz", points=points, force=force, hertz_force=exact, gap=dual)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "grid": args.grid,
            "quadrature": args.quadrature,
            "kernel_relative_error": error,
            "raw_reciprocity_relative_error": reciprocity,
            "active_cells": int(np.count_nonzero(force > 0)),
            "iterations": iterations,
            "indentation": float(indentation[0]),
            "indentation_relative_error": float(depth_error),
            "pressure_cell_l2_relative_error": float(pressure_error),
            "analytic_pressure_force_integration_relative_error": float(abs(exact.sum() / (4 / 3) - 1)),
            "force_relative_error": float(abs(force.sum() / (4 / 3) - 1)),
            "pressure_10pct_depth_2pct_targets_met": bool(pressure_error < 0.10 and depth_error < 0.02),
            "scope": "Independent analytical circular Hertz pressure/indentation check for the numerical normal-contact operator. Vector displacement integration, force balance, unilateral KKT and remote unloaded boundaries are checked. This control covers smooth single-face normal contact, with sharp-corner geometry and friction remaining separate questions.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
