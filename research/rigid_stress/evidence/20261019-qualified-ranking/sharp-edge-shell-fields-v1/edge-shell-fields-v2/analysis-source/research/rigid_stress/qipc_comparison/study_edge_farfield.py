"""Validate a four-source remote sharp-edge field against continuous pressure integration."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_fields import composite_tensor
from .hertz_reference import halfspace_stress
from .hertz_tangential import cerruti_stress
from .study_edge_fields import field_metrics


def pressure_rule(width, along, exponent, nx, ny):
    """Integrate the normalized sharp-edge pressure with separable Jacobi weights."""
    u, wu = roots_jacobi(nx, exponent, -exponent)
    v, wv = roots_jacobi(ny, 1, 1)
    u, wu, wv = (u + 1) / 2, wu / wu.sum(), wv / wv.sum()
    x = width * (1 - v[:, None] ** 2) * u
    source = np.column_stack([x.reshape(-1), np.repeat(along * v, nx)])
    return source, (wv[:, None] * wu).reshape(-1)


def moment_rule(width, along, exponent):
    """Match the first and second pressure moments with four equal quadrature weights."""
    center = 2 * (1 - exponent) * width / 5
    spread = width * np.sqrt(4 * (1 - exponent) * (3 + 2 * exponent) / 175)
    source = np.array([[center + sx * spread, sy * along / np.sqrt(5)] for sx in [-1, 1] for sy in [-1, 1]])
    return source, np.full(4, 0.25)


def evaluate_rule(points, source, weights, force, tangent, poisson):
    loads = force * weights
    return halfspace_stress(points, source, loads, poisson) + cerruti_stress(
        points, source, loads[:, None] * tangent, poisson
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.geometry)],
            "targets": {"reference_change": 1e-8, "far_maximum_point_tensor_relative": 0.002},
            "scope": "Independent continuous prescribed-pressure integration and four-source moment quadrature. Two captured physical parameter sets and nine synthetic slender profiles. This checks a remote field approximation, with pressure/contact and finite-shell physical accuracy assessed separately.",
        },
    )
    cases = [c for c in json.loads(args.geometry.read_text())["cases"] if c["kind"] == "pressure"]
    for ratio in [0.001, 0.01, 0.049]:
        for exponent in [0.2, 0.5, 0.8]:
            width, along = ratio, 1.0
            amplitude = 3 * np.sin(np.pi * exponent) / (4 * np.pi * exponent * width * along)
            cases.append(
                {
                    "width": width,
                    "along": along,
                    "exponent": exponent,
                    "amplitude": amplitude,
                    "force": [0.7, -0.4, 1.0],
                }
            )
    azimuth = np.arange(16) * np.pi / 8
    directions = np.array(
        [
            [np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z]
            for z in [1e-4, 0.05, 0.3, 0.7, 1.0]
            for a in azimuth
        ]
    )
    positions, inputs, expected, independent, rows = [], [], [], [], []
    for index, case in enumerate(cases):
        width, along, exponent = case["width"], case["along"], case["exponent"]
        force = np.array(case["force"])
        tangent = force[:2] / force[2]
        points = np.concatenate([radius * along * directions for radius in [7.99, 8.001, 8.01, 16, 64]])
        far = np.linalg.norm(points, axis=1) >= 8 * along
        source, weights = pressure_rule(width, along, exponent, 12, 20)
        low = evaluate_rule(points, source, weights, force[2], tangent, 0.3)
        source, weights = pressure_rule(width, along, exponent, 24, 40)
        high = evaluate_rule(points, source, weights, force[2], tangent, 0.3)
        expected_mean = np.array([2 * (1 - exponent) * width / 5, 0])
        expected_second = np.diag([4 * (1 - exponent) * (2 - exponent) * width**2 / 35, along**2 / 5])
        np.testing.assert_allclose(weights @ source, expected_mean, rtol=1e-11, atol=1e-13 * along)
        np.testing.assert_allclose(
            source.T @ (weights[:, None] * source), expected_second, rtol=1e-11, atol=1e-13 * along**2
        )
        source, weights = moment_rule(width, along, exponent)
        np.testing.assert_allclose(weights @ source, expected_mean, rtol=1e-12, atol=1e-15 * along)
        np.testing.assert_allclose(
            source.T @ (weights[:, None] * source), expected_second, rtol=1e-12, atol=1e-15 * along**2
        )
        moment = evaluate_rule(points, source, weights, force[2], tangent, 0.3)
        model = EdgeScales(width, along, exponent, case["amplitude"])
        composite = composite_tensor(points[:160], model, force[2], tangent)
        candidate = moment.copy()
        candidate[~far] = composite[:80]
        relative = np.linalg.norm(moment - high, axis=(1, 2)) / np.linalg.norm(high, axis=(1, 2))
        convergence = np.linalg.norm(low - high, axis=(1, 2)) / np.linalg.norm(high, axis=(1, 2))
        transition = np.linalg.norm(composite[80:] - moment[80:160], axis=(1, 2)) / np.linalg.norm(
            high[80:160], axis=(1, 2)
        )
        row = {
            "case": index,
            "width_over_along": width / along,
            "exponent": exponent,
            "maximum_reference_change": float(convergence.max()),
            "maximum_far_point_error": float(relative[far].max()),
            "same_point_branch_difference_at_8_001": float(transition.max()),
            "far_field_metrics": field_metrics(moment[far], high[far]),
        }
        rows.append(row)
        positions.extend(points)
        inputs.extend([[width, along, exponent, case["amplitude"], force[2], *tangent, 0.3]] * len(points))
        expected.extend(candidate)
        independent.extend(high)
        write_json(args.output / "progress.json", {"cases": rows})
    np.savez_compressed(
        args.output / "controls.npz", points_m=positions, inputs=inputs, expected_pa=expected, reference_pa=independent
    )
    passed = all(r["maximum_reference_change"] < 1e-8 and r["maximum_far_point_error"] < 0.002 for r in rows)
    write_json(args.output / "result.json", {"passed": passed, "cases": rows, "whole_goal_complete": False})
    assert passed


if __name__ == "__main__":
    main()
