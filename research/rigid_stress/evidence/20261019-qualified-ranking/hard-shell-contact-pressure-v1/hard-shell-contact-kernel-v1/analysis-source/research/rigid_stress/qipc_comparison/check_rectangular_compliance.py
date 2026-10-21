"""Check rectangular Green integrals against old squares and Duffy quadrature."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json
from .hertz_reference import rectangular_compliance, square_compliance


def singular_rectangle_integral(widths, target):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes, weights = (nodes + 1) / 2, weights / 2
    result = 0.0
    # Each rectangle incident on the target is split into two triangles. A
    # radial Duffy map cancels the 1/r singularity with its radial Jacobian.
    for x in [widths[0] / 2 + target[0], widths[0] / 2 - target[0]]:
        for y in [widths[1] / 2 + target[1], widths[1] / 2 - target[1]]:
            if x == 0 or y == 0:
                continue
            result += x * y * np.dot(weights, 1 / np.sqrt(x * x + (y * nodes) ** 2))
            result += x * y * np.dot(weights, 1 / np.sqrt((x * nodes) ** 2 + y * y))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("old_square_hertz_reference", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    modulus, width = 1e10, 40e-6
    grid = np.stack(np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij"), axis=-1).reshape(-1, 2)
    rows = []
    for angle in [0.0, 0.31]:
        c, s = np.cos(angle), np.sin(angle)
        points = (grid @ np.array([[c, -s], [s, c]])) * width
        old, new = baseline.square_compliance(points, width, modulus), square_compliance(points, width, modulus)
        np.testing.assert_allclose(new, old, rtol=2e-12, atol=1e-20)
        rows.append(
            {
                "case": "old_square",
                "angle_radians": angle,
                "relative_change": float(np.linalg.norm(new - old) / np.linalg.norm(old)),
            }
        )
    widths = np.array([40e-6, 70e-6])
    targets = np.array([[0, 0], [widths[0] / 2, 0], widths / 2])
    actual = rectangular_compliance(targets, np.zeros((1, 2)), widths, modulus)[:, 0]
    expected = np.array([singular_rectangle_integral(widths, target) for target in targets]) / (
        np.pi * modulus * widths.prod()
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0)
    for label, first, second in zip(["center", "edge", "vertex"], actual, expected):
        rows.append({"case": label, "relative_error": float(abs(first / second - 1))})
    nodes, weights = np.polynomial.legendre.leggauss(48)
    source = np.stack(np.meshgrid(*(nodes * width / 2 for width in widths), indexing="ij"), axis=-1).reshape(-1, 2)
    target = np.array([[3 * widths[0], 4 * widths[1]]])
    direct = np.dot(np.outer(weights, weights).reshape(-1) / 4, 1 / np.linalg.norm(source - target, axis=1)) / (
        np.pi * modulus
    )
    integral = rectangular_compliance(target, np.zeros((1, 2)), widths, modulus)[0, 0]
    np.testing.assert_allclose(integral, direct, rtol=1e-10, atol=0)
    rows.append({"case": "separated_quadrature", "relative_error": float(abs(integral / direct - 1))})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "baseline": file_id(args.baseline),
            "scope": "Independent nonsingular Gauss and singular Duffy integration, plus unchanged square-cell controls. This verifies the mathematical kernel, not physical contact or stress accuracy.",
        },
    )


if __name__ == "__main__":
    main()
