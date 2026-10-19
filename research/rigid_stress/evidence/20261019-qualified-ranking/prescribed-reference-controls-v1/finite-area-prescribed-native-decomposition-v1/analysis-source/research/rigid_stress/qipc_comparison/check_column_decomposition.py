"""Manufactured affine tensor controls for signed reconstruction/discretization splits."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import runtime_identity, write_json
from .column_decomposition import integrate


class AffineControl:
    def __init__(self, coefficient, scale, boundaries):
        self.coefficient, self.scale, self.boundaries = coefficient, scale, boundaries
        self.volumes = np.array([2.0, 3.0])

    def breaks(self):
        return np.tile(self.boundaries, (2, 1)).T

    def volume_density(self, depth):
        return np.full(len(depth), 5.0)

    def sample(self, depth, basis):
        reference = np.stack((2 + depth, 3 - depth), axis=-1)
        perturbation = np.stack((1 + 2 * depth, -0.5 + depth), axis=-1)
        diagonal = self.scale * np.array([1, 2])[:, None] * (reference + self.coefficient * perturbation)
        return diagonal[:, :, None] * np.eye(2)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    nodes, weights = np.polynomial.legendre.leggauss(2)
    basis = np.tile(np.eye(3)[:, :2], (2, 1, 1))
    area, angular = np.array([2.0, 3.0]), np.array([0.4, 0.6])
    # Analytic integrals of diag(1+2z, -1/2+z), independently expanded by hand.
    h = 0.0005
    expected_first_mean_square = {
        "tangent": 14 / 5 * 53 / 12,
        "membrane": 14 / 5 * 4 * h**2,
        "bending": 14 / 5 * 5 * h**4 / 144,
        "outer_tangent": 14 / 5 * 5 / 4,
        "inner_tangent": 14 / 5 * 37 / 4,
    }
    cases = []
    for coefficient in (-0.5, 0.5):
        for scale in (1e-12, 1.0, 1e12):
            fields = [
                AffineControl(1 + coefficient, scale, [0, 0.2, 1]),
                AffineControl(coefficient, scale, [0, 0.8, 1]),
                AffineControl(0, scale, [0, 0.3, 0.4, 1]),
            ]
            result = integrate(fields, basis, area, angular, nodes, weights)
            for metric, expected in expected_first_mean_square.items():
                row = result["metrics"][metric]
                np.testing.assert_allclose(row["first_difference_rms"], scale * np.sqrt(expected), rtol=1e-12, atol=0)
                np.testing.assert_allclose(
                    row["second_difference_rms"], abs(coefficient) * scale * np.sqrt(expected), rtol=1e-12, atol=0
                )
                np.testing.assert_allclose(
                    row["total_difference_rms"], abs(1 + coefficient) * scale * np.sqrt(expected), rtol=1e-12, atol=0
                )
                np.testing.assert_allclose(
                    row["twice_cross_mean_square"], 2 * coefficient * scale**2 * expected, rtol=1e-12, atol=0
                )
                np.testing.assert_allclose(row["cross_correlation"], np.sign(coefficient), rtol=1e-12, atol=0)
            # Fine tangent reference norm integrates (2+z)^2+(3-z)^2 to38/3.
            np.testing.assert_allclose(
                result["metrics"]["tangent"]["reference_rms"], scale * np.sqrt(14 / 5 * 38 / 3), rtol=1e-12, atol=0
            )
            cases.append({"coefficient": coefficient, "scale": scale, "passed": True, "metrics": result["metrics"]})
    fields = [AffineControl(0, 0, [0, 1])] * 3
    zero = integrate(fields, basis, area, angular, nodes, weights)
    assert all(
        row["total_difference_rms"] == 0 and row["total_relative_l2"] is None and row["zero_reference_norm"]
        for row in zero["metrics"].values()
    )
    cases.append({"case": "exact_zero_reference_and_error", "passed": True})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": True,
            "cases": cases,
            "scope": "Seven manufactured controls: nonmatching material partitions; analytic volume, membrane, bending and boundary integrals; signed cancellation/reinforcement;24-decade stress scaling; explicit zero-normalization handling. No physical reference qualification.",
        },
    )


if __name__ == "__main__":
    main()
