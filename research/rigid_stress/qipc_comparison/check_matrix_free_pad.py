"""Manufactured full-body contact controls, including redundant quadrature rows."""

import argparse
from pathlib import Path

import numpy as np
from scipy import linalg, sparse

from .artifacts import runtime_identity, write_json
from .condensed_pad_dynamics import CondensedPadDynamics, potential
from .matrix_free_pad_dynamics import MatrixFreePadDynamics
from .modal_pad_dynamics import ModalPadDynamics


class DenseFactor:
    def __init__(self, operator):
        self.factor = linalg.cho_factor(operator)

    def solve(self, rhs):
        return linalg.cho_solve(self.factor, rhs)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    generator = np.random.default_rng(937)
    rows = []
    for body_dofs, points in ((18, 4), (8, 6)):
        matrix = generator.normal(size=(body_dofs, body_dofs))
        operator = 0.01 * (np.eye(body_dofs) + matrix @ matrix.T)
        jacobian = generator.normal(size=(3 * points, body_dofs)) / np.sqrt(body_dofs)
        sign, area = np.where(np.arange(points) % 2, -1.0, 1.0), generator.uniform(0.5e-6, 2e-6, points)
        previous, gap = np.full(points, 0.005), np.full(points, -5e-6)
        factor = DenseFactor(operator)
        solver = MatrixFreePadDynamics(sparse.csc_matrix(operator), factor, sparse.csr_matrix(jacobian), sign, area)
        warm = MatrixFreePadDynamics(
            sparse.csc_matrix(operator), factor, sparse.csr_matrix(jacobian), sign, area, warm_start=True
        )
        modal = ModalPadDynamics(
            sparse.csc_matrix(operator),
            factor,
            sparse.csr_matrix(jacobian),
            np.eye(body_dofs)[:, :7],
            sign,
            area,
            warm_start=True,
        )
        dense = (
            CondensedPadDynamics(jacobian @ factor.solve(jacobian.T), sign, area, warm_start=True)
            if body_dofs >= 3 * points
            else None
        )
        for dt in (0.001, 0.002):
            for speed in (0.0, 1e-5, 1e-3):
                for friction in (0.0, 0.3):
                    expected_body = generator.normal(size=body_dofs) * speed
                    expected = (jacobian @ expected_body).reshape(-1, 3)
                    _, impulse, _ = potential(expected, gap, sign, area, previous, dt, 1e9, friction)
                    free = expected_body - factor.solve(jacobian.T @ impulse.reshape(-1))
                    result = solver.solve(free, gap, previous, dt, 1e9, friction, 0.02, 0.01)
                    warm_result = warm.solve(free, gap, previous, dt, 1e9, friction, 0.02, 0.01)
                    modal_result = modal.solve(free, gap, previous, dt, 1e9, friction, 0.02, 0.01)
                    error = float(np.max(np.abs(result["velocity"] - expected)))
                    np.testing.assert_allclose(result["velocity"], expected, rtol=0, atol=1e-11)
                    np.testing.assert_allclose(warm_result["velocity"], expected, rtol=0, atol=1e-11)
                    np.testing.assert_allclose(modal_result["velocity"], expected, rtol=0, atol=1e-11)
                    np.testing.assert_allclose(result["force"], impulse / dt, rtol=1e-8, atol=1e-10)
                    assert result["point_equation_residual"] <= 1e-11
                    condensed_error = None
                    if dense is not None:
                        direct = dense.solve(
                            (jacobian @ free).reshape(-1, 3), gap, previous, dt, 1e9, friction, 0.02, 0.01
                        )
                        condensed_error = float(np.max(np.abs(result["velocity"] - direct["velocity"])))
                        assert condensed_error <= 1e-11
                    rows.append(
                        {
                            "body_dofs": body_dofs,
                            "contact_velocity_dofs": 3 * points,
                            "dt": dt,
                            "speed_scale": speed,
                            "friction": friction,
                            "maximum_velocity_error": error,
                            "maximum_condensed_velocity_difference": condensed_error,
                            "newton_iterations": result["iterations"],
                            "warm_newton_iterations": warm_result["iterations"],
                            "modal_fixed_point_iterations": modal_result["fixed_point_iterations"],
                            "pcg_iterations": result["linear_iterations"],
                        }
                    )
    write_json(args.output / "result.json", {"identity": runtime_identity(), "cases": rows, "passed": True})


if __name__ == "__main__":
    main()
