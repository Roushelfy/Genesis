"""Analytic velocity/force checks of the research native-pad boundary condition."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .rigid_pad_coupling import RigidPadState, prepare_geometry, solve


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    points = np.array([[x, y, z] for z in (-0.01, 0.01) for x in (-0.001, 0.001) for y in (-0.001, 0.001)])
    state = RigidPadState.allocate(points, np.full(8, 1e-6))
    prepare_geometry(np.zeros(3), np.eye(3), np.zeros(3), -0.009995, 0.009995, state)
    state.previous_normal.from_numpy(np.full(8, 0.005))
    mass = np.diag([0.01] * 3 + [0.003] * 3 + [0.05])
    rows = []
    for dt in (0.001, 0.002):
        for friction in (0.0, 0.3):
            body_force = 0.001 if friction else 0.0
            speed = 1e-4 * (1 - np.sqrt(1 - body_force / (friction * 0.04))) if friction else 0.0
            predicted = np.zeros(7)
            predicted[0] = speed + dt * body_force / 0.01
            predicted[6] = dt * 0.02 / 0.05
            iterations, residual = solve(state, mass, predicted, dt, 1e9, friction, 0.02)
            velocity = qd_to_numpy(state.velocity, transpose=True)
            force = qd_to_numpy(state.force, transpose=True)
            expected = np.zeros(7)
            expected[0] = speed
            np.testing.assert_allclose(velocity, expected, rtol=0, atol=1e-11)
            np.testing.assert_allclose(force[:4, 2].sum(), 0.02, rtol=0, atol=1e-10)
            np.testing.assert_allclose(force[4:, 2].sum(), -0.02, rtol=0, atol=1e-10)
            np.testing.assert_allclose(force[:, :2].sum(axis=0), [-body_force, 0], rtol=0, atol=1e-10)
            rows.append(
                {
                    "dt": dt,
                    "friction": friction,
                    "iterations": iterations,
                    "momentum_residual": residual,
                    "velocity_maximum_error": float(np.max(np.abs(velocity - expected))),
                    "passed": True,
                }
            )
    # A nonsymmetric velocity excites every normal translation/rotation coupling.
    # All points stay compressed, making this an independent exact linear solve.
    dt = 0.001
    sign = np.where(points[:, 2] > 0, -1.0, 1.0)
    jacobian = np.column_stack(
        (
            np.zeros((8, 2)),
            sign,
            sign * points[:, 1] / 0.03,
            -sign * points[:, 0] / 0.03,
            np.zeros(8),
            -(points[:, 2] > 0).astype(float),
        )
    )
    predicted = np.array([2.0, -1.0, 0.7, -1.1, 0.4, 0.2, -0.6]) * 1e-5
    expected = np.linalg.solve(
        mass + dt**2 * 1000 * jacobian.T @ jacobian, mass @ predicted + dt * 0.005 * jacobian.sum(axis=0)
    )
    assert (-5e-6 + dt * jacobian @ expected < 0).all()
    iterations, residual = solve(state, mass, predicted, dt, 1e9, 0.0, 0.02)
    velocity = qd_to_numpy(state.velocity, transpose=True)
    np.testing.assert_allclose(velocity, expected, rtol=0, atol=1e-11)
    rows.append(
        {
            "case": "coupled-normal-linear-system",
            "iterations": iterations,
            "momentum_residual": residual,
            "velocity_maximum_error": float(np.max(np.abs(velocity - expected))),
            "passed": True,
        }
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "cases": rows,
            "scope": "Analytic balanced compression and steady smoothed-Coulomb creep; native integration not yet tested.",
        },
    )


if __name__ == "__main__":
    main()
