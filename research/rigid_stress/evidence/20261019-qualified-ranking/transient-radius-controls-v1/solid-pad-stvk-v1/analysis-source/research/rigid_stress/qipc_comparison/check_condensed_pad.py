"""Analytic controls for the independent solid contact-space dynamic solver."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import runtime_identity, write_json
from .condensed_pad_dynamics import CondensedPadDynamics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mass, normal, friction = 0.01, 0.005, 0.3
    solver = CondensedPadDynamics(np.eye(3) / mass, np.ones(1), np.full(1, 1e-6))
    cases = []
    for dt in (0.001, 0.002):
        for name, coefficient, body_force, speed in (
            ("normal", 0.0, 0.0, 0.0),
            ("smooth-creep", friction, 0.0005, 1e-4 * (1 - np.sqrt(1 - 0.0005 / (friction * normal)))),
            ("saturated-slip", friction, friction * normal, 3e-4),
        ):
            expected = np.array([[speed, 0.0, 0.0]])
            free = expected + dt * np.array([[body_force, 0.0, -normal]]) / mass
            result = solver.solve(free, np.array([-5e-6]), np.array([normal]), dt, 1e9, coefficient, normal, mass)
            error = float(np.max(np.abs(result["velocity"] - expected)))
            np.testing.assert_allclose(result["velocity"], expected, rtol=0, atol=1e-11)
            np.testing.assert_allclose(result["force"], [[-body_force, 0.0, normal]], rtol=0, atol=1e-10)
            cases.append({"case": name, "dt": dt, "maximum_velocity_error": error, "iterations": result["iterations"]})
    generator = np.random.default_rng(17)
    block = generator.normal(size=(12, 12))
    compliance = (np.eye(12) + 0.002 * block @ block.T) / mass
    sign, area = np.array([1.0, -1.0, 1.0, -1.0]), np.array([1.0, 1.5, 0.8, 1.2]) * 1e-6
    gap, dt = -np.array([4.0, 6.0, 8.0, 5.0]) * 1e-6, 0.001
    free = generator.normal(size=(4, 3)) * 1e-5
    diagonal = np.zeros(12)
    diagonal[2::3] = dt**2 * 1e9 * area
    impulse = np.zeros(12)
    impulse[2::3] = -dt * sign * 1e9 * area * gap
    expected = np.linalg.solve(np.eye(12) + compliance * diagonal[None], free.reshape(-1) + compliance @ impulse)
    assert (gap + dt * sign * expected[2::3] < 0).all()
    solver = CondensedPadDynamics(compliance, sign, area)
    result = solver.solve(free, gap, np.zeros(4), dt, 1e9, 0.0, 0.02, 4 * mass)
    error = float(np.max(np.abs(result["velocity"].reshape(-1) - expected)))
    assert error < 1e-11 and result["point_equation_residual"] < 1e-11
    cases.append({"case": "coupled-normal-linear-system", "maximum_velocity_error": error})
    write_json(args.output / "result.json", {"identity": runtime_identity(), "cases": cases, "passed": True})


if __name__ == "__main__":
    main()
