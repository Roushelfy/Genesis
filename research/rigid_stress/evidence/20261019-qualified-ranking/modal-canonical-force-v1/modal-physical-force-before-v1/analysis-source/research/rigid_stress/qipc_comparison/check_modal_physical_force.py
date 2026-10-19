"""Check physical friction at full returned velocities during near-zero-load detachment."""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from .artifacts import runtime_identity, write_json
from .check_matrix_free_pad import DenseFactor
from .modal_pad_dynamics import ModalPadDynamics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    operator = sparse.eye(6, format="csc")
    jacobian = sparse.eye(6, format="csr")
    factor = DenseFactor(np.eye(6))
    previous = np.full(2, 0.001)
    rows = []
    for scale in (1e-13, 1e-11, 1e-9):
        for dt in (0.001, 0.002):
            for warm in (False, True):
                solver = ModalPadDynamics(
                    operator, factor, jacobian, np.eye(6)[:, [2, 5]],
                    np.array([1.0, -1.0]), np.full(2, 1e-6), warm_start=warm,
                )
                free = scale * np.array([1.0, 1 / 3, 0.0, -0.5, -0.2, 0.0])
                result = solver.solve(free, np.full(2, 1e-5), previous, dt, 1e9, 0.3, 0.1, 1.0)
                actual = (jacobian @ result["full_velocity"]).reshape(-1, 3)
                speed = np.linalg.norm(actual[:, :2], axis=1)
                assert np.all(speed < 1e-4)
                expected = np.zeros_like(actual)
                expected[:, :2] = -(0.3 * previous * (2 / 1e-4 - speed / 1e-8))[:, None] * actual[:, :2]
                force_error = float(np.linalg.norm(result["force"] - expected) / max(np.linalg.norm(result["force"]), 1e-12))
                defect = operator @ (result["full_velocity"] - free) - dt * jacobian.T @ result["force"].reshape(-1)
                residual = float(np.sqrt(defect @ factor.solve(defect)))
                tolerance = result["whitened_momentum_tolerance"]
                # The scalar smooth-friction equation is B*s-C*s^2=s_free.
                # This stable quadratic root is independent of the iterative solver.
                free_speed = np.linalg.norm(free.reshape(-1, 3)[:, :2], axis=1)
                coefficient = 1 + 2 * dt * 0.3 * previous / 1e-4
                quadratic = dt * 0.3 * previous / 1e-8
                exact_speed = 2 * free_speed / (coefficient + np.sqrt(coefficient**2 - 4 * quadratic * free_speed))
                exact = free.reshape(-1, 3).copy()
                exact[:, :2] *= (exact_speed / free_speed)[:, None]
                velocity_error = float(np.linalg.norm(actual - exact))
                checks = {
                    "physical_friction_law": force_error <= 1e-7,
                    "unchanged_whitened_momentum": residual <= tolerance,
                    "analytic_velocity": velocity_error <= 2 * tolerance,
                    "open_normal_contact": np.array_equal(result["force"][:, 2], np.zeros(2)),
                }
                rows.append({
                    "free_speed_scale": scale, "dt": dt, "warm_start": warm,
                    "force_relative_error": force_error, "force_norm_n": float(np.linalg.norm(result["force"])),
                    "momentum_residual": residual, "momentum_tolerance": tolerance,
                    "analytic_velocity_error": velocity_error, "checks": checks,
                })
    passed = all(all(row["checks"].values()) for row in rows)
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "passed": passed, "cases": rows,
        "scope": "Detached current normal contact with nonzero lagged friction bound, independent full-DOF law evaluation, unchanged1e-7 friction gate and original momentum tolerance. Stable analytic quadratic root controls the physical velocity. No force is clipped and no environment or transient is omitted.",
    })
    if not passed:
        raise SystemExit("Modal force and actual returned velocity do not satisfy the declared physical checks")


if __name__ == "__main__":
    main()
