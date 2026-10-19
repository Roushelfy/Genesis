"""Full dynamic compliance fixed point with a small modal contact Newton solve.

C=J A^-1 J^T is split exactly into B B^T and C-B B^T, where B is the
A-orthonormal image of selected free modes. Only the latter part is lagged
inside the fixed-point iteration. Final full-compliance force, velocity and
preconditioned momentum defects must converge; no mode is fixed or discarded.
The returned force is evaluated at the full returned velocity. The auxiliary
response impulse is retained separately, with their mismatch bounded by the
original preconditioned momentum test.
"""

import numpy as np
from scipy import linalg

from .condensed_pad_dynamics import potential


class ModalPadDynamics:
    def __init__(self, operator, factor, jacobian, modes, sign, area, warm_start=False):
        self.factor, self.jacobian = factor, jacobian
        self.sign, self.area = np.asarray(sign), np.asarray(area)
        gram = modes.T @ (operator @ modes)
        lower = linalg.cholesky((gram + gram.T) / 2, lower=True)
        self.modal = linalg.solve_triangular(lower, np.asarray(jacobian @ modes).T, lower=True).T
        self.identity = np.eye(modes.shape[1])
        self.warm_start, self.previous_impulse = warm_start, None

    def compliance(self, value):
        return self.jacobian @ self.factor.solve(self.jacobian.T @ value)

    def solve(self, free_velocity, gap, previous_normal, dt, stiffness, friction, load, mass):
        free = self.jacobian @ free_velocity
        impulse = np.zeros_like(free) if self.previous_impulse is None else self.previous_impulse.copy()
        response = self.compliance(impulse) if np.any(impulse) else np.zeros_like(impulse)
        tolerance = 1e-10 * dt * max(load, 1e-3) / np.sqrt(mass)
        newton_iterations = []
        previous_increment, relaxation = None, 1.0
        relaxation_history = []
        for outer in range(80):
            old_impulse, old_response = impulse, response
            base = free + response - self.modal @ (self.modal.T @ impulse)
            coordinate = self.modal.T @ impulse

            def evaluate(value):
                velocity = (base + self.modal @ value).reshape(-1, 3)
                energy, force_impulse, blocks = potential(
                    velocity, gap, self.sign, self.area, previous_normal, dt, stiffness, friction
                )
                gradient = value - self.modal.T @ force_impulse.reshape(-1)
                return energy + 0.5 * np.dot(value, value), gradient, blocks, velocity, force_impulse

            for inner in range(80):
                energy, gradient, blocks, velocity, current = evaluate(coordinate)
                if np.linalg.norm(gradient) <= tolerance:
                    break
                product = np.einsum("qij,qjk->qik", blocks, self.modal.reshape(-1, 3, len(coordinate)))
                hessian = self.identity + self.modal.T @ product.reshape(len(free), -1)
                direction = linalg.cho_solve(linalg.cho_factor(hessian), -gradient)
                slope = float(np.dot(gradient, direction))
                assert slope < 0 and np.isfinite(slope)
                step = 1.0
                for _ in range(40):
                    candidate = coordinate + step * direction
                    rounding = 16 * np.finfo(float).eps * max(abs(energy), 1e-30)
                    if evaluate(candidate)[0] <= energy + 1e-4 * step * slope + rounding:
                        coordinate = candidate
                        break
                    step *= 0.5
                else:
                    raise RuntimeError("Modal pad Newton line search exhausted")
            else:
                raise RuntimeError("Modal pad Newton iteration limit exhausted")
            newton_iterations.append(inner + 1)
            impulse = current.reshape(-1)
            body_response = self.factor.solve(self.jacobian.T @ impulse)
            response = self.jacobian @ body_response
            full_velocity = free_velocity + body_response
            actual = (self.jacobian @ full_velocity).reshape(-1, 3)
            actual_impulse = potential(actual, gap, self.sign, self.area, previous_normal, dt, stiffness, friction)[1]
            defect = impulse - actual_impulse.reshape(-1)
            force_error = float(np.linalg.norm(defect) / max(np.linalg.norm(impulse), dt * max(load, 1e-3) * 1e-3))
            point_error = float(np.max(np.abs(actual - velocity)))
            if force_error <= 1e-10 and point_error <= 1e-11:
                nodal_defect = self.jacobian.T @ defect
                squared = float(np.dot(nodal_defect, self.factor.solve(nodal_defect)))
                assert squared >= 0
                residual = np.sqrt(squared)
                if residual <= tolerance:
                    if self.warm_start:
                        self.previous_impulse = actual_impulse.reshape(-1).copy()
                    # Evaluate the physical law at the returned full-DOF
                    # velocity. The auxiliary modal velocity is only an
                    # iterate; its tiny difference can matter relatively at
                    # detachment. The momentum defect above measures the
                    # mismatch with this physical impulse under the original
                    # tolerance, without clipping or changing the force law.
                    return {
                        "velocity": actual,
                        "full_velocity": full_velocity,
                        "force": actual_impulse / dt,
                        "response_impulse": current,
                        "force_evaluation": "full_velocity",
                        "auxiliary_force_defect_n": float(np.linalg.norm(defect) / dt),
                        "iterations": sum(newton_iterations),
                        "fixed_point_iterations": outer + 1,
                        "modal_newton_iterations": newton_iterations,
                        "relaxation_history": relaxation_history,
                        "fixed_point_force_relative_error": force_error,
                        "whitened_momentum_residual": float(residual),
                        "whitened_momentum_tolerance": float(tolerance),
                        "point_equation_residual": float(np.linalg.norm(actual - velocity)),
                    }
            # Aitken damping stabilizes a noncontractive elastic complement;
            # acceptance still uses the complete, unrelaxed physical equations.
            increment = impulse - old_impulse
            if previous_increment is not None:
                change = increment - previous_increment
                denominator = float(np.dot(change, change))
                if denominator > 0:
                    relaxation = float(
                        np.clip(-relaxation * np.dot(previous_increment, change) / denominator, 0.05, 1.0)
                    )
            relaxation_history.append(relaxation)
            impulse = old_impulse + relaxation * increment
            response = old_response + relaxation * (response - old_response)
            previous_increment = increment
        raise RuntimeError(
            f"Full-compliance modal iteration limit: force_error={force_error}, point_error={point_error}, relaxation={relaxation_history[-5:]}"
        )
