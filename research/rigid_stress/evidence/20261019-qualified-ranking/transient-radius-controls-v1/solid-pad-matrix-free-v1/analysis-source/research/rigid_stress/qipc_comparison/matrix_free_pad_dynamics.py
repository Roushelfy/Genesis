"""Full-body Newton/PCG pad solve without a dense or invertible contact compliance."""

import numpy as np

from .condensed_pad_dynamics import potential


class MatrixFreePadDynamics:
    def __init__(self, operator, factor, jacobian, sign, area):
        self.operator, self.factor, self.jacobian = operator, factor, jacobian
        self.sign, self.area = np.asarray(sign), np.asarray(area)

    def solve(self, free_velocity, gap, previous_normal, dt, stiffness, friction, load, mass):
        coordinate = free_velocity.copy()
        tolerance = 1e-10 * dt * max(load, 1e-3) / np.sqrt(mass)

        def evaluate(value):
            velocity = (self.jacobian @ value).reshape(-1, 3)
            energy, impulse, hessian = potential(
                velocity, gap, self.sign, self.area, previous_normal, dt, stiffness, friction
            )
            delta = value - free_velocity
            elastic = self.operator @ delta
            gradient = elastic - self.jacobian.T @ impulse.reshape(-1)
            return energy + 0.5 * np.dot(delta, elastic), gradient, hessian, velocity, impulse

        linear_iterations = []
        for iteration in range(80):
            energy, gradient, blocks, velocity, impulse = evaluate(coordinate)
            preconditioned = self.factor.solve(gradient)
            squared = float(np.dot(gradient, preconditioned))
            assert squared >= 0 and np.isfinite(squared)
            residual = np.sqrt(squared)
            if residual <= tolerance:
                predicted = self.jacobian @ self.factor.solve(self.jacobian.T @ impulse.reshape(-1))
                return {
                    "velocity": velocity,
                    "force": impulse / dt,
                    "iterations": iteration + 1,
                    "linear_iterations": linear_iterations,
                    "whitened_momentum_residual": float(residual),
                    "whitened_momentum_tolerance": float(tolerance),
                    "point_equation_residual": float(
                        np.linalg.norm(velocity.reshape(-1) - self.jacobian @ free_velocity - predicted)
                    ),
                }

            def product(value):
                point_value = (self.jacobian @ value).reshape(-1, 3)
                contact_value = np.einsum("qij,qj->qi", blocks, point_value).reshape(-1)
                return self.operator @ value + self.jacobian.T @ contact_value

            direction = np.zeros_like(coordinate)
            linear_residual, search = -gradient.copy(), -preconditioned
            rho = squared
            linear_target = max(1e-4 * residual, 0.01 * tolerance)
            for inner in range(200):
                action = product(search)
                curvature = float(np.dot(search, action))
                assert curvature > 0 and np.isfinite(curvature)
                alpha = rho / curvature
                direction += alpha * search
                linear_residual -= alpha * action
                z = self.factor.solve(linear_residual)
                new_rho = float(np.dot(linear_residual, z))
                assert new_rho >= 0 and np.isfinite(new_rho)
                if np.sqrt(new_rho) <= linear_target:
                    # Verify the actual linear residual, not only its recurrence.
                    linear_residual = -gradient - product(direction)
                    z = self.factor.solve(linear_residual)
                    new_rho = float(np.dot(linear_residual, z))
                    assert new_rho >= 0 and np.isfinite(new_rho)
                    if np.sqrt(new_rho) <= linear_target:
                        break
                    search, rho = z.copy(), new_rho
                    continue
                search = z + (new_rho / rho) * search
                rho = new_rho
            else:
                raise RuntimeError("Full-body pad PCG iteration limit exhausted")
            linear_iterations.append(inner + 1)
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
                raise RuntimeError("Full-body pad Newton line search exhausted")
        raise RuntimeError("Full-body pad Newton iteration limit exhausted")
