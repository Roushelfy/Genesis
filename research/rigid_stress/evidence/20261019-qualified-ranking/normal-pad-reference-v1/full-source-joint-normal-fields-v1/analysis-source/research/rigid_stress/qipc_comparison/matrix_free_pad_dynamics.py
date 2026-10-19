"""Contact-impulse Newton/PCG without dense compliance or stiff-body subtraction.

The unknown impulse p produces v=v_free+A^-1 J^T p. With contact impulse
I(v), solve p-I(v)=0. Its Jacobian I+H C, C=J A^-1 J^T and H=-dI/dv,
is reduced by Woodbury to the SPD operator I+sqrt(H) C sqrt(H). Redundant
quadrature rows are allowed; no inverse or regularization of C is used.
"""

import numpy as np

from .condensed_pad_dynamics import potential


class MatrixFreePadDynamics:
    def __init__(self, operator, factor, jacobian, sign, area):
        assert operator.shape[0] == jacobian.shape[1]
        self.factor, self.jacobian = factor, jacobian
        self.sign, self.area = np.asarray(sign), np.asarray(area)

    def compliance(self, value):
        return self.jacobian @ self.factor.solve(self.jacobian.T @ value)

    def solve(self, free_velocity, gap, previous_normal, dt, stiffness, friction, load, mass):
        free = self.jacobian @ free_velocity
        coordinate = np.zeros_like(free)
        tolerance = 1e-10 * dt * max(load, 1e-3) / np.sqrt(mass)

        def evaluate(value):
            increment = self.compliance(value)
            velocity = (free + increment).reshape(-1, 3)
            energy, impulse, hessian = potential(
                velocity, gap, self.sign, self.area, previous_normal, dt, stiffness, friction
            )
            defect = value - impulse.reshape(-1)
            return energy + 0.5 * np.dot(value, increment), defect, hessian, velocity, impulse

        linear_iterations = []
        for iteration in range(80):
            energy, defect, blocks, velocity, impulse = evaluate(coordinate)
            nodal_defect = self.jacobian.T @ defect
            preconditioned = self.factor.solve(nodal_defect)
            squared = float(np.dot(nodal_defect, preconditioned))
            assert squared >= 0 and np.isfinite(squared)
            residual = np.sqrt(squared)
            contact_defect = self.jacobian @ preconditioned
            if residual <= tolerance:
                return {
                    "velocity": velocity,
                    "force": impulse / dt,
                    "iterations": iteration + 1,
                    "linear_iterations": linear_iterations,
                    "whitened_momentum_residual": float(residual),
                    "whitened_momentum_tolerance": float(tolerance),
                    "point_equation_residual": float(np.linalg.norm(contact_defect)),
                }
            eigenvalues, eigenvectors = np.linalg.eigh(blocks)
            assert eigenvalues.min() >= -64 * np.finfo(float).eps * max(eigenvalues.max(), 1e-30)
            root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None]) @ eigenvectors.transpose(0, 2, 1)

            def root_product(value):
                return np.einsum("qij,qj->qi", root, value.reshape(-1, 3)).reshape(-1)

            def product(value):
                return value + root_product(self.compliance(root_product(value)))

            rhs = root_product(contact_defect)
            solution = np.zeros_like(rhs)
            linear_residual, search = rhs.copy(), rhs.copy()
            rho = float(np.dot(rhs, rhs))
            linear_target = 1e-4 * np.sqrt(rho)
            if rho > 0:
                for inner in range(200):
                    action = product(search)
                    curvature = float(np.dot(search, action))
                    assert curvature > 0 and np.isfinite(curvature)
                    alpha = rho / curvature
                    solution += alpha * search
                    linear_residual -= alpha * action
                    new_rho = float(np.dot(linear_residual, linear_residual))
                    if np.sqrt(new_rho) <= linear_target:
                        linear_residual = rhs - product(solution)
                        new_rho = float(np.dot(linear_residual, linear_residual))
                        if np.sqrt(new_rho) <= linear_target:
                            break
                        search, rho = linear_residual.copy(), new_rho
                        continue
                    search = linear_residual + (new_rho / rho) * search
                    rho = new_rho
                else:
                    raise RuntimeError("Contact-impulse pad PCG iteration limit exhausted")
                linear_iterations.append(inner + 1)
            else:
                linear_iterations.append(0)
            direction = -defect + root_product(solution)
            slope = float(np.dot(contact_defect, direction))
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
                raise RuntimeError(
                    f"Contact-impulse pad Newton line search exhausted: residual={residual}, tolerance={tolerance}"
                )
        raise RuntimeError("Contact-impulse pad Newton iteration limit exhausted")
