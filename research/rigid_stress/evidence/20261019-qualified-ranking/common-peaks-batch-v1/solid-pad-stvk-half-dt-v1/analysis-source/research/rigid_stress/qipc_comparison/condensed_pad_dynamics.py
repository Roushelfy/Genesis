"""Independent contact-space Newton solve for implicit linear-solid pad dynamics.

For A=M+dt^2 K, C=J A^-1 J^T maps point impulses to relative point velocities.
The Cholesky coordinates C=L L^T give a convex identity-quadratic objective
without pinning rigid modes or modifying the physical pad/friction potential.
"""

import numpy as np
from scipy import linalg


def potential(velocity, gap, sign, area, previous_normal, dt, stiffness, friction):
    """Return physical potential, opposing impulse and velocity Hessian blocks."""
    compression = np.maximum(-(gap + dt * sign * velocity[:, 2]), 0)
    spring = stiffness * area
    impulse = np.zeros_like(velocity)
    impulse[:, 2] = dt * sign * spring * compression
    hessian = np.zeros((len(velocity), 3, 3))
    hessian[:, 2, 2] = dt**2 * spring * (compression > 0)
    energy = 0.5 * np.dot(spring, compression**2)
    epsilon = 1e-4
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    direction = velocity[:, :2] / np.maximum(speed[:, None], 1e-30)
    bound = dt * friction * previous_normal
    smooth = speed < epsilon
    ratio = np.minimum(speed / epsilon, 1)
    impulse[:, :2] = -(bound * (2 * ratio - ratio**2))[:, None] * direction
    energy += np.dot(bound, np.where(smooth, speed**2 / epsilon - speed**3 / (3 * epsilon**2), speed - epsilon / 3))
    isotropic = np.where(smooth, 2 / epsilon - speed / epsilon**2, 1 / np.maximum(speed, 1e-30))
    radial = np.where(smooth, speed / epsilon**2, 1 / np.maximum(speed, 1e-30))
    hessian[:, :2, :2] = bound[:, None, None] * (
        isotropic[:, None, None] * np.eye(2) - radial[:, None, None] * direction[:, :, None] * direction[:, None, :]
    )
    return float(energy), impulse, hessian


class CondensedPadDynamics:
    def __init__(self, compliance, sign, area, warm_start=False):
        self.sign, self.area = np.asarray(sign), np.asarray(area)
        assert compliance.shape == (3 * len(sign), 3 * len(sign))
        self.reciprocity_error = float(np.linalg.norm(compliance - compliance.T) / np.linalg.norm(compliance))
        assert self.reciprocity_error <= 1e-8
        self.compliance = (compliance + compliance.T) / 2
        # A failed factor is evidence to inspect, not a reason to add jitter.
        self.lower = linalg.cholesky(self.compliance, lower=True)
        self.identity = np.eye(len(compliance))
        self.warm_start, self.previous_coordinate = warm_start, None

    def solve(self, free_velocity, gap, previous_normal, dt, stiffness, friction, load, mass):
        target = linalg.solve_triangular(self.lower, free_velocity.reshape(-1), lower=True)
        coordinate = target.copy() if self.previous_coordinate is None else self.previous_coordinate.copy()
        tolerance = 1e-10 * dt * max(load, 1e-3) / np.sqrt(mass)

        def evaluate(value):
            velocity = (self.lower @ value).reshape(-1, 3)
            energy, impulse, hessian = potential(
                velocity, gap, self.sign, self.area, previous_normal, dt, stiffness, friction
            )
            delta = value - target
            gradient = delta - self.lower.T @ impulse.reshape(-1)
            return energy + 0.5 * np.dot(delta, delta), gradient, hessian, velocity, impulse

        for iteration in range(80):
            energy, gradient, blocks, velocity, impulse = evaluate(coordinate)
            residual = float(np.linalg.norm(gradient))
            if residual <= tolerance:
                if self.warm_start:
                    self.previous_coordinate = coordinate.copy()
                return {
                    "velocity": velocity,
                    "force": impulse / dt,
                    "iterations": iteration + 1,
                    "whitened_momentum_residual": residual,
                    "whitened_momentum_tolerance": tolerance,
                    "point_equation_residual": float(
                        np.linalg.norm(
                            velocity.reshape(-1) - free_velocity.reshape(-1) - self.compliance @ impulse.reshape(-1)
                        )
                    ),
                }
            product = np.einsum("qij,qjk->qik", blocks, self.lower.reshape(-1, 3, len(coordinate)))
            hessian = self.identity + self.lower.T @ product.reshape(len(coordinate), -1)
            direction = linalg.cho_solve(linalg.cho_factor(hessian), -gradient)
            slope = float(np.dot(gradient, direction))
            assert slope < 0 and np.isfinite(slope)
            step = 1.0
            for _ in range(40):
                candidate = coordinate + step * direction
                candidate_energy = evaluate(candidate)[0]
                rounding = 16 * np.finfo(float).eps * max(abs(energy), 1e-30)
                if candidate_energy <= energy + 1e-4 * step * slope + rounding:
                    coordinate = candidate
                    break
                step *= 0.5
            else:
                raise RuntimeError("Independent pad Newton line search exhausted")
        raise RuntimeError("Independent pad Newton iteration limit exhausted")
