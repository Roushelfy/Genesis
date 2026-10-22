"""A small variational radial-memory control with a continuous surface bound.

Bounding the line value alone does not bound its inverse-Abel surface traction
under changing vector history. Bound each slope in squared radius instead.
The two-component total force remains an equality. This is an approximate
friction model, not an independently qualified general contact solution.
"""

import numpy as np
from scipy.optimize import minimize

from .hertz_tangential import elliptic_tangential_stress


def line_weights(nodes):
    lower, upper = nodes[:-1], nodes[1:]
    gauss = np.array([-1, 1]) / np.sqrt(3)
    radii = lower[:, None] + (upper - lower)[:, None] * (gauss + 1) / 2
    fraction = (radii**2 - lower[:, None] ** 2) / (upper[:, None] ** 2 - lower[:, None] ** 2)
    weights = np.zeros(len(nodes))
    weights[:-1] += 0.75 * (upper - lower) * np.sum(1 - fraction, axis=1)
    weights[1:] += 0.75 * (upper - lower) * np.sum(fraction, axis=1)
    np.testing.assert_allclose(weights @ (1 - nodes**2), 1, rtol=0, atol=1e-14)
    return weights


class ConeFrictionHistory:
    """Offline force-controlled weighted projection onto vector slope disks."""

    def __init__(self, intervals):
        self.nodes = np.linspace(0, 1, intervals + 1)
        self.weights = line_weights(self.nodes)
        self.widths = np.diff(self.nodes**2)
        self.line = np.zeros((intervals + 1, 2))
        self.coefficients = np.zeros((intervals, 2))
        self.radius, self.scale, self.residual, self.traction_bound = 0.0, 0.0, 0.0, 0.0
        self.solver = {}
        self.difference = (np.eye(intervals) - np.eye(intervals, k=1)) / self.widths[:, None]
        self.equality = np.kron(self.weights[None, :-1], np.eye(2))

    def update(self, radius, normal, friction, force, angle=0.0, reset=False):
        force = np.asarray(force, dtype=float)
        assert np.isfinite([radius, normal, friction, *force, angle]).all()
        assert radius >= 0 and normal >= 0 and friction >= 0
        assert np.linalg.norm(force) <= friction * normal * (1 + 1e-12)
        if normal == 0 or friction == 0:
            self.line.fill(0)
            self.coefficients.fill(0)
            self.radius, self.scale, self.residual, self.traction_bound = 0.0, 0.0, 0.0, 0.0
            self.solver = {}
            return
        assert radius > 0
        scale = 0.75 * friction * normal / radius
        prior = np.zeros_like(self.line)
        if not reset and self.radius > 0:
            positions = (self.nodes * radius / self.radius) ** 2
            prior = np.column_stack([np.interp(positions, self.nodes**2, self.line[:, i], right=0) for i in range(2)])
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            prior = (prior @ rotation.T) * (self.scale / scale)
        target = force / (friction * normal)
        magnitude = np.linalg.norm(target)
        if magnitude >= 1 - 1e-12:
            self.line = (1 - self.nodes**2)[:, None] * target / magnitude
            self.solver = {"status": "gross_sliding", "iterations": 0}
        else:
            weight = self.weights[:-1] / self.weights[:-1].mean()

            def objective(value):
                difference = value.reshape(-1, 2) - prior[:-1]
                return 0.5 * np.sum(weight[:, None] * difference * difference)

            def gradient(value):
                return (weight[:, None] * (value.reshape(-1, 2) - prior[:-1])).reshape(-1)

            def constraints(value):
                slope = self.difference @ value.reshape(-1, 2)
                return 1 - np.sum(slope * slope, axis=1)

            def constraint_gradient(value):
                slope = self.difference @ value.reshape(-1, 2)
                return (-2 * self.difference[:, :, None] * slope[:, None, :]).reshape(len(self.widths), -1)

            initial = ((1 - self.nodes[:-1] ** 2)[:, None] * target).reshape(-1)
            result = minimize(
                objective,
                initial,
                jac=gradient,
                method="SLSQP",
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda value: self.equality @ value - target,
                        "jac": lambda value: self.equality,
                    },
                    {"type": "ineq", "fun": constraints, "jac": constraint_gradient},
                ],
                options={"ftol": 1e-13, "maxiter": 1000},
            )
            self.solver = {
                "status": str(result.message),
                "success": bool(result.success),
                "iterations": int(result.nit),
                "objective": float(result.fun),
            }
            assert result.success, self.solver
            self.line[:-1] = result.x.reshape(-1, 2)
            self.line[-1] = 0
        slopes = (self.line[:-1] - self.line[1:]) / self.widths[:, None]
        self.residual = float(np.linalg.norm(self.weights @ self.line - target))
        self.traction_bound = float(np.linalg.norm(slopes, axis=1).max())
        assert self.residual < 1e-9, self.residual
        assert self.traction_bound <= 1 + 1e-10, self.traction_bound
        self.coefficients[:-1] = slopes[:-1] - slopes[1:]
        self.coefficients[-1] = slopes[-1]
        self.radius, self.scale = radius, scale

    def traction(self, radius):
        r2 = np.asarray(radius)[:, None] ** 2
        weights = np.sqrt(np.maximum(self.nodes[None, 1:] ** 2 - r2, 0)) - np.sqrt(
            np.maximum(self.nodes[None, :-1] ** 2 - r2, 0)
        )
        slopes = (self.line[:-1] - self.line[1:]) / self.widths[:, None]
        return weights @ slopes

    def stress(self, points, axes, peak, friction, poisson=0.3, order=48):
        result = np.zeros((len(points), 3, 3))
        for radius, coefficient in zip(self.nodes[1:], self.coefficients):
            if np.any(coefficient):
                result += elliptic_tangential_stress(
                    points, axes * radius, coefficient, peak * friction * radius, poisson, order
                )
        return result
