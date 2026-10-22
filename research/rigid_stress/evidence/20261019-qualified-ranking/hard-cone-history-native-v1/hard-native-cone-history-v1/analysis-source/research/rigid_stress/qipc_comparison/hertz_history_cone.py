"""A small variational radial-memory control with a continuous surface bound.

Bounding the line value alone does not bound its inverse-Abel surface traction
under changing vector history. Bound each slope in squared radius instead.
The two-component total force remains an equality. This is an approximate
friction model, not an independently qualified general contact solution.
"""

import numpy as np

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
        from cvxopt import matrix

        self.nodes = np.linspace(0, 1, intervals + 1)
        self.weights = line_weights(self.nodes)
        self.widths = np.diff(self.nodes**2)
        self.line = np.zeros((intervals + 1, 2))
        self.coefficients = np.zeros((intervals, 2))
        self.radius, self.scale, self.residual, self.traction_bound = 0.0, 0.0, 0.0, 0.0
        self.solver = {}
        self.integration = np.triu(np.ones((intervals, intervals))) * self.widths[None]
        gradient = np.eye(2 * intervals)
        cone = np.zeros((3 * intervals, 2 * intervals))
        bound = np.zeros(3 * intervals)
        for i in range(intervals):
            cone[3 * i + 1 : 3 * i + 3] = gradient[2 * i : 2 * i + 2]
            bound[3 * i] = 1
        self.quadratic = matrix(np.kron(self.integration.T @ (self.weights[:-1, None] * self.integration), np.eye(2)))
        self.cone, self.bound = matrix(cone), matrix(bound)
        self.equality = matrix(np.kron(self.weights[None, :-1] @ self.integration, np.eye(2)))
        self.dimensions = {"l": 0, "q": [3] * intervals, "s": []}

    def update(self, radius, normal, friction, force, angle=0.0, reset=False):
        from cvxopt import matrix, solvers

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
            previous = dict(solvers.options)
            try:
                solvers.options.update(
                    show_progress=False, abstol=1e-12, reltol=1e-12, feastol=1e-13, maxiters=100, refinement=3
                )
                result = solvers.coneqp(
                    self.quadratic,
                    matrix((-(self.integration.T @ (self.weights[:-1, None] * prior[:-1]))).reshape(-1)),
                    self.cone,
                    self.bound,
                    self.dimensions,
                    self.equality,
                    matrix(target),
                    kktsolver="ldl",
                )
            finally:
                solvers.options.clear()
                solvers.options.update(previous)
            self.solver = {
                key: result[key]
                for key in ["status", "iterations", "gap", "primal infeasibility", "dual infeasibility"]
            }
            assert result["status"] == "optimal", self.solver
            self.line[:-1] = self.integration @ np.asarray(result["x"]).reshape(-1, 2)
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
