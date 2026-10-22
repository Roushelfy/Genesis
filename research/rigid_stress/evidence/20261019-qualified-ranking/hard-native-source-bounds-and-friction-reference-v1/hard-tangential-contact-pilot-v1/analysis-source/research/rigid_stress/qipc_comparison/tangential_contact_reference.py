"""Independent force-controlled surface contact using Cerruti compliance."""

from dataclasses import dataclass

from cvxopt import matrix, solvers, spmatrix
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class TangentialContactStep:
    force: np.ndarray
    capacity: np.ndarray
    rigid_increment: np.ndarray
    force_residual: float
    cone_excess: float
    stationarity: float
    complementarity: float
    normal_integration_relative: float
    iterations: int
    solver_status: str


def square_rule(order):
    nodes, weights = leggauss(order)
    x, y = np.meshgrid(nodes / 2, nodes / 2, indexing="ij")
    return np.column_stack([x.reshape(-1), y.reshape(-1)]), np.outer(weights, weights).reshape(-1) / 4


class TangentialContactReference:
    """Incremental vector Coulomb contact for two elastically similar half-spaces.

    Normal Hertz pressure is supplied. Cell forces minimize the Cerruti elastic
    energy of the increment from the prior force, with exact total tangential
    force and local vector traction disks. The equality multiplier is the
    uniform rigid tangential increment. Normal/tangential coupling cancels for
    the declared similar contacting materials. Cell-average pressure is used
    for the discrete Coulomb capacity, with its integration error recorded.
    """

    def __init__(self, grid, maximum_radius, poisson, order=12):
        assert grid >= 5 and grid % 2 == 1 and maximum_radius > 0 and -1 < poisson < 0.5
        self.width = 2 * maximum_radius / (grid - 1)
        coordinates = np.linspace(-maximum_radius, maximum_radius, grid)
        x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
        centers = np.column_stack([x.reshape(-1), y.reshape(-1)])
        self.centers = centers[np.linalg.norm(centers, axis=1) < maximum_radius + self.width / np.sqrt(2)]
        self.poisson = poisson
        self.force = np.zeros((len(self.centers), 2))
        offsets, weights = square_rule(order)
        self.pressure_positions = self.centers[:, None] + self.width * offsets
        self.pressure_weights = weights
        count = len(self.centers)
        tensor = np.empty((count, count, 2, 2))
        for start in range(0, count, 24):
            difference = (
                self.centers[start : start + 24, None, None] - self.centers[None, :, None] - self.width * offsets
            )
            radius = np.linalg.norm(difference, axis=-1)
            diagonal = (1 - poisson) / radius
            outer = poisson * difference[..., :, None] * difference[..., None, :] / radius[..., None, None] ** 3
            kernel = diagonal[..., None, None] * np.eye(2) + outer
            tensor[start : start + 24] = np.einsum("ijqkl,q->ijkl", kernel, weights)
        # The square-cell self integral is isotropic. Its exact integral of
        # 1/r is 4*h*asinh(1), including the Poisson tensor's half trace.
        tensor[np.arange(count), np.arange(count)] = (1 - poisson / 2) * 4 * np.arcsinh(1) / self.width * np.eye(2)
        compliance = tensor.transpose(0, 2, 1, 3).reshape(2 * count, 2 * count)
        np.testing.assert_allclose(compliance, compliance.T, rtol=1e-12, atol=1e-12)
        self.compliance = (compliance + compliance.T) / (2 * np.mean(np.diag(compliance)))
        np.linalg.cholesky(self.compliance)
        solvers.options.update(show_progress=False, abstol=1e-10, reltol=1e-10, feastol=1e-10, maxiters=100)

    def update(self, radius, normal, friction, force):
        assert radius > 0 and normal > 0 and friction > 0
        assert np.linalg.norm(force) < friction * normal
        r2 = np.sum(self.pressure_positions**2, axis=-1) / radius**2
        pressure = 3 * normal / (2 * np.pi * radius**2) * np.sqrt(np.maximum(1 - r2, 0))
        integrated = self.width**2 * (pressure @ self.pressure_weights)
        normal_error = float(abs(integrated.sum() / normal - 1))
        capacity = friction * integrated * (normal / integrated.sum())
        active = np.flatnonzero(capacity > 0)
        dofs = (2 * active[:, None] + np.arange(2)).reshape(-1)
        compliance = self.compliance[np.ix_(dofs, dofs)]
        linear = -self.compliance[dofs] @ self.force.reshape(-1)
        count = len(active)
        rows = (3 * np.arange(count)[:, None] + np.array([1, 2])).reshape(-1)
        columns = np.arange(2 * count)
        constraints = spmatrix([-1.0] * len(rows), rows.tolist(), columns.tolist(), (3 * count, 2 * count))
        limit = np.zeros(3 * count)
        limit[::3] = capacity[active]
        equality = np.tile(np.eye(2), (1, count))
        result = solvers.coneqp(
            matrix(compliance),
            matrix(linear),
            constraints,
            matrix(limit),
            {"l": 0, "q": [3] * count, "s": []},
            matrix(equality),
            matrix(force),
        )
        assert result["status"] == "optimal", (result["status"], result["gap"], result["primal infeasibility"])
        solution = np.array(result["x"])[:, 0]
        dual = np.array(result["z"]).reshape(count, 3)
        multiplier = np.array(result["y"])[:, 0]
        residual = compliance @ solution + linear + equality.T @ multiplier - dual[:, 1:].reshape(-1)
        slack = np.column_stack([capacity[active], solution.reshape(count, 2)])
        force_error = float(np.linalg.norm(equality @ solution - force) / (friction * normal))
        excess = float(np.maximum(np.linalg.norm(solution.reshape(count, 2), axis=1) - capacity[active], 0).max())
        stationarity = float(np.linalg.norm(residual) / max(np.linalg.norm(linear), np.linalg.norm(solution), 1e-12))
        complementarity = float(abs(np.sum(slack * dual)))
        assert force_error < 1e-8 and excess / (friction * normal) < 1e-9
        assert stationarity < 1e-7 and complementarity < 1e-8
        self.force.fill(0)
        self.force[active] = solution.reshape(count, 2)
        return TangentialContactStep(
            self.force.copy(),
            capacity,
            -multiplier,
            force_error,
            excess / (friction * normal),
            stationarity,
            complementarity,
            normal_error,
            result["iterations"],
            result["status"],
        )

    def sources(self, order=4):
        offsets, weights = square_rule(order)
        positions = self.centers[:, None] + self.width * offsets
        forces = self.force[:, None] * weights[None, :, None]
        return positions.reshape(-1, 2), forces.reshape(-1, 2)
