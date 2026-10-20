"""CPU validation of the pad energy after eliminating its local spring variables."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky


@dataclass
class ReducedPadResult:
    point_force: np.ndarray
    nodal_force: np.ndarray
    coordinates: np.ndarray
    elastic_gap: np.ndarray
    rounds: list
    seconds: float
    wrench_error: float
    stationarity: float
    law_error: float


class ReducedPadEquilibrium:
    """Retain full regional compliance and the four free plate/link coordinates.

    With C=L L^T, dimensionless elastic coordinates y and rigid coordinates q,
    minimize

        1/2 y^T y + 1/2 sum_i w_i [-g_i - (R L y)_i - (R Q q)_i]_+^2 + t^T q.

    Eliminating each independent pad spring is exact. The elastic coordinates
    and four rigid coordinates remain coupled. No physical pin, pressure
    smoothing, fitted radius or altered complementarity condition is added.
    This CPU implementation is a validation prototype, not a production path.
    """

    def __init__(self, compliance, interpolation, rigid):
        self.compliance = compliance
        self.interpolation = interpolation
        self.factor = cholesky(compliance, lower=True)
        self.point_factor = interpolation @ self.factor
        self.point_rigid = interpolation @ rigid
        self.rigid = rigid

    def solve(self, gap, spring, load, scale, initial_coordinates, initial_force=None):
        started = perf_counter()
        assert load > 0 and scale > 0 and (spring > 0).all()
        assert gap.shape == (len(self.compliance),)
        assert spring.shape == (len(self.interpolation),)
        target = np.array([0.0, 0.0, 0.0, 1.0])
        weight = spring * scale / load
        point_gap = self.interpolation @ gap / scale
        mapping = np.column_stack((self.point_factor * np.sqrt(load / scale), self.point_rigid))
        count = len(self.compliance)
        coordinates = np.zeros(count + 4)
        coordinates[count:] = initial_coordinates / scale
        if initial_force is not None:
            coordinates[:count] = self.factor.T @ initial_force / np.sqrt(load * scale)
        linear = np.r_[np.zeros(count), target]
        diagonal = np.r_[np.ones(count), np.zeros(4)]
        rounds = []
        for iteration in range(40):
            actual_gap = point_gap + mapping @ coordinates
            compression = np.maximum(-actual_gap, 0)
            point_force = weight * compression
            gradient = diagonal * coordinates + linear - mapping.T @ point_force
            gradient_error = np.linalg.norm(gradient)
            rounds.append({"iteration": iteration + 1, "gradient_norm": gradient_error.item(),
                           "active_points": int(np.count_nonzero(compression))})
            if gradient_error <= 1e-10:
                break
            active = actual_gap < 0
            weighted = mapping[active] * np.sqrt(weight[active, None])
            hessian = weighted.T @ weighted
            hessian[np.diag_indices_from(hessian)] += diagonal
            direction = cho_solve(cho_factor(hessian, lower=True), -gradient)
            slope = gradient @ direction
            assert slope < 0
            energy = 0.5 * np.sum(diagonal * coordinates**2) + linear @ coordinates
            energy += 0.5 * np.sum(weight * compression**2)
            step = 1.0
            for backtrack in range(40):
                trial = coordinates + step * direction
                trial_compression = np.maximum(-(point_gap + mapping @ trial), 0)
                trial_energy = 0.5 * np.sum(diagonal * trial**2) + linear @ trial
                trial_energy += 0.5 * np.sum(weight * trial_compression**2)
                if trial_energy <= energy + 1e-4 * step * slope + 16 * np.finfo(float).eps * abs(energy):
                    coordinates = trial
                    rounds[-1].update(step=step, backtracks=backtrack)
                    break
                step *= 0.5
            else:
                raise RuntimeError("Reduced pad energy line search did not converge")
        else:
            raise RuntimeError("Reduced pad energy force/stationarity equations did not converge")
        point_force *= load
        nodal_force = self.interpolation.T @ point_force
        pose = coordinates[count:] * scale
        elastic_gap = self.compliance @ nodal_force
        measured_gap = self.interpolation @ (gap + elastic_gap + self.rigid @ pose)
        law_force = spring * np.maximum(-measured_gap, 0)
        law_error = np.linalg.norm(point_force - law_force) / np.linalg.norm(point_force)
        wrench_error = np.linalg.norm(self.point_rigid.T @ point_force / load - target)
        stationarity = np.linalg.norm(
            coordinates[:count] - self.factor.T @ nodal_force / np.sqrt(load * scale)
        )
        assert max(wrench_error, stationarity) <= 1e-8
        assert law_error <= 1e-7
        return ReducedPadResult(
            point_force, nodal_force, pose, elastic_gap, rounds, perf_counter() - started,
            wrench_error.item(), stationarity.item(), law_error.item(),
        )
