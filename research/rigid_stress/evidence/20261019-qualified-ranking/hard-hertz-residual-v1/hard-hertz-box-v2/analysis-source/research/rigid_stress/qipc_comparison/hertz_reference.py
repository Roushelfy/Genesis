"""Independent CPU Hertz integrals and force-controlled half-space contact.

The rectangle BEM uses the Boussinesq surface Green function integrated over
constant-pressure square cells. Its contact support is an output of unilateral
minimization; no Hertz support, pressure or radius is prescribed to that solve.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from scipy.optimize import brentq


def hertz_integrals(ratio):
    def integrand(theta, power):
        c, s = np.cos(theta), np.sin(theta)
        return 2 * (c * c if power else 1) / (ratio * ratio * c * c + s * s) ** (0.5 if power == 0 else power)

    return np.array([quad(integrand, 0, np.pi / 2, args=(p,), epsabs=1e-12, epsrel=1e-12)[0] for p in (0.5, 1.5, 0)])


def ellipse(load, modulus, curvature):
    values, vectors = np.linalg.eigh(curvature)
    assert load > 0 and modulus > 0 and values.min() > 0
    ratio = values[0] / values[1]

    def equation(q):
        ix, iy, _ = hertz_integrals(q)
        return ix / iy - ratio

    q = 1.0 if abs(ratio - 1) < 1e-14 else brentq(equation, 0.001, 1, xtol=1e-14)
    ix, _, i0 = hertz_integrals(q)
    a = (3 * load * ix / (2 * np.pi * modulus * values[0])) ** (1 / 3)
    axes = np.array([a, a * q])
    peak = 3 * load / (2 * np.pi * axes.prod())
    depth = 3 * load * i0 / (4 * np.pi * modulus * a)
    return axes, vectors[:, 0], depth, peak


def ellipse_pressure(points, axes, direction, peak):
    tangent = np.array([-direction[1], direction[0]])
    radius2 = (points @ direction / axes[0]) ** 2 + (points @ tangent / axes[1]) ** 2
    return peak * np.sqrt(np.maximum(1 - radius2, 0))


def square_compliance(points, width, modulus):
    delta = points[:, None] - points[None, :]
    integral = np.zeros(delta.shape[:2])
    for sx, sy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        x, y = delta[..., 0] + sx * width / 2, delta[..., 1] + sy * width / 2
        primitive = x * np.arcsinh(y / np.abs(x)) + y * np.arcsinh(x / np.abs(y))
        integral += sx * sy * primitive
    matrix = integral / (np.pi * modulus * width * width)
    np.testing.assert_allclose(matrix, matrix.T, rtol=2e-12, atol=1e-20)
    return matrix


@dataclass(frozen=True)
class HardContactSolution:
    force: np.ndarray
    indentation: float
    gap: np.ndarray
    iterations: int


def solve_hard_contact(matrix, gap, load):
    """Primal active-set QP with force equality and explicit complete KKT checks."""
    scale = np.max(np.diag(matrix)) * load
    scaled_matrix, scaled_gap = matrix * load / scale, (gap - gap.min()) / scale
    force = np.zeros(len(gap))
    force[gap.argmin()] = 1
    active = force > 0
    for iteration in range(5 * len(gap)):
        current = scaled_matrix[np.ix_(active, active)]
        response = solve(current, np.column_stack((np.ones(active.sum()), scaled_gap[active])), assume_a="pos")
        depth = (1 + response[:, 1].sum()) / response[:, 0].sum()
        candidate = np.zeros_like(force)
        candidate[active] = depth * response[:, 0] - response[:, 1]
        negative = np.flatnonzero(active & (candidate < -1e-12))
        if len(negative):
            fractions = force[negative] / (force[negative] - candidate[negative])
            leaving = negative[fractions.argmin()]
            force += fractions.min() * (candidate - force)
            force[leaving], active[leaving] = 0, False
            continue
        force = candidate
        dual = scaled_matrix @ force + scaled_gap - depth
        inactive = np.flatnonzero(~active)
        if not len(inactive) or dual[inactive].min() >= -1e-10:
            assert force.min() >= -1e-12 and abs(force.sum() - 1) < 1e-10
            assert np.max(np.abs(force * dual)) < 1e-10 and dual.min() > -1e-10
            return HardContactSolution(force * load, depth * scale + gap.min(), dual * scale, iteration + 1)
        active[inactive[dual[inactive].argmin()]] = True
    raise RuntimeError("Hard-contact active set did not converge.")


def halfspace_stress(points, source, force, poisson=0.3):
    """Differentiate Boussinesq displacement by complex step, away from the surface.

    Unit shear modulus cancels in Hooke's law. Source forces are point resultants;
    callers must resolve the source spacing relative to the observation depth.
    """
    assert np.all(points[:, 2] > 0)
    result = np.empty((len(points), 3, 3))
    for start in range(0, len(points), 64):
        position = points[start : start + 64, None, :] - np.column_stack((source, np.zeros(len(source))))[None]
        gradient = np.empty((len(position), 3, 3))
        for axis in range(3):
            shifted = position.astype(complex)
            shifted[..., axis] += 1e-30j
            radius = np.sqrt(np.sum(shifted * shifted, axis=-1))
            z = shifted[..., 2]
            common = z / radius**3 - (1 - 2 * poisson) / (radius * (radius + z))
            displacement = np.stack(
                (shifted[..., 0] * common, shifted[..., 1] * common, z * z / radius**3 + 2 * (1 - poisson) / radius),
                axis=-1,
            ) / (4 * np.pi)
            gradient[..., axis] = np.einsum("ijk,j->ik", displacement.imag / 1e-30, force)
        strain = 0.5 * (gradient + gradient.swapaxes(-1, -2))
        result[start : start + len(position)] = 2 * strain + (
            2 * poisson / (1 - 2 * poisson) * np.trace(strain, axis1=-2, axis2=-1)
        )[:, None, None] * np.eye(3)
    return result
