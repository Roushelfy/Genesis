"""Elliptic sqrt-traction stress and an independent Cerruti point-force check.

Depth z is positive into the half-space. The imposed surface traction is
q = peak * tangent * sqrt(1-x^2/a^2-y^2/b^2), so sigma_iz = -q_i.
This solves a declared traction problem, not the frictional contact law.

The harmonic potentials Phi=int(p/R), Psi_z=Phi and K_z=-Psi give the
Hamilton--Goodman tangential stress relations (Willert, arXiv:2310.03508,
Eq.12). Integrating the elliptical Newton potential in z reduces all terms
to one-dimensional integrals. Integration by parts removes the Q^-3/2
endpoint singularity before quadrature, preserving the exact surface limit.
"""

import numpy as np
from scipy.special import elliprd

from .hertz_stress import confocal_coordinate


def elliptic_tangential_stress(points, axes, tangent, peak=1.0, poisson=0.3, order=48):
    assert np.all(points[:, 2] >= 0) and np.all(axes > 0) and -1 < poisson < 0.5
    scale = axes.max()
    position, semiaxes = points / scale, axes / scale
    aa, bb = semiaxes**2
    x, y, z = position.T
    coordinate = confocal_coordinate(position, semiaxes)
    argument = coordinate[:, None] + np.array([aa, bb, 0])
    interior = z > 0
    gauss, weights = np.polynomial.legendre.leggauss(order)
    theta, weights = (gauss + 1) * np.pi / 4, weights * np.pi / 4
    tan = np.tan(theta)
    increment = (bb + coordinate[:, None]) * tan**2
    s = coordinate[:, None] + increment
    measure = 2 * (bb + coordinate[:, None]) * tan * (1 + tan**2) * weights
    dimensions = np.stack((aa + s, bb + s), axis=-1)
    ab = dimensions.prod(axis=-1)
    delta = np.sqrt(ab * s)
    coefficient = aa + bb - x**2 - y**2
    q0 = np.maximum(aa * bb - bb * x**2 - aa * y**2, 0)
    q0[interior] = argument[interior, 0] * argument[interior, 1] * z[interior] ** 2 / coordinate[interior]
    root = np.sqrt(q0[:, None] + increment * (2 * coordinate[:, None] + coefficient[:, None] + increment))
    derivative = 2 * s + coefficient[:, None]
    integrals = np.zeros_like(position)
    for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        integrals[interior, i] = 2 * elliprd(argument[interior, j], argument[interior, k], argument[interior, i]) / 3
    for i in (0, 1):
        integrals[~interior, i] = (measure / (dimensions[..., i] * delta))[~interior].sum(axis=1)
    pair = np.empty((len(points), 2, 2))
    for i in (0, 1):
        for j in (0, 1):
            denominator = dimensions[..., i] * dimensions[..., j]
            pair[:, i, j] = (
                measure * (2 * poisson * s / delta + (1 - 2 * poisson) * z[:, None] / root) / denominator
            ).sum(axis=1)
    positive = coordinate > 0
    planar_product = argument[:, 0] * argument[:, 1]
    equation_derivative = np.zeros(len(points))
    equation_derivative[positive] = (position[positive] ** 2 / argument[positive] ** 2).sum(axis=1)
    tensor = np.zeros((len(points), 3, 3))
    gradient = -position[:, :2] * integrals[:, :2]
    for i in (0, 1):
        for j in (0, 1):
            tensor[:, i, j] = tangent[i] * gradient[:, j] + tangent[j] * gradient[:, i]
            if i == j:
                tensor[:, i, j] += 2 * poisson * (gradient @ tangent)
            for k in (0, 1):
                rational = ab / (dimensions[..., i] * dimensions[..., j] * dimensions[..., k])
                log_derivative = (
                    1 / dimensions[..., 0]
                    + 1 / dimensions[..., 1]
                    - 1 / dimensions[..., i]
                    - 1 / dimensions[..., j]
                    - 1 / dimensions[..., k]
                )
                regular = (measure * rational * (log_derivative / derivative - 2 / derivative**2) / root).sum(axis=1)
                lower_product = argument[:, i] * argument[:, j] * argument[:, k]
                z_third = 2 * z * regular
                boundary = np.zeros(len(points))
                z_third[positive] += (
                    2
                    * np.sqrt(planar_product[positive] * coordinate[positive])
                    / (lower_product[positive] * (2 * coordinate[positive] + coefficient[positive]))
                )
                boundary[positive] = (
                    2
                    * np.sqrt(coordinate[positive] / planar_product[positive])
                    / (lower_product[positive] * equation_derivative[positive])
                )
                third = position[:, i] * position[:, j] * position[:, k] * ((1 - 2 * poisson) * z_third - boundary)
                if i == j:
                    third += position[:, k] * pair[:, i, k]
                if i == k:
                    third += position[:, j] * pair[:, i, j]
                if j == k:
                    third += position[:, i] * pair[:, i, j]
                tensor[:, i, j] += tangent[k] * third
    if np.any(interior):
        p, arg = position[interior], argument[interior]
        denominator = np.sqrt(arg.prod(axis=1)) * equation_derivative[interior]
        hessian = 2 * p[:, :, None] * p[:, None, :] / (arg[:, :, None] * arg[:, None, :] * denominator[:, None, None])
        for i in (0, 1, 2):
            hessian[:, i, i] -= integrals[interior, i]
        shear = -z[interior, None] * (hessian[:, :2, :2] @ tangent + integrals[interior, 2, None] * tangent)
        tensor[interior, :2, 2] = tensor[interior, 2, :2] = shear
        tensor[interior, 2, 2] = -z[interior] * (hessian[:, 2, :2] @ tangent)
    tensor *= semiaxes.prod() * peak / 2
    pressure = peak * np.sqrt(np.maximum(1 - x**2 / aa - y**2 / bb, 0))
    tensor[~interior, :2, 2] = tensor[~interior, 2, :2] = -pressure[~interior, None] * tangent
    return tensor


def cerruti_stress(points, source, force, poisson=0.3):
    """Independent positive-depth point-force integration via displacement gradients.

    force has two tangential components per source; unit shear modulus cancels.
    The Cerruti displacement is differentiated by complex step, not by the
    integrated-potential formulas used by elliptic_tangential_stress.
    """
    assert np.all(points[:, 2] > 0)
    result = np.empty((len(points), 3, 3))
    for start in range(0, len(points), 16):
        position = points[start : start + 16, None] - np.column_stack((source, np.zeros(len(source))))[None]
        gradient = np.empty((len(position), 3, 3))
        for axis in range(3):
            shifted = position.astype(complex)
            shifted[..., axis] += 1e-30j
            radius = np.sqrt(np.sum(shifted**2, axis=-1))
            z = shifted[..., 2]
            dot = (shifted[..., :2] * force[None]).sum(axis=-1)
            diagonal = 1 / radius + (1 - 2 * poisson) / (radius + z)
            outer = 1 / radius**3 - (1 - 2 * poisson) / (radius * (radius + z) ** 2)
            displacement = np.empty_like(shifted)
            displacement[..., :2] = diagonal[..., None] * force + outer[..., None] * dot[..., None] * shifted[..., :2]
            displacement[..., 2] = dot * (z / radius**3 + (1 - 2 * poisson) / (radius * (radius + z)))
            gradient[..., axis] = displacement.imag.sum(axis=1) / (4 * np.pi * 1e-30)
        strain = 0.5 * (gradient + gradient.swapaxes(-1, -2))
        result[start : start + len(position)] = 2 * strain + (
            2 * poisson / (1 - 2 * poisson) * np.trace(strain, axis1=-2, axis2=-1)
        )[:, None, None] * np.eye(3)
    return result
