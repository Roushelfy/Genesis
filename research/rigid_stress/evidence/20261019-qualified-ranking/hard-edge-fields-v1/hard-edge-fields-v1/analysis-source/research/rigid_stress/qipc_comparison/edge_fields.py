"""Continuous sharp-edge stress references and a slender-strip composite field."""

import numpy as np
from scipy.special import roots_jacobi

from .hertz_reference import halfspace_stress
from .hertz_tangential import cerruti_stress


def strip_tensor(point, width, exponent, amplitude, tangent, poisson=0.3):
    """Integrate the infinite strip by its complex Cauchy transform.

    H(zeta)=pi*C/sin(pi*lambda)*(1-(1-width/zeta)**lambda).
    Integrating the three-dimensional Green tensors along the strip gives
    Flamant plane strain and the antiplane shear field below.
    """
    coordinate = point[..., 0] + 1j * point[..., 2]
    argument = 1 - width / coordinate
    # Use the upper-half-plane boundary value on the loaded interval.
    argument = argument.real + 1j * np.maximum(argument.imag, 0)
    coefficient = np.pi * amplitude / np.sin(np.pi * exponent)
    transform = coefficient * (1 - argument**exponent)
    derivative = -coefficient * exponent * width * argument ** (exponent - 1) / coordinate**2
    return flamant_tensor(transform, derivative, point[..., 2], tangent, poisson)


def flamant_tensor(transform, derivative, depth, tangent, poisson=0.3):
    """Recover normal, transverse and longitudinal strip stress components."""
    real, imag = transform.real, transform.imag
    dr, di = depth * derivative.real, depth * derivative.imag
    result = np.zeros((*np.shape(transform), 3, 3))
    result[..., 0, 0] = imag + dr + tangent[0] * (-2 * real + di)
    result[..., 2, 2] = imag - dr - tangent[0] * di
    result[..., 0, 2] = result[..., 2, 0] = -di + tangent[0] * (imag + dr)
    result[..., 1, 1] = poisson * (result[..., 0, 0] + result[..., 2, 2])
    result[..., 0, 1] = result[..., 1, 0] = -tangent[1] * real
    result[..., 1, 2] = result[..., 2, 1] = tangent[1] * imag
    return result / np.pi


def stretched_line_rule(point, width, along, order):
    """Resolve the Green kernel in y with a hyperbolic-sine change of variable."""
    knots, weights = np.polynomial.legendre.leggauss(order)
    outside = max(-point[0], point[0] - width, 0)
    scale = np.hypot(outside, point[2])
    assert scale > 0
    limits = [-along, along]
    if -along < point[1] < along:
        limits.insert(1, point[1])
    values, measures = [], []
    for lower, upper in zip(limits[:-1], limits[1:]):
        a, b = np.arcsinh((np.array([lower, upper]) - point[1]) / scale)
        t = (a + b) / 2 + (b - a) * knots / 2
        values.append(point[1] + scale * np.sinh(t))
        measures.append(weights * (b - a) / 2 * scale * np.cosh(t))
    return np.concatenate(values), np.concatenate(measures)


def continuous_tensor(points, model, load, tangent, order_x=64, order_y=48, poisson=0.3):
    """Independent full two-dimensional pressure integration at positive depths."""
    knots, weights = roots_jacobi(order_x, model.exponent, -model.exponent)
    u, weights = (knots + 1) / 2, weights / weights.sum()
    tensors = []
    for point in points:
        y, measure = stretched_line_rule(point, model.width, model.along, order_y)
        profile = np.maximum(1 - (y / model.along) ** 2, 0)
        x = model.width * profile[:, None] * u
        source = np.column_stack((x.reshape(-1), np.repeat(y, order_x)))
        force = (3 * load / (4 * model.along) * profile * measure)[:, None] * weights
        normal = halfspace_stress(point[None], source, force.reshape(-1), poisson)
        shear = cerruti_stress(point[None], source, force.reshape(-1, 1) * tangent, poisson)
        tensors.append((normal + shear)[0])
    return np.array(tensors)


def composite_tensor(points, model, load, tangent, order=32, depth_floor=0.001, poisson=0.3):
    """Combine a local exact strip with the finite parabolic centroid-line field.

    The finite line uses the strip's exact force and transverse first moment
    at each y. Subtract its infinite local line before adding the exact strip.
    Evaluate the regular line correction at z>=depth_floor*b to avoid a
    spurious line singularity in a difference that is regular away from ends.
    This floor changes only the finite-length correction, not strip pressure.
    """
    tensors = []
    for point in points:
        profile = max(1 - (point[1] / model.along) ** 2, 0)
        width = model.width * profile
        strip = np.zeros((3, 3))
        if profile > 0:
            strip = strip_tensor(point, width, model.exponent, model.amplitude, tangent, poisson)
        regular = point.copy()
        regular[2] = max(regular[2], depth_floor * model.along)
        y, measure = stretched_line_rule(regular, model.width, model.along, order)
        density = 3 * load / (4 * model.along) * np.maximum(1 - (y / model.along) ** 2, 0)
        center = model.width * (1 - model.exponent) / 2 * np.maximum(1 - (y / model.along) ** 2, 0)
        source, force = np.column_stack((center, y)), density * measure
        finite = halfspace_stress(regular[None], source, force, poisson)[0]
        finite += cerruti_stress(regular[None], source, force[:, None] * tangent, poisson)[0]
        coordinate = regular[0] - width * (1 - model.exponent) / 2 + 1j * regular[2]
        local_density = 3 * load / (4 * model.along) * profile
        infinite = flamant_tensor(
            local_density / coordinate, -local_density / coordinate**2, regular[2], tangent, poisson
        )
        tensors.append(strip + finite - infinite)
    return np.array(tensors)
