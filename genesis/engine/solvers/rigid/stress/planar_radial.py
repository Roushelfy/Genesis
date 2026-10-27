"""Integrate Hertz pressure polynomials along a planar section ray.

These pure primitives provide an isolated integration candidate. The production
PlanarLoading route selects its existing quadrature independently.
"""

import math

import quadrants as qd

import genesis as gs

from .planar_moments import _func_planar_radial_interval

_SINC_DIFFERENCE_COEFFICIENTS = tuple((-1.0) ** (k + 1) / math.factorial(2 * k + 1) for k in range(1, 15))


@qd.func
def func_hertz_radial_moments(low: float, high: float):
    """Integrate rho, rho squared and rho cubed against sqrt(1-rho squared)."""
    result = qd.Vector.zero(gs.qd_float, 3)
    if 0.0 <= low < high <= 1.0:
        first = qd.sqrt((1.0 - low) * (1.0 + low))
        second = qd.sqrt((1.0 - high) * (1.0 + high))
        difference = (high - low) * (high + low)
        root_difference = difference / (first + second)
        result[0] = root_difference * (first * first + first * second + second * second) / 3.0
        result[2] = low * low * result[0]
        result[2] += (
            root_difference
            * root_difference
            * (2.0 * first**3 + 4.0 * first * first * second + 6.0 * first * second * second + 3.0 * second**3)
            / 15.0
        )
        sine_sum = low * second + high * first
        delta = qd.atan2(difference / sine_sum, low * high + first * second)
        argument = 2.0 * delta
        squared = argument * argument
        series = gs.qd_float(0.0)
        # On [0, pi], the alternating remainder is bounded by argument**30 / 31!.
        for i in qd.static(range(14)):
            series = _SINC_DIFFERENCE_COEFFICIENTS[13 - i] + squared * series
        sinc = gs.qd_float(1.0)
        if argument > 0.0:
            sinc = qd.sin(argument) / argument
        result[1] = delta * (squared * series + 2.0 * sine_sum * sine_sum * sinc) / 8.0
    return result


@qd.func
def func_planar_active_radial_interval(
    low: float,
    high: float,
    mean: qd.types.vector(3),
    slope: qd.types.vector(3),
    coefficient: qd.types.vector(3),
):
    """Restrict a ray interval to positive affine pressure without a length cutoff."""
    lower, upper = low, high
    intercept, derivative = mean.dot(coefficient), slope.dot(coefficient)
    first, second = intercept + low * derivative, intercept + high * derivative
    if high <= low or (first <= 0.0 and second <= 0.0):
        upper = lower
    elif first < 0.0:
        lower = qd.min(high, qd.max(low, low + (high - low) * (-first / (second - first))))
    elif second < 0.0:
        upper = qd.min(high, qd.max(low, low + (high - low) * (first / (first - second))))
    return lower, upper


@qd.func
def func_planar_ray_moments(
    low: float,
    high: float,
    scale: float,
    mean: qd.types.vector(3),
    slope: qd.types.vector(3),
    parent_mean: qd.types.vector(4),
    parent_slope: qd.types.vector(4),
):
    """Return the geometric Gram matrix and parent load moments of one ray."""
    moments = scale * func_hertz_radial_moments(low, high)
    gram = moments[0] * mean.outer_product(mean)
    gram += moments[1] * (mean.outer_product(slope) + slope.outer_product(mean))
    gram += moments[2] * slope.outer_product(slope)
    load = moments[0] * parent_mean.outer_product(mean)
    load += moments[1] * (parent_mean.outer_product(slope) + parent_slope.outer_product(mean))
    load += moments[2] * parent_slope.outer_product(slope)
    return gram, load


@qd.func
def func_planar_ray_count(n_angles: int, n_angular: int, is_contained: bool):
    """Return the number of angular rays in the existing planar quadrature rule."""
    count = qd.max(0, n_angles - 1) * n_angular
    if is_contained:
        count = qd.max(16, 2 * n_angular)
    return count


@qd.func
def func_planar_hertz_ray(
    i_ray: int,
    origin: qd.types.vector(3),
    dual: qd.types.matrix(2, 3),
    parent_shapes: qd.types.matrix(3, 4),
    section_center: qd.types.vector(3),
    plane: qd.types.matrix(2, 3),
    polygon: qd.types.matrix(3, 2),
    remaining: float,
    angles: qd.types.vector(11),
    n_angles: int,
    original_position: qd.types.vector(3),
    radius: float,
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    gauss_angular: qd.Tensor,
    weights_angular: qd.Tensor,
    is_contained: bool,
):
    """Return a fixed angular ray's physical scale and affine interpolation bases."""
    low, high, scale = gs.qd_float(0.0), gs.qd_float(0.0), gs.qd_float(0.0)
    mean, slope = qd.Vector([1.0, 0.0, 0.0]), qd.Vector.zero(gs.qd_float, 3)
    parent_mean, parent_slope = qd.Vector.zero(gs.qd_float, 4), qd.Vector.zero(gs.qd_float, 4)
    n_angular = gauss_angular.shape[0]
    if (
        i_ray >= 0
        and i_ray < func_planar_ray_count(n_angles, n_angular, is_contained)
        and remaining > 0.0
        and radius > 0.0
    ):
        section_radius = qd.sqrt(remaining)
        angle, angular_weight = gs.qd_float(0.0), gs.qd_float(0.0)
        if is_contained:
            n_circle = qd.max(16, 2 * n_angular)
            angle = 2.0 * qd.math.pi * (i_ray + 0.5) / n_circle
            angular_weight = 2.0 * qd.math.pi / n_circle
        else:
            i_interval, i_angular = i_ray // n_angular, i_ray % n_angular
            left, right = angles[i_interval], angles[i_interval + 1]
            angular_node = gauss_angular[i_angular]
            angle = left + (right - left) * qd.sin(0.5 * qd.math.pi * angular_node) ** 2
            angular_weight = (
                (right - left) * 0.5 * qd.math.pi * qd.sin(qd.math.pi * angular_node) * weights_angular[i_angular]
            )
        lower, upper = gs.qd_float(0.0), section_radius
        if not is_contained:
            lower, upper = _func_planar_radial_interval(polygon, angle, section_radius)
        if upper > lower and angular_weight > 0.0:
            low, high = qd.max(0.0, lower / section_radius), qd.min(1.0, upper / section_radius)
            direction = qd.Vector([qd.cos(angle), qd.sin(angle)]) @ plane
            offset = (section_center - original_position) / radius
            mean = qd.Vector([1.0, offset.dot(first), offset.dot(second)])
            slope = qd.Vector([0.0, direction.dot(first), direction.dot(second)]) * (section_radius / radius)
            local = dual @ (section_center - origin)
            local_slope = dual @ (section_radius * direction)
            barycentric = qd.Vector([1.0 - local.sum(), local[0], local[1]])
            barycentric_slope = qd.Vector([-local_slope.sum(), local_slope[0], local_slope[1]])
            parent_mean = parent_shapes.transpose() @ barycentric
            parent_slope = parent_shapes.transpose() @ barycentric_slope
            scale = remaining * section_radius / radius * angular_weight
    return low, high, scale, mean, slope, parent_mean, parent_slope
