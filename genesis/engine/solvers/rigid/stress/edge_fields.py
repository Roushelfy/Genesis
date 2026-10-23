"""Slender sharp-edge strip stress with a finite parabolic centroid-line correction."""

from math import pi

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC

from .edge import func_edge_pressure
from .hertz import func_boussinesq_tensor, func_legendre_root
from .hertz_tangential import func_cerruti_tensor


class StressEdgeFields:
    """Provide native quadrature for local prescribed sharp-edge pressure fields.

    The finite-length correction uses a minimum depth of 0.001 half-lengths;
    the strip retains its actual depth and surface traction. Source binding,
    friction history and the finite material correction remain with the caller.
    """

    def __init__(self, order: int = 32):
        assert order > 1
        self.quadrature = V_VEC(2, dtype=gs.qd_float, shape=(order,))
        kernel_edge_field_quadrature(self.quadrature)


@qd.func
def func_edge_probe_offset(index: int, width: float, half_length: float, exponent: float):
    """Locate 89 finite-depth probes relative to the pressure centroid.

    The first 84 samples span seven transverse positions, three along-edge
    positions and depths of 0.01, 0.05, 0.2 and 1 widths. Five central samples
    extend to 2, 4, 8, 16 and 32 widths. These define sampled peaks.
    """
    mean = 2 * (1 - exponent) * width / 5
    local = qd.Vector.zero(gs.qd_float, 3)
    if index < 84:
        transverse = qd.Vector([-0.5, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5])
        along = qd.Vector([-0.75, 0.0, 0.75])
        depth = qd.Vector([0.01, 0.05, 0.2, 1.0])
        local = qd.Vector(
            [transverse[index // 12] * width - mean, along[(index // 4) % 3] * half_length, depth[index % 4] * width]
        )
    else:
        local[2] = width * 2 ** (index - 83)
    return local


@qd.kernel
def kernel_edge_field_quadrature(quadrature: qd.Tensor):
    for i_q in range(quadrature.shape[0]):
        root, derivative = func_legendre_root(i_q, quadrature.shape[0])
        quadrature[i_q] = qd.Vector([root, 2 / ((1 - root * root) * derivative * derivative)])


@qd.func
def func_edge_transform(position: qd.types.vector(2), width: float, exponent: float, amplitude: float):
    """Return the strip Cauchy transform and its depth-scaled derivative."""
    square = position.dot(position)
    inverse = qd.Vector([position[0], -position[1]]) / square
    ratio = width * inverse
    argument = (
        qd.Vector([(position[0] - width) * position[0] + position[1] * position[1], width * position[1]]) / square
    )
    radius, angle = argument.norm(), qd.atan2(argument[1], argument[0])
    if not position[1] > 0:
        angle = 0.0
        if argument[0] < 0:
            angle = pi
    coefficient = pi * amplitude / qd.sin(pi * exponent)
    transform = qd.Vector.zero(gs.qd_float, 2)
    derivative = qd.Vector.zero(gs.qd_float, 2)
    if ratio.norm() < 0.1:
        # Avoid subtracting nearly equal powers in the remote strip tail.
        term = exponent * ratio
        for i_k in range(1, 9):
            transform += term
            term = (
                (i_k - exponent)
                / (i_k + 1)
                * qd.Vector([term[0] * ratio[0] - term[1] * ratio[1], term[0] * ratio[1] + term[1] * ratio[0]])
            )
        transform *= coefficient
    else:
        power = radius**exponent
        transform = coefficient * qd.Vector([1 - power * qd.cos(exponent * angle), -power * qd.sin(exponent * angle)])
    if position[1] > 0:
        power = radius ** (exponent - 1)
        first = power * qd.Vector([qd.cos((exponent - 1) * angle), qd.sin((exponent - 1) * angle)])
        second = qd.Vector([inverse[0] ** 2 - inverse[1] ** 2, 2 * inverse[0] * inverse[1]])
        derivative = (
            -coefficient
            * exponent
            * width
            * position[1]
            * qd.Vector([first[0] * second[0] - first[1] * second[1], first[0] * second[1] + first[1] * second[0]])
        )
    return transform, derivative


@qd.func
def func_edge_particular(
    point: qd.types.vector(3),
    width: float,
    half_length: float,
    exponent: float,
    amplitude: float,
    force: float,
    tangent: qd.types.vector(2),
    poisson: float,
    quadrature: qd.Tensor,
):
    """Return a finite tensor and status: 0 finite, 2 invalid, 3 unbounded sharp edge.

    Beyond eight maximum patch dimensions, four quadrature sources match the
    pressure moments through second order for remote field evaluation.
    """
    tensor, status = qd.Matrix.zero(gs.qd_float, 3, 3), 2
    is_valid = point.norm() < float("inf") and point[2] >= 0 and tangent.norm() < float("inf")
    is_valid = is_valid and width > 0 and half_length > 0 and width < float("inf") and half_length < float("inf")
    is_valid = is_valid and exponent > 0 and exponent < 1 and amplitude > 0 and amplitude < float("inf")
    is_valid = is_valid and force > 0 and force < float("inf") and poisson > -1 and poisson < 0.5
    if is_valid:
        profile = qd.max(1 - (point[1] / half_length) ** 2, 0.0)
        extent = width * profile
        is_singular = profile > 0 and not (point[0] * point[0] + point[2] * point[2] > 0)
        status = 3
        if not is_singular:
            status = 0
            is_far = point.norm() >= 8 * qd.max(width, half_length)
            depth = point[2]
            if not is_far:
                transform, derivative = qd.Vector.zero(gs.qd_float, 2), qd.Vector.zero(gs.qd_float, 2)
                if profile > 0:
                    transform, derivative = func_edge_transform(
                        qd.Vector([point[0], point[2]]), extent, exponent, amplitude
                    )
                depth = qd.max(point[2], 0.001 * half_length)
                transverse = point[0] - extent * (1 - exponent) / 2
                square = transverse * transverse + depth * depth
                density = 3 * force * profile / (4 * half_length)
                transform -= density * qd.Vector([transverse, -depth]) / square
                derivative += (
                    density
                    * depth
                    * qd.Vector([transverse * transverse - depth * depth, -2 * transverse * depth])
                    / square**2
                )
                tensor[0, 0] = transform[1] + derivative[0] + tangent[0] * (-2 * transform[0] + derivative[1])
                tensor[2, 2] = transform[1] - derivative[0] - tangent[0] * derivative[1]
                tensor[0, 2] = -derivative[1] + tangent[0] * (transform[1] + derivative[0])
                tensor[2, 0] = tensor[0, 2]
                tensor[1, 1] = poisson * (tensor[0, 0] + tensor[2, 2])
                tensor[0, 1], tensor[1, 0] = -tangent[1] * transform[0], -tangent[1] * transform[0]
                tensor[1, 2], tensor[2, 1] = tangent[1] * transform[1], tangent[1] * transform[1]
                tensor /= pi
            limits = qd.Matrix.zero(gs.qd_float, 2, 2)
            scale = gs.qd_float(1.0)
            n_samples = 4
            if not is_far:
                distance = qd.max(qd.max(-point[0], point[0] - width), 0.0)
                scale = qd.sqrt(distance * distance + depth * depth)
                split = qd.min(qd.max(point[1], -half_length), half_length)
                for i_s in range(2):
                    bounds = qd.Vector([-half_length, split])
                    if i_s > 0:
                        bounds = qd.Vector([split, half_length])
                    for i_d in qd.static(range(2)):
                        value = (bounds[i_d] - point[1]) / scale
                        limits[i_s, i_d] = qd.log(qd.abs(value) + qd.sqrt(value * value + 1))
                        if value < 0:
                            limits[i_s, i_d] = -limits[i_s, i_d]
                n_samples = 2 * quadrature.shape[0]
            for i_q in range(n_samples):
                source = qd.Vector.zero(gs.qd_float, 3)
                load = force / 4
                is_sample = True
                if is_far:
                    center = 2 * (1 - exponent) * width / 5
                    spread = width * qd.sqrt(4 * (1 - exponent) * (3 + 2 * exponent) / 175)
                    source[0] = center + (2 * (i_q // 2) - 1) * spread
                    source[1] = (2 * (i_q % 2) - 1) * half_length / qd.sqrt(5.0)
                else:
                    i_s, i_r = i_q // quadrature.shape[0], i_q % quadrature.shape[0]
                    lower, upper = limits[i_s, 0], limits[i_s, 1]
                    is_sample = upper > lower
                    if is_sample:
                        rule = quadrature[i_r]
                        coordinate = (lower + upper + (upper - lower) * rule[0]) / 2
                        exponential = qd.exp(coordinate)
                        along = point[1] + scale * (exponential - 1 / exponential) / 2
                        measure = rule[1] * (upper - lower) * scale * (exponential + 1 / exponential) / 4
                        shape = qd.max(1 - (along / half_length) ** 2, 0.0)
                        source = qd.Vector([width * (1 - exponent) * shape / 2, along, 0.0])
                        load = 3 * force * shape * measure / (4 * half_length)
                if is_sample:
                    position = qd.Vector([point[0], point[1], depth]) - source
                    tensor += func_boussinesq_tensor(position, load, poisson)
                    tensor += func_cerruti_tensor(position, load * tangent, poisson)
            if not point[2] > 0:
                pressure = func_edge_pressure(qd.Vector([point[0], point[1]]), width, half_length, exponent, amplitude)
                for i_d in qd.static(range(3)):
                    traction = -pressure
                    if qd.static(i_d < 2):
                        traction *= tangent[i_d]
                    tensor[i_d, 2], tensor[2, i_d] = traction, traction
            if not tensor.norm() < float("inf"):
                tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
                status = 2
    return tensor, status
