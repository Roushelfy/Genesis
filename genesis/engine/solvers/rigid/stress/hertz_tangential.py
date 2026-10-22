"""Tangential half-space tensors for elliptic Hertz traction primitives."""

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import quadrants as qd
import genesis as gs
from genesis.utils.array_class import DataKind

from .hertz import (
    func_boussinesq_tensor,
    func_hertz_coordinate,
    func_hertz_integrals,
    func_hertz_normal_from_coordinate,
)


@dataclass(frozen=True)
class StressHertzShear:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    friction: qd.Tensor
    force: qd.Tensor
    coefficients: qd.Tensor


@qd.func
def func_hertz_tangential_from_coordinate(
    point: qd.types.vector(3),
    ratio: float,
    tangent: qd.types.vector(2),
    poisson: float,
    coordinate: float,
    common_integrals: qd.types.vector(3),
    quadrature: qd.Tensor,
):
    """Evaluate the tensor using a shared confocal coordinate and interior integrals."""
    bb = ratio * ratio
    squared = point * point
    coefficient = 1 + bb - squared[0] - squared[1]
    argument = qd.Vector([1 + coordinate, bb + coordinate, coordinate])
    q0 = qd.max(bb - bb * squared[0] - squared[1], 0.0)
    equation_derivative = gs.qd_float(0.0)
    if coordinate > 0:
        q0 = argument[0] * argument[1] * squared[2] / coordinate
        equation_derivative = (squared / (argument * argument)).sum()
    integrals = qd.Vector.zero(gs.qd_float, 3)
    pair = qd.Matrix.zero(gs.qd_float, 2, 2)
    regular = qd.Vector.zero(gs.qd_float, 4)
    for i_q in range(quadrature.shape[0]):
        rule = quadrature[i_q]
        (increment, weight) = ((bb + coordinate) * rule[0], (bb + coordinate) * rule[1])
        s = coordinate + increment
        dimensions = qd.Vector([1 + s, bb + s])
        ab = dimensions[0] * dimensions[1]
        delta = qd.sqrt(ab * s)
        root = qd.sqrt(q0 + increment * (2 * coordinate + coefficient + increment))
        derivative = 2 * s + coefficient
        for i_d, j_d in qd.static(qd.ndrange(2, 2)):
            pair[i_d, j_d] += (
                weight
                * (2 * poisson * s / delta + (1 - 2 * poisson) * point[2] / root)
                / (dimensions[i_d] * dimensions[j_d])
            )
        for i_d in qd.static(range(2)):
            integrals[i_d] += weight / (dimensions[i_d] * delta)
        for i_d in qd.static(range(4)):
            rational = ab / (dimensions[0] ** (3 - i_d) * dimensions[1] ** i_d)
            log_derivative = (i_d - 2) / dimensions[0] + (1 - i_d) / dimensions[1]
            regular[i_d] += weight * rational * (log_derivative / derivative - 2 / (derivative * derivative)) / root
    if point[2] > 0:
        integrals = common_integrals
    gradient = qd.Vector([-point[0] * integrals[0], -point[1] * integrals[1]])
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_d, j_d in qd.static(qd.ndrange(2, 2)):
        tensor[i_d, j_d] = tangent[i_d] * gradient[j_d] + tangent[j_d] * gradient[i_d]
        if qd.static(i_d == j_d):
            tensor[i_d, j_d] += 2 * poisson * gradient.dot(tangent)
        for k_d in qd.static(range(2)):
            (z_third, boundary) = (2 * point[2] * regular[i_d + j_d + k_d], gs.qd_float(0.0))
            if coordinate > 0:
                product = argument[i_d] * argument[j_d] * argument[k_d]
                ab = argument[0] * argument[1]
                z_third += 2 * qd.sqrt(ab * coordinate) / (product * (2 * coordinate + coefficient))
                boundary = 2 * qd.sqrt(coordinate / ab) / (product * equation_derivative)
            third = point[i_d] * point[j_d] * point[k_d] * ((1 - 2 * poisson) * z_third - boundary)
            if qd.static(i_d == j_d):
                third += point[k_d] * pair[i_d, k_d]
            if qd.static(i_d == k_d):
                third += point[j_d] * pair[i_d, j_d]
            if qd.static(j_d == k_d):
                third += point[i_d] * pair[i_d, j_d]
            tensor[i_d, j_d] += tangent[k_d] * third
    if point[2] > 0:
        denominator = qd.sqrt(argument[0] * argument[1] * argument[2]) * equation_derivative
        for i_d in qd.static(range(3)):
            value = gs.qd_float(0.0)
            if qd.static(i_d < 2):
                value = point[2] * tangent[i_d] * (integrals[i_d] - integrals[2])
            for j_d in qd.static(range(2)):
                value -= (
                    2
                    * point[2]
                    * point[i_d]
                    * point[j_d]
                    * tangent[j_d]
                    / (argument[i_d] * argument[j_d] * denominator)
                )
            (tensor[i_d, 2], tensor[2, i_d]) = (value, value)
    tensor *= ratio / 2
    if point[2] == 0:
        pressure = qd.sqrt(qd.max(1 - squared[0] - squared[1] / bb, 0.0))
        for i_d in qd.static(range(2)):
            (tensor[i_d, 2], tensor[2, i_d]) = (-pressure * tangent[i_d], -pressure * tangent[i_d])
    return tensor


@qd.func
def func_hertz_tangential_tensor(
    point: qd.types.vector(3), ratio: float, tangent: qd.types.vector(2), poisson: float, quadrature: qd.Tensor
):
    """Evaluate the pressure-normalized tangential Hertz tensor."""
    (coordinate, is_resolved) = func_hertz_coordinate(point, ratio)
    common_integrals = qd.Vector.zero(gs.qd_float, 3)
    if point[2] > 0:
        common_integrals = func_hertz_integrals(qd.Vector([1 + coordinate, ratio * ratio + coordinate, coordinate]))
    tensor = func_hertz_tangential_from_coordinate(
        point, ratio, tangent, poisson, coordinate, common_integrals, quadrature
    )
    return (tensor, is_resolved)


@qd.func
def func_cerruti_tensor(point: qd.types.vector(3), force: qd.types.vector(2), poisson: float):
    """Evaluate the positive-depth stress of a surface tangential point force.

    The analytical displacement divergence cancels the incompressible
    denominator before evaluation, keeping the stress finite as nu grows.
    """
    radius = point.norm()
    height = radius + point[2]
    contrast = 1 - 2 * poisson
    product = point[0] * force[0] + point[1] * force[1]
    outer = 1 / radius**3 - contrast / (radius * height**2)
    vertical = point[2] / radius**3 + contrast / (radius * height)
    gradient = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i_d in qd.static(range(3)):
        radius_derivative = point[i_d] / radius
        height_derivative, force_component, depth_derivative = radius_derivative, gs.qd_float(0.0), gs.qd_float(0.0)
        if qd.static(i_d < 2):
            force_component = force[i_d]
        else:
            height_derivative += 1
            depth_derivative = 1.0
        diagonal_derivative = -radius_derivative / radius**2 - contrast * height_derivative / height**2
        outer_derivative = -3 * radius_derivative / radius**4 + contrast * (
            radius_derivative / (radius**2 * height**2) + 2 * height_derivative / (radius * height**3)
        )
        vertical_derivative = depth_derivative / radius**3 - 3 * point[2] * radius_derivative / radius**4
        vertical_derivative -= contrast * (
            radius_derivative / (radius**2 * height) + height_derivative / (radius * height**2)
        )
        for j_d in qd.static(range(2)):
            value = diagonal_derivative * force[j_d] + outer_derivative * product * point[j_d]
            value += outer * force_component * point[j_d]
            if qd.static(i_d == j_d):
                value += outer * product
            gradient[j_d, i_d] = value
        gradient[2, i_d] = force_component * vertical + product * vertical_derivative
    tensor = (gradient + gradient.transpose()) / (4 * pi)
    for i_d in qd.static(range(3)):
        tensor[i_d, i_d] -= poisson * product / (pi * radius**3)
    return tensor


@qd.func
def func_hertz_friction_particular(
    i_p: int,
    point: qd.types.vector(3),
    axes: qd.types.vector(2),
    peak: float,
    force: float,
    poisson: float,
    quadrature: qd.Tensor,
    shear: StressHertzShear,
    far_field: qd.template(),
):
    """Combine normal pressure and the smooth nested traction history."""
    tensor, is_resolved = qd.Matrix.zero(gs.qd_float, 3, 3), True
    distance, ratio = point.norm(), axes[1] / axes[0]
    for i_s in range(16):
        radius = (i_s + 1) / 16
        tangent = shear.friction[i_p] * shear.coefficients[i_s, i_p]
        is_outer, has_tangent = i_s == 15, tangent.norm() > 0
        is_resolved = is_resolved and tangent.norm() < float("inf")
        if is_outer or has_tangent:
            semiaxes = radius * axes
            is_far = False
            if qd.static(far_field):
                is_far = distance >= 8 * semiaxes[0]
            if is_far:
                for i_m in range(4):
                    source = qd.Vector(
                        [(2 * (i_m // 2) - 1) * semiaxes[0], (2 * (i_m % 2) - 1) * semiaxes[1], 0.0]
                    ) / qd.sqrt(5.0)
                    if is_outer:
                        tensor += func_boussinesq_tensor(point - source, force / 4, poisson)
                    if has_tangent:
                        tensor += func_cerruti_tensor(point - source, force * radius**3 * tangent / 4, poisson)
            else:
                normalized = point / semiaxes[0]
                coordinate, is_root = func_hertz_coordinate(normalized, ratio)
                common_integrals = qd.Vector.zero(gs.qd_float, 3)
                if normalized[2] > 0:
                    common_integrals = func_hertz_integrals(
                        qd.Vector([1 + coordinate, ratio * ratio + coordinate, coordinate])
                    )
                if is_outer:
                    tensor += peak * func_hertz_normal_from_coordinate(
                        normalized, ratio, poisson, coordinate, common_integrals, quadrature
                    )
                if has_tangent:
                    tensor += (
                        radius
                        * peak
                        * func_hertz_tangential_from_coordinate(
                            normalized, ratio, tangent, poisson, coordinate, common_integrals, quadrature
                        )
                    )
                is_resolved = is_resolved and is_root
    return tensor, is_resolved
