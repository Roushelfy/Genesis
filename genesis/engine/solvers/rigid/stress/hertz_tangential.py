"""Tangential half-space tensors for elliptic Hertz traction primitives."""

import quadrants as qd
import genesis as gs
from .hertz import func_hertz_coordinate, func_hertz_integrals


@qd.func
def func_hertz_tangential_tensor(
    point: qd.types.vector(3), ratio: float, tangent: qd.types.vector(2), poisson: float, quadrature: qd.Tensor
):
    """Evaluate the pressure-normalized tangential Hertz tensor.

    Coordinates use the major semiaxis as length unit. The two-component
    traction ratio multiplies normal pressure in the ellipse frame."""
    (coordinate, is_resolved) = func_hertz_coordinate(point, ratio)
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
        integrals = func_hertz_integrals(argument)
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
    return (tensor, is_resolved)
