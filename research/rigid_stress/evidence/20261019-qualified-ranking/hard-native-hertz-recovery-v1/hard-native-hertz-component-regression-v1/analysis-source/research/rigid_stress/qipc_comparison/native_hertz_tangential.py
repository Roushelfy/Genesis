"""Native complete tensors for an explicitly supplied elliptic tangential traction."""

from dataclasses import dataclass

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, V_MAT, V_VEC

from .native_hertz import HertzState
from .native_hertz_stress import func_hertz_coordinate, func_hertz_integrals, kernel_hertz_stress_quadrature


@dataclass(frozen=True)
class HertzTangentialProbeState:
    position: qd.Tensor
    poisson: qd.Tensor
    tangent: qd.Tensor
    tensor: qd.Tensor
    status: qd.Tensor


class NativeHertzTangentialStress:
    """Evaluate q = tangent * normal Hertz pressure in the ellipse frame.

    The two-component tangent is a traction ratio, not a velocity or friction
    coefficient. No contact/friction history is inferred here. Status0 is a
    finite solution,1 unloaded,2 invalid inputs and3 a failed confocal root.
    """

    def __init__(self, count: int, probes: int, order: int = 16):
        assert count > 0 and probes > 0 and order >= 8
        self.quadrature = V_VEC(2, dtype=gs.qd_float, shape=(order,))
        self.state = HertzTangentialProbeState(
            position=V_VEC(3, dtype=gs.qd_float, shape=(probes, count)),
            poisson=V(dtype=gs.qd_float, shape=(count,)),
            tangent=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
            tensor=V_MAT(3, 3, dtype=gs.qd_float, shape=(probes, count)),
            status=V(dtype=gs.qd_int, shape=(probes, count)),
        )
        kernel_hertz_stress_quadrature(self.quadrature)

    def update(self, hertz: HertzState):
        kernel_hertz_tangential_stress(self.state, hertz, self.quadrature)


@qd.func
def func_hertz_tangential_tensor(
    point: qd.types.vector(3), ratio: float, tangent: qd.types.vector(2), poisson: float, quadrature: qd.template()
):
    coordinate, is_resolved = func_hertz_coordinate(point, ratio)
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
        increment, weight = (bb + coordinate) * rule[0], (bb + coordinate) * rule[1]
        s = coordinate + increment
        dimensions = qd.Vector([1 + s, bb + s])
        ab = dimensions[0] * dimensions[1]
        delta = qd.sqrt(ab * s)
        root = qd.sqrt(q0 + increment * (2 * coordinate + coefficient + increment))
        derivative = 2 * s + coefficient
        for i, j in qd.static(qd.ndrange(2, 2)):
            pair[i, j] += (
                weight
                * (2 * poisson * s / delta + (1 - 2 * poisson) * point[2] / root)
                / (dimensions[i] * dimensions[j])
            )
        for i in qd.static(range(2)):
            integrals[i] += weight / (dimensions[i] * delta)
        for i in qd.static(range(4)):
            rational = ab / (dimensions[0] ** (3 - i) * dimensions[1] ** i)
            log_derivative = (i - 2) / dimensions[0] + (1 - i) / dimensions[1]
            regular[i] += weight * rational * (log_derivative / derivative - 2 / (derivative * derivative)) / root
    if point[2] > 0:
        integrals = func_hertz_integrals(argument)
    gradient = qd.Vector([-point[0] * integrals[0], -point[1] * integrals[1]])
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(2, 2)):
        tensor[i, j] = tangent[i] * gradient[j] + tangent[j] * gradient[i]
        if qd.static(i == j):
            tensor[i, j] += 2 * poisson * gradient.dot(tangent)
        for k in qd.static(range(2)):
            z_third, boundary = 2 * point[2] * regular[i + j + k], gs.qd_float(0.0)
            if coordinate > 0:
                product = argument[i] * argument[j] * argument[k]
                ab = argument[0] * argument[1]
                z_third += 2 * qd.sqrt(ab * coordinate) / (product * (2 * coordinate + coefficient))
                boundary = 2 * qd.sqrt(coordinate / ab) / (product * equation_derivative)
            third = point[i] * point[j] * point[k] * ((1 - 2 * poisson) * z_third - boundary)
            if qd.static(i == j):
                third += point[k] * pair[i, k]
            if qd.static(i == k):
                third += point[j] * pair[i, j]
            if qd.static(j == k):
                third += point[i] * pair[i, j]
            tensor[i, j] += tangent[k] * third
    if point[2] > 0:
        denominator = qd.sqrt(argument[0] * argument[1] * argument[2]) * equation_derivative
        for i in qd.static(range(3)):
            value = gs.qd_float(0.0)
            if qd.static(i < 2):
                value = point[2] * tangent[i] * (integrals[i] - integrals[2])
            for j in qd.static(range(2)):
                value -= 2 * point[2] * point[i] * point[j] * tangent[j] / (argument[i] * argument[j] * denominator)
            tensor[i, 2], tensor[2, i] = value, value
    tensor *= ratio / 2
    if point[2] == 0:
        pressure = qd.sqrt(qd.max(1 - squared[0] - squared[1] / bb, 0.0))
        for i in qd.static(range(2)):
            tensor[i, 2], tensor[2, i] = -pressure * tangent[i], -pressure * tangent[i]
    return tensor, is_resolved


@qd.kernel
def kernel_hertz_tangential_stress(probe: HertzTangentialProbeState, hertz: HertzState, quadrature: qd.types.ndarray()):
    for i_q, i_p in qd.ndrange(probe.status.shape[0], probe.status.shape[1]):
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        status = 2
        if hertz.status[i_p] == 1:
            status = 1
        elif hertz.status[i_p] == 0:
            point, axes, poisson = probe.position[i_q, i_p], hertz.axes[i_p], probe.poisson[i_p]
            tangent, peak = probe.tangent[i_p], hertz.peak_pressure[i_p]
            is_valid = point.norm() < float("inf") and point[2] >= 0 and tangent.norm() < float("inf")
            is_valid = is_valid and axes.norm() < float("inf") and axes[0] > 0
            is_valid = is_valid and axes[1] >= 0.05 * axes[0] and axes[1] <= axes[0]
            is_valid = is_valid and poisson > -1 and poisson < 0.5 and peak >= 0 and peak < float("inf")
            if is_valid:
                candidate, is_resolved = func_hertz_tangential_tensor(
                    point / axes[0], axes[1] / axes[0], tangent, poisson, quadrature
                )
                status = 3
                if is_resolved and candidate.norm() < float("inf"):
                    tensor, status = candidate * peak, 0
        probe.tensor[i_q, i_p], probe.status[i_q, i_p] = tensor, status
