"""Inexpensive native elliptic Hertz estimates for supplied physical regions.

Inputs are the total normal load, reduced modulus and relative curvature in a
common tangent frame. Region grouping, boundary provenance and friction remain
the caller's responsibility. This component does not infer them from collider
point count, penetration or solver stiffness.
"""

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, V, V_MAT, V_VEC


@dataclass(frozen=True)
class StressHertzInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    log_ratio: qd.Tensor
    integrals: qd.Tensor


@dataclass(frozen=True)
class StressHertzState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    force: qd.Tensor
    modulus: qd.Tensor
    curvature: qd.Tensor
    axes: qd.Tensor
    direction: qd.Tensor
    indentation: qd.Tensor
    peak_pressure: qd.Tensor
    status: qd.Tensor


class StressHertz:
    """Use one shared 257-row table and one bounded lookup per physical region.

    Status 0 admits the local Hertz calculation, 1 is unloaded, 2 is invalid
    input and 3 is unsupported curvature. Admission does not certify half-space
    assumptions, shell accuracy, boundary resolution or a tangential law.
    """

    def __init__(self, count: int):
        assert count > 0
        self.info = StressHertzInfo(
            log_ratio=V(dtype=gs.qd_float, shape=(257,)),
            integrals=V_VEC(2, dtype=gs.qd_float, shape=(257,)),
        )
        self.state = StressHertzState(
            force=V(dtype=gs.qd_float, shape=(count,)),
            modulus=V(dtype=gs.qd_float, shape=(count,)),
            curvature=V_MAT(2, 2, dtype=gs.qd_float, shape=(count,)),
            axes=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
            direction=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
            indentation=V(dtype=gs.qd_float, shape=(count,)),
            peak_pressure=V(dtype=gs.qd_float, shape=(count,)),
            status=V(dtype=gs.qd_int, shape=(count,)),
        )
        kernel_hertz_table(self.info)

    def update(self):
        kernel_hertz(self.state, self.info)


@qd.kernel
def kernel_hertz_table(hertz_info: StressHertzInfo):
    for i in range(hertz_info.log_ratio.shape[0]):
        # s = tan(theta)^2 removes the endpoint singularities of the Hertz
        # potential integrals. q=b/a is sampled logarithmically down to 0.05.
        q = qd.exp(qd.log(0.05) * (1 - i / 256.0))
        j_x, j_y, j_0 = 0.0, 0.0, 0.0
        for k in range(256):
            theta = (k + 0.5) * pi / 512
            c, s = qd.cos(theta), qd.sin(theta)
            d = qd.sqrt(q * q * c * c + s * s)
            j_x += c * c / d * pi / 256
            j_y += c * c / (d * d * d) * pi / 256
            j_0 += 1 / d * pi / 256
        hertz_info.log_ratio[i] = qd.log(j_x / j_y)
        hertz_info.integrals[i] = qd.Vector([j_x, j_0])


@qd.kernel
def kernel_hertz(hertz_state: StressHertzState, hertz_info: StressHertzInfo):
    for i in range(hertz_state.force.shape[0]):
        load, modulus = hertz_state.force[i], hertz_state.modulus[i]
        curvature = hertz_state.curvature[i]
        finite = load >= 0 and load < float("inf") and modulus > 0 and modulus < float("inf")
        for a, b in qd.static(qd.ndrange(2, 2)):
            finite = finite and qd.abs(curvature[a, b]) < float("inf")
        hertz_state.axes[i] = qd.Vector.zero(gs.qd_float, 2)
        hertz_state.direction[i] = qd.Vector([1.0, 0.0])
        hertz_state.indentation[i] = 0
        hertz_state.peak_pressure[i] = 0
        hertz_state.status[i] = 2
        if finite:
            if load > 0:
                off = 0.5 * (curvature[0, 1] + curvature[1, 0])
                mean = 0.5 * (curvature[0, 0] + curvature[1, 1])
                span = qd.sqrt((0.5 * (curvature[0, 0] - curvature[1, 1])) ** 2 + off * off)
                first, second = mean - span, mean + span
                hertz_state.status[i] = 3
                symmetric = qd.abs(curvature[0, 1] - curvature[1, 0]) <= 1e-12 * qd.max(second, 1.0)
                if first > 0 and symmetric:
                    ratio = qd.log(first / second)
                    if ratio >= hertz_info.log_ratio[0]:
                        low, high = 0, hertz_info.log_ratio.shape[0] - 1
                        for _ in range(8):
                            middle = (low + high) // 2
                            if hertz_info.log_ratio[middle] <= ratio:
                                low = middle
                            else:
                                high = middle
                        fraction = (ratio - hertz_info.log_ratio[low]) / (
                            hertz_info.log_ratio[high] - hertz_info.log_ratio[low]
                        )
                        q = qd.exp(qd.log(0.05) * (1 - (low + fraction) / 256.0))
                        integral = (1 - fraction) * hertz_info.integrals[low] + fraction * hertz_info.integrals[high]
                        major = (3 * load * integral[0] / (2 * pi * modulus * first)) ** (1 / 3)
                        hertz_state.axes[i] = qd.Vector([major, major * q])
                        angle = 0.5 * qd.atan2(-2 * off, curvature[1, 1] - curvature[0, 0])
                        hertz_state.direction[i] = qd.Vector([qd.cos(angle), qd.sin(angle)])
                        hertz_state.indentation[i] = 3 * load * integral[1] / (4 * pi * modulus * major)
                        hertz_state.peak_pressure[i] = 3 * load / (2 * pi * major * major * q)
                        hertz_state.status[i] = 0
            else:
                hertz_state.status[i] = 1


@qd.func
def func_hertz_pressure(
    offset: qd.types.vector(2), axes: qd.types.vector(2), direction: qd.types.vector(2), peak: float
):
    pressure = 0.0
    if axes[0] > 0 and axes[1] > 0:
        x = offset.dot(direction) / axes[0]
        y = (offset[1] * direction[0] - offset[0] * direction[1]) / axes[1]
        pressure = peak * qd.sqrt(qd.max(1 - x * x - y * y, 0.0))
    return pressure


@qd.kernel
def kernel_hertz_stress_quadrature(quadrature: qd.types.ndarray()):
    for i in range(quadrature.shape[0]):
        count = quadrature.shape[0]
        root = qd.cos(pi * (i + 0.75) / (count + 0.5))
        derivative = gs.qd_float(0.0)
        for _ in range(8):
            previous, current = gs.qd_float(1.0), root
            for j in range(2, count + 1):
                following = ((2 * j - 1) * root * current - (j - 1) * previous) / j
                previous, current = current, following
            derivative = count * (root * current - previous) / (root * root - 1)
            root -= current / derivative
        weight = pi / (2 * (1 - root * root) * derivative * derivative)
        theta = (root + 1) * pi / 4
        tangent = qd.sin(theta) / qd.cos(theta)
        quadrature[i] = qd.Vector([tangent * tangent, 2 * tangent * (1 + tangent * tangent) * weight])


@qd.func
def func_hertz_integrals(argument: qd.types.vector(3)):
    """Evaluate all three differentiated Newton integrals by shared duplication."""
    value, accumulated, factor = argument, qd.Vector.zero(gs.qd_float, 3), gs.qd_float(1.0)
    for _ in range(12):
        root = qd.sqrt(value)
        addition = root[0] * root[1] + root[1] * root[2] + root[2] * root[0]
        accumulated += factor / (root * (value + addition))
        value = (value + addition) / 4
        factor /= 4
    # Weighted means cancel the first-order remainder of Carlson RD.
    mean = (value.sum() + 2 * value) / 5
    return 2 * accumulated + 2 * factor / (3 * mean * qd.sqrt(mean))


@qd.func
def func_hertz_coordinate(point: qd.types.vector(3), ratio: float):
    bb = ratio * ratio
    squared = point * point
    coefficient = 1 + bb - squared[0] - squared[1]
    constant = bb - bb * squared[0] - squared[1]
    coordinate = gs.qd_float(0.0)
    if constant < 0:
        coordinate = -2 * constant / (coefficient + qd.sqrt(coefficient * coefficient - 4 * constant))
    is_resolved = True
    if point[2] > 0:
        low, high = qd.max(coordinate, squared[2]), squared.sum()
        surface_slack = 1 - squared[0] - squared[1] / bb
        if surface_slack > 0:
            # The in-plane terms decrease with the confocal coordinate, so
            # z²/slack bounds the root above, including near-surface probes.
            high = qd.min(high, squared[2] / surface_slack)
        coordinate = qd.max(low, high / 2)
        is_iterating = True
        for _ in range(48):
            if is_iterating:
                argument = qd.Vector([1 + coordinate, bb + coordinate, coordinate])
                equation = (squared / argument).sum() - 1
                if qd.abs(equation) < 1e-12:
                    is_iterating = False
                else:
                    if equation > 0:
                        low = coordinate
                    else:
                        high = coordinate
                    derivative = -(squared / (argument * argument)).sum()
                    proposal = coordinate - equation / derivative
                    coordinate = (low + high) / 2
                    if proposal > low and proposal < high:
                        coordinate = proposal
        argument = qd.Vector([1 + coordinate, bb + coordinate, coordinate])
        is_resolved = qd.abs((squared / argument).sum() - 1) < 1e-9
    return coordinate, is_resolved


@qd.func
def func_hertz_tensor(point: qd.types.vector(3), ratio: float, poisson: float, quadrature: qd.Tensor):
    bb = ratio * ratio
    squared = point * point
    coefficient = 1 + bb - squared[0] - squared[1]
    constant = bb - bb * squared[0] - squared[1]
    coordinate, is_resolved = func_hertz_coordinate(point, ratio)
    q0 = qd.max(constant, 0)
    if coordinate > 0:
        q0 = (1 + coordinate) * (bb + coordinate) * squared[2] / coordinate
    ell = qd.Vector.zero(gs.qd_float, 2)
    moment = qd.Matrix.zero(gs.qd_float, 2, 2)
    for i in range(quadrature.shape[0]):
        rule = quadrature[i]
        increment, weight = (bb + coordinate) * rule[0], (bb + coordinate) * rule[1]
        ax, by = 1 + coordinate + increment, bb + coordinate + increment
        root = qd.sqrt(q0 + increment * (2 * coordinate + coefficient + increment))
        ell += weight * root / (ax * by) * qd.Vector([1 / ax, 1 / by])
        moment[0, 0] += weight / (ax * ax * root)
        moment[1, 1] += weight / (by * by * root)
        moment[0, 1] += weight / (ax * by * root)
    moment[1, 0] = moment[0, 1]
    tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
    factor = ratio / 2
    for i, j in qd.static(qd.ndrange(2, 2)):
        tensor[i, j] = factor * (1 - 2 * poisson) * point[i] * point[j] * moment[i, j]
    for i in qd.static(range(2)):
        tensor[i, i] -= factor * (1 - 2 * poisson) * ell[i]
    if point[2] == 0:
        pressure = qd.sqrt(qd.max(1 - squared[0] - squared[1] / bb, 0))
        tensor[0, 0] -= 2 * poisson * pressure
        tensor[1, 1] -= 2 * poisson * pressure
        tensor[2, 2] = -pressure
    else:
        argument = qd.Vector([1 + coordinate, bb + coordinate, coordinate])
        integrals = func_hertz_integrals(argument)
        denominator = qd.sqrt(argument[0] * argument[1] * argument[2]) * (squared / (argument * argument)).sum()
        for i, j in qd.static(qd.ndrange(3, 3)):
            tensor[i, j] -= ratio * point[2] * point[i] * point[j] / (argument[i] * argument[j] * denominator)
        for i in qd.static(range(2)):
            tensor[i, i] += ratio * point[2] * ((1 - poisson) * integrals[i] - poisson * integrals[2])
    return tensor, is_resolved
