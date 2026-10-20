"""Native normal Hertz tensors at supplied principal-frame material probes."""

from dataclasses import dataclass
from math import pi

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, V_MAT, V_VEC

from .native_hertz import HertzState


@dataclass(frozen=True)
class HertzProbeState:
    position: qd.Tensor
    poisson: qd.Tensor
    tensor: qd.Tensor
    status: qd.Tensor


class NativeHertzStress:
    """Evaluate the finite half-space field, including its surface limit.

    Probe coordinates use the ellipse principal frame with depth into material.
    Status 0 is a finite normal solution, 1 is unloaded, 2 rejects inputs, and 3
    rejects a failed confocal root. Shell and tangential fields require their
    own models; this component supplies a normal near-field tensor.
    """

    def __init__(self, count: int, probes: int, order: int = 16):
        assert count > 0 and probes > 0 and order >= 8
        self.quadrature = V_VEC(2, dtype=gs.qd_float, shape=(order,))
        self.state = HertzProbeState(
            position=V_VEC(3, dtype=gs.qd_float, shape=(probes, count)),
            poisson=V(dtype=gs.qd_float, shape=(count,)),
            tensor=V_MAT(3, 3, dtype=gs.qd_float, shape=(probes, count)),
            status=V(dtype=gs.qd_int, shape=(probes, count)),
        )
        kernel_hertz_stress_quadrature(self.quadrature)

    def update(self, hertz: HertzState):
        kernel_hertz_stress(self.state, hertz, self.quadrature)


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
def func_hertz_tensor(point: qd.types.vector(3), ratio: float, poisson: float, quadrature: qd.template()):
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


@qd.kernel
def kernel_hertz_stress(probe: HertzProbeState, hertz: HertzState, quadrature: qd.types.ndarray()):
    for i_q, i_p in qd.ndrange(probe.status.shape[0], probe.status.shape[1]):
        tensor = qd.Matrix.zero(gs.qd_float, 3, 3)
        status = 2
        if hertz.status[i_p] == 1:
            status = 1
        elif hertz.status[i_p] == 0:
            point, axes, poisson = probe.position[i_q, i_p], hertz.axes[i_p], probe.poisson[i_p]
            is_valid = point.norm() < float("inf") and point[2] >= 0
            is_valid = is_valid and axes[0] > 0 and axes[1] >= 0.05 * axes[0] and axes[1] <= axes[0]
            is_valid = is_valid and poisson > -1 and poisson < 0.5
            if is_valid:
                candidate, is_resolved = func_hertz_tensor(point / axes[0], axes[1] / axes[0], poisson, quadrature)
                status = 3
                if is_resolved and candidate.norm() < float("inf"):
                    tensor, status = candidate * hertz.peak_pressure[i_p], 0
        probe.tensor[i_q, i_p], probe.status[i_q, i_p] = tensor, status
