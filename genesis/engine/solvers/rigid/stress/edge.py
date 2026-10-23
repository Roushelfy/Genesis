"""Slender sharp-edge pressure estimates for supplied local contact geometry."""

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, V


@dataclass(frozen=True)
class StressEdgeState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    force: qd.Tensor
    modulus: qd.Tensor
    slope: qd.Tensor
    curvature: qd.Tensor
    shear_ratio: qd.Tensor
    poisson: qd.Tensor
    width: qd.Tensor
    half_length: qd.Tensor
    exponent: qd.Tensor
    amplitude: qd.Tensor
    status: qd.Tensor


class StressEdge:
    """Estimate a narrow footprint from an edge slope and along-edge curvature.

    A strip solution includes proportional transverse shear. A logarithmic
    surface-compatibility equation determines the along-edge extent, retaining
    footprints with width/half_length < 0.05. Supplied geometry and force ratios
    describe the local approximation; source binding and friction history belong
    to the caller. Status0 succeeds,1 is unloaded,2 is invalid input and3 is
    outside the slender regime;4 reports an unconverged scalar solve. Amplitude
    multiplies an unbounded edge profile.
    """

    def __init__(self, count: int):
        assert count >= 0
        self.state = StressEdgeState(
            force=V(dtype=gs.qd_float, shape=(count,)),
            modulus=V(dtype=gs.qd_float, shape=(count,)),
            slope=V(dtype=gs.qd_float, shape=(count,)),
            curvature=V(dtype=gs.qd_float, shape=(count,)),
            shear_ratio=V(dtype=gs.qd_float, shape=(count,)),
            poisson=V(dtype=gs.qd_float, shape=(count,)),
            width=V(dtype=gs.qd_float, shape=(count,)),
            half_length=V(dtype=gs.qd_float, shape=(count,)),
            exponent=V(dtype=gs.qd_float, shape=(count,)),
            amplitude=V(dtype=gs.qd_float, shape=(count,)),
            status=V(dtype=gs.qd_int, shape=(count,)),
        )

    def update(self):
        kernel_edge_scales(self.state)


@qd.func
def func_digamma_unit_interval(value: float):
    shifted = value + 8
    inverse = 1 / shifted
    square = inverse * inverse
    result = qd.log(shifted) - inverse / 2
    result += square * (-1 / 12 + square * (1 / 120 + square * (-1 / 252 + square * (1 / 240 - square / 132))))
    for k in range(8):
        result -= 1 / (value + k)
    return result


@qd.kernel
def kernel_edge_scales(edge_state: StressEdgeState):
    for i_p in range(edge_state.force.shape[0]):
        load, modulus = edge_state.force[i_p], edge_state.modulus[i_p]
        slope, curvature = edge_state.slope[i_p], edge_state.curvature[i_p]
        shear, poisson = edge_state.shear_ratio[i_p], edge_state.poisson[i_p]
        is_valid = load >= 0 and load < float("inf") and modulus > 0 and modulus < float("inf")
        is_valid = is_valid and slope > 0 and slope < float("inf") and curvature > 0 and curvature < float("inf")
        is_valid = is_valid and qd.abs(shear) < float("inf") and poisson > -1 and poisson < 0.5
        edge_state.width[i_p], edge_state.half_length[i_p] = 0, 0
        edge_state.exponent[i_p], edge_state.amplitude[i_p] = 0, 0
        edge_state.status[i_p] = 2
        if is_valid:
            edge_state.status[i_p] = 1
            if load > 0:
                edge_state.status[i_p] = 3
                beta = (1 - 2 * poisson) / (2 * (1 - poisson))
                exponent = 0.5 + qd.atan2(beta * shear, 1.0) / pi
                if exponent > 0 and exponent < 1:
                    coefficient = (load / modulus) / curvature * (3 / pi)
                    product = (load / modulus) / slope * (1.5 / (pi * exponent))
                    if coefficient > 0 and coefficient < float("inf") and product > 0 and product < float("inf"):
                        critical = (2 * coefficient / 3) ** (1 / 3)
                        constant = -func_digamma_unit_interval(1 - exponent) + 0.42278433509846714
                        constant -= 1.5 + pi * beta * shear / 2
                        target = qd.log(2 * critical * critical / product) + constant
                        if target > 2 / 3 and target < float("inf"):
                            edge_state.status[i_p] = 4
                            lower, upper = 1.0, 2 * qd.max(target, 1.0) ** (1 / 3)
                            root = (lower + upper) / 2
                            tolerance = qd.max(1e-12, 8 * gs.EPS) * qd.max(target, 1.0)
                            is_converged = False
                            for _ in range(12):
                                if not is_converged:
                                    residual = (2 / 3) * root**3 - 2 * qd.log(root) - target
                                    is_converged = qd.abs(residual) <= tolerance
                                    if not is_converged:
                                        if residual > 0:
                                            upper = root
                                        else:
                                            lower = root
                                        candidate = root - residual / (2 * root * root - 2 / root)
                                        if candidate > lower and candidate < upper:
                                            root = candidate
                                        else:
                                            root = (lower + upper) / 2
                            half_length = critical * root
                            width = product / half_length
                            amplitude = modulus * slope * qd.sin(pi * exponent) / 2
                            is_converged = qd.abs((2 / 3) * root**3 - 2 * qd.log(root) - target) <= tolerance
                            if is_converged:
                                edge_state.status[i_p] = 3
                            if (
                                is_converged
                                and width > 0
                                and width / half_length < 0.05
                                and half_length < float("inf")
                                and amplitude < float("inf")
                            ):
                                edge_state.width[i_p], edge_state.half_length[i_p] = width, half_length
                                edge_state.exponent[i_p], edge_state.amplitude[i_p] = exponent, amplitude
                                edge_state.status[i_p] = 0


@qd.func
def func_edge_pressure(
    position: qd.types.vector(2), width: float, half_length: float, exponent: float, amplitude: float
):
    extent = width * (1 - (position[1] / half_length) ** 2)
    pressure = 0.0
    if extent > 0 and position[0] >= 0 and position[0] < extent:
        if position[0] > 0:
            pressure = amplitude * ((extent - position[0]) / position[0]) ** exponent
        else:
            pressure = float("inf")
    return pressure
