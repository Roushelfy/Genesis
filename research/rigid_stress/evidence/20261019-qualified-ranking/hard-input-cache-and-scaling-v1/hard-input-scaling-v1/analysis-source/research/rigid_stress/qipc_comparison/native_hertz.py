"""Inexpensive native elliptic Hertz estimates for supplied physical regions.

Inputs are the total normal load, reduced modulus and relative curvature in a
common tangent frame. Region grouping, boundary provenance and friction remain
the caller's responsibility. This component does not infer them from collider
point count, penetration or solver stiffness.
"""

from dataclasses import dataclass
from math import pi

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, V_MAT, V_VEC


@dataclass(frozen=True)
class HertzInfo:
    log_ratio: qd.Tensor
    integrals: qd.Tensor


@dataclass(frozen=True)
class HertzState:
    force: qd.Tensor
    modulus: qd.Tensor
    curvature: qd.Tensor
    axes: qd.Tensor
    direction: qd.Tensor
    indentation: qd.Tensor
    peak_pressure: qd.Tensor
    status: qd.Tensor


class NativeHertz:
    """Use one shared 257-row table and one bounded lookup per physical region.

    Status 0 admits the local Hertz calculation, 1 is unloaded, 2 is invalid
    input and 3 is unsupported curvature. Admission does not certify half-space
    assumptions, shell accuracy, boundary resolution or a tangential law.
    """

    def __init__(self, count: int):
        assert count > 0
        self.info = HertzInfo(
            log_ratio=V(dtype=gs.qd_float, shape=(257,)),
            integrals=V_VEC(2, dtype=gs.qd_float, shape=(257,)),
        )
        self.state = HertzState(
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
def kernel_hertz_table(hertz_info: HertzInfo):
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
def kernel_hertz(hertz_state: HertzState, hertz_info: HertzInfo):
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
