"""Material probes that follow an actual normal contact's ellipse and depth."""

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind

from .data import StressInfo


@dataclass(frozen=True)
class StressHertzProbeState:
    kind: ClassVar[DataKind] = DataKind.DERIVED

    position: qd.Tensor
    element: qd.Tensor
    barycentric: qd.Tensor
    tensor: qd.Tensor


@qd.kernel
def kernel_hertz_probe_offsets(offsets: qd.Tensor):
    for i_q in range(offsets.shape[0]):
        point = qd.Vector.zero(gs.qd_float, 3)
        if i_q < 64:
            radius = 1.0 if i_q < 32 else 0.5
            angle = 2 * pi * (i_q % 32) / 32
            point = qd.Vector([radius * qd.cos(angle), radius * qd.sin(angle), 0.0])
        elif i_q > 64:
            point[2] = 0.01 * qd.exp(qd.log(400.0) * (i_q - 65) / 23)
        offsets[i_q] = point


@qd.func
def func_hertz_probe_location(
    i_e: int,
    center: qd.types.vector(3),
    position: qd.types.vector(3),
    info: StressInfo,
    walk_steps: int,
):
    """Follow the source-to-probe segment through shared material faces only."""
    bary = qd.Vector.zero(gs.qd_float, 4)
    parameter, located, failed = 0.0, False, False
    for _ in range(walk_steps):
        if not located and not failed:
            i_origin = info.elements[i_e, 0]
            origin, gradients = info.vertices[i_origin], info.gradients[i_e]
            bary = gradients @ (position - origin)
            bary[0] += 1.0
            if bary.min() >= -1e-10 and bary.max() <= 1 + 1e-10:
                located = True
            else:
                source = gradients @ (center - origin)
                source[0] += 1.0
                change = bary - source
                current = source + parameter * change
                crossing, i_face = 2.0, -1
                for i in qd.static(range(4)):
                    if current.min() >= -1e-10 and bary[i] < -1e-10 and change[i] < 0:
                        # Containment bounds barycentric roundoff, not its ratio
                        # to a possibly tiny segment derivative. A tolerated
                        # negative current coordinate crosses at this parameter.
                        value = qd.max(parameter, parameter - current[i] / change[i])
                        if value < crossing:
                            crossing, i_face = value, i
                if i_face >= 0:
                    i_next = info.element_neighbors[i_e, i_face]
                    if i_next >= 0:
                        i_e, parameter = i_next, qd.max(parameter, crossing)
                    else:
                        failed = True
                else:
                    failed = True
    if not located:
        i_e = -1
    return i_e, bary
