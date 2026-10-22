"""Material probes that follow an actual normal contact's ellipse and depth."""

from dataclasses import dataclass
from math import pi
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind

from .data import StressInfo
from .operators import func_material_gradients, func_material_position
from .surface import func_surface_projection


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


@qd.func
def func_hertz_curved_probe_location(
    i_f: int,
    center: qd.types.vector(3),
    basis: qd.types.matrix(3, 3),
    local: qd.types.vector(3),
    info: StressInfo,
    walk_steps: int,
    epsilon: float,
):
    """Follow shared surface and volume faces without crossing to another sheet."""
    plane = basis[:, :2].transpose()
    target = center + basis[:, 0] * local[0] + basis[:, 1] * local[1]
    surface_point, position = target, target
    is_located, is_failed = False, False
    i_e = -1
    bary = qd.Vector.zero(gs.qd_float, 4)
    for _ in range(walk_steps):
        if not is_located and not is_failed:
            surface_point, surface_bary, area, is_valid = func_surface_projection(i_f, target, plane, info)
            is_valid = is_valid and (plane @ (surface_point - target)).norm() <= epsilon
            is_valid = is_valid and area.dot(basis[:, 2]) < 0 and surface_bary.norm() < float("inf")
            if not is_valid:
                is_failed = True
            elif surface_bary.min() >= -1e-10 and surface_bary.max() <= 1 + 1e-10:
                is_located = True
                i_e = info.surface_owners[i_f]
                position = surface_point + basis[:, 2] * local[2]
            else:
                i_opposite = 0
                for i in qd.static(range(1, 3)):
                    if surface_bary[i] < surface_bary[i_opposite]:
                        i_opposite = i
                i_next = info.surface_neighbors[i_f, (i_opposite + 1) % 3]
                if i_next >= 0:
                    i_f = i_next
                else:
                    is_failed = True
    is_failed = is_failed or not is_located
    is_located = False
    for _ in range(walk_steps):
        if not is_located and not is_failed:
            i_origin = info.elements[i_e, 0]
            bary = info.gradients[i_e] @ (position - info.vertices[i_origin])
            bary[0] += 1
            located = position
            for i_iteration in range(9):
                located = func_material_position(i_e, bary, info, is_curved=True)
                if i_iteration < 8:
                    gradients = func_material_gradients(i_e, bary, info, is_curved=True)
                    bary -= gradients @ (located - position)
            is_valid = bary.norm() < float("inf") and (located - position).norm() <= epsilon
            if not is_valid:
                is_failed = True
            elif bary.min() >= -1e-10 and bary.max() <= 1 + 1e-10:
                is_located = True
            else:
                i_opposite = 0
                for i in qd.static(range(1, 4)):
                    if bary[i] < bary[i_opposite]:
                        i_opposite = i
                i_next = info.element_neighbors[i_e, i_opposite]
                if i_next >= 0:
                    i_e = i_next
                else:
                    is_failed = True
    if not is_located:
        i_e = -1
    return position, i_e, bary
