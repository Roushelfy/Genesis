"""Follow material triangles using their shared vertex identities."""

import quadrants as qd

import genesis as gs

from .data import StressInfo
from .surface import StressSurfaceInfo


@qd.func
def func_projected_surface_walk(
    anchor: int,
    center: qd.types.vector(3),
    offset: qd.types.vector(3),
    normal: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    """Lift a tangent-plane segment through adjacent facing triangles only."""
    current, result = anchor, -1
    parameter = gs.qd_float(0.0)
    point, barycentric = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    active = anchor >= 0 and anchor < stress_info.surface_nodes.shape[0]
    for _ in range(stress_info.surface_nodes.shape[0]):
        if active:
            origin = surface_info.face_origin[current]
            face_normal = surface_info.face_normal[current]
            denominator = normal.dot(face_normal)
            if denominator <= 64 * epsilon:
                active = False
            else:
                start = center + normal * face_normal.dot(origin - center) / denominator
                direction = offset - normal * face_normal.dot(offset) / denominator
                dual = surface_info.face_dual[current]
                local = dual @ (start - origin)
                change = dual @ direction
                bary = qd.Vector([1 - local.sum(), local[0], local[1]])
                derivative = qd.Vector([-change.sum(), change[0], change[1]])
                tolerance = 64 * epsilon * (1 + (center.norm() + origin.norm() + offset.norm()) * dual.norm())
                contained, reached = True, True
                for a in qd.static(range(3)):
                    contained = contained and bary[a] + parameter * derivative[a] >= -tolerance
                    reached = reached and bary[a] + derivative[a] >= -tolerance
                if not contained:
                    active = False
                elif reached:
                    point, barycentric = start + direction, bary + derivative
                    result, active = current, False
                else:
                    crossing, edge = gs.qd_float(float("inf")), -1
                    for a in qd.static(range(3)):
                        if derivative[a] < -tolerance:
                            current_bary = bary[a] + parameter * derivative[a]
                            candidate = qd.max(parameter, parameter - current_bary / derivative[a])
                            if candidate < crossing:
                                crossing, edge = candidate, (a + 1) % 3
                    if edge < 0 or crossing > 1 + tolerance:
                        active = False
                    else:
                        neighbor = stress_info.surface_neighbors[current, edge]
                        if neighbor < 0:
                            active = False
                        else:
                            current, parameter = neighbor, qd.max(parameter, crossing)
    return result, barycentric, point
