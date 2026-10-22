"""Follow material triangles using their shared vertex identities."""

import quadrants as qd

import genesis as gs

from .data import StressInfo
from .surface import StressSurfaceInfo, func_surface_projection


@qd.func
def func_affine_surface_walk(
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


@qd.func
def func_curved_surface_walk(
    anchor: int,
    center: qd.types.vector(3),
    offset: qd.types.vector(3),
    normal: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    """Lift a projected segment across its first outgoing quadratic edge at each face."""
    i_axis = 0
    for i in qd.static(range(1, 3)):
        if qd.abs(normal[i]) < qd.abs(normal[i_axis]):
            i_axis = i
    unit = qd.Vector.zero(gs.qd_float, 3)
    unit[i_axis] = 1.0
    tangent = normal.cross(unit).normalized()
    plane = qd.Matrix.rows([tangent, normal.cross(tangent)])
    travel = plane @ offset
    current, result = anchor, -1
    parameter = gs.qd_float(0.0)
    point, barycentric = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    is_active = anchor >= 0 and anchor < stress_info.surface_nodes.shape[0]
    for _ in range(stress_info.surface_nodes.shape[0]):
        if is_active:
            nodes = qd.Matrix.zero(gs.qd_float, 6, 3)
            for i in range(6):
                i_n = stress_info.surface_nodes[current, i]
                nodes[i, :] = stress_info.vertices[i_n]
            chord_a, chord_b = nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]
            bulge_a = nodes[3, :] - 0.5 * (nodes[0, :] + nodes[1, :])
            bulge_b = nodes[4, :] - 0.5 * (nodes[0, :] + nodes[2, :])
            bulge_c = nodes[5, :] - 0.5 * (nodes[1, :] + nodes[2, :])
            derivative_u = qd.Matrix.rows(
                [chord_a + 4 * bulge_a, chord_a - 4 * bulge_a, chord_a + 4 * (-bulge_b + bulge_c)]
            )
            derivative_v = qd.Matrix.rows(
                [chord_b + 4 * bulge_b, chord_b + 4 * (-bulge_a + bulge_c), chord_b - 4 * bulge_b]
            )
            orientation = qd.select(chord_a.cross(chord_b).dot(surface_info.face_normal[current]) > 0, 1.0, -1.0)
            scale = qd.max(chord_a.norm(), chord_b.norm())
            tolerance = 256 * epsilon * (scale + center.norm() + offset.norm() + nodes[0, :].norm())
            # Positive Bernstein coefficients bound the projected Jacobian over the whole face.
            is_facing = True
            for i, j in qd.ndrange(3, 3):
                coefficient = 0.5 * (
                    derivative_u[i, :].cross(derivative_v[j, :]) + derivative_u[j, :].cross(derivative_v[i, :])
                )
                is_facing = is_facing and orientation * coefficient.dot(normal) > 256 * epsilon * scale**2
            is_contained, is_reached = False, False
            target_point, target_bary = point, barycentric
            for i_query in range(2):
                fraction = parameter if i_query == 0 else 1.0
                target = center + fraction * offset
                located, bary, area, is_valid = func_surface_projection(current, target, plane, stress_info)
                is_valid = is_valid and (plane @ (located - target)).norm() <= tolerance
                is_valid = is_valid and bary.norm() < float("inf") and area.dot(normal) > 0
                is_inside = is_valid and bary.min() >= -256 * epsilon * (1 + center.norm() / scale)
                if i_query == 0:
                    is_contained = is_inside
                else:
                    is_reached, target_point, target_bary = is_inside, located, bary
            crossing, i_crossing = gs.qd_float(float("inf")), -1
            if is_facing and is_contained and travel.norm() > tolerance:
                perpendicular = normal.cross(offset).normalized()
                midpoints = qd.Vector([3, 5, 4])
                for i_edge in range(3):
                    i_second, i_midpoint = (i_edge + 1) % 3, midpoints[i_edge]
                    first, second, midpoint = nodes[i_edge, :], nodes[i_second, :], nodes[i_midpoint, :]
                    chord = second - first
                    bulge = midpoint - 0.5 * (first + second)
                    a = -4 * bulge.dot(perpendicular)
                    b = (chord + 4 * bulge).dot(perpendicular)
                    c = (first - center).dot(perpendicular)
                    roots = qd.Vector([-1.0, -1.0])
                    if qd.abs(a) <= 128 * epsilon * scale:
                        if qd.abs(b) > 128 * epsilon * scale:
                            roots[0] = -c / b
                    else:
                        discriminant = b * b - 4 * a * c
                        if discriminant >= -128 * epsilon * (b * b + qd.abs(4 * a * c)):
                            value = -0.5 * (b + qd.select(b >= 0, 1.0, -1.0) * qd.sqrt(qd.max(0.0, discriminant)))
                            roots[0] = value / a
                            if qd.abs(value) > 0:
                                roots[1] = c / value
                    for i_root in range(2):
                        fraction = roots[i_root]
                        if fraction >= -256 * epsilon and fraction <= 1 + 256 * epsilon:
                            fraction = qd.min(1.0, qd.max(0.0, fraction))
                            boundary = first + fraction * chord + 4 * fraction * (1 - fraction) * bulge
                            candidate = (plane @ (boundary - center)).dot(travel) / travel.norm_sqr()
                            bary = qd.Vector.zero(gs.qd_float, 3)
                            bary[i_edge], bary[i_second] = 1 - fraction, fraction
                            du, dv = derivative_u.transpose() @ bary, derivative_v.transpose() @ bary
                            change = (plane @ qd.Matrix.cols([du, dv])).inverse() @ travel
                            derivative = qd.Vector([-change.sum(), change[0], change[1]])
                            i_opposite = (i_edge + 2) % 3
                            if (
                                candidate >= parameter - tolerance / travel.norm()
                                and candidate <= 1 + tolerance / travel.norm()
                                and derivative[i_opposite] < -256 * epsilon * travel.norm() / scale
                                and candidate < crossing
                            ):
                                crossing, i_crossing = qd.max(parameter, candidate), i_edge
            if not is_facing or not is_contained:
                is_active = False
            elif is_reached and crossing >= 1 - tolerance / qd.max(travel.norm(), tolerance):
                result, point, barycentric, is_active = current, target_point, target_bary, False
            elif i_crossing >= 0:
                neighbor = stress_info.surface_neighbors[current, i_crossing]
                if neighbor < 0:
                    is_active = False
                else:
                    current, parameter = neighbor, crossing
            else:
                is_active = False
    return result, barycentric, point


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
    """Lift a projected segment through adjacent facing material surfaces."""
    result = -1
    barycentric, point = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    if stress_info.volume_points.shape[0] > 0:
        result, barycentric, point = func_curved_surface_walk(
            anchor, center, offset, normal, stress_info, surface_info, epsilon
        )
    else:
        result, barycentric, point = func_affine_surface_walk(
            anchor, center, offset, normal, stress_info, surface_info, epsilon
        )
    return result, barycentric, point
