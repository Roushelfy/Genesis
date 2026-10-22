"""Sharp box-edge minima on a connected quadratic material surface."""

from enum import IntEnum

import quadrants as qd

import genesis as gs

from .data import StressInfo
from .surface import StressSurfaceInfo, func_surface_height_derivatives
from .surface_walk import func_projected_surface_walk


class EdgeGeometryStatus(IntEnum):
    SUCCESS = 0
    INTERIOR = 1
    INVALID_INPUT = 2
    PROJECTION = 3
    CURVATURE = 4
    ITERATIONS = 5
    CORNER = 6


@qd.func
def func_box_edge_minimum(
    anchor: int,
    center: qd.types.vector(3),
    root: qd.types.vector(3),
    normal: qd.types.vector(3),
    tangent: qd.types.vector(3),
    boundaries: qd.types.matrix(4, 3),
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    """Find a finite box edge's material minimum using connected surface walks.

    The root and boundary half-planes describe the opposing face in the
    material frame. The anchor identifies its material component. One violated
    half-plane selects an edge. Corners and nonconvex height sections return a
    failure status. The caller groups regions before predicting pressure.
    """
    status = EdgeGeometryStatus.INVALID_INPUT
    point, bary = center, qd.Vector.zero(gs.qd_float, 3)
    basis = qd.Matrix.zero(gs.qd_float, 3, 3)
    slope, curvature, residual = 0.0, 0.0, float("inf")
    position, lower, upper = 0.0, -float("inf"), float("inf")
    current, i_edge = anchor, -1
    n_outside, minimum = 0, gs.qd_float(float("inf"))
    is_valid = anchor >= 0 and anchor < info.surface_nodes.shape[0]
    is_valid = is_valid and qd.abs(normal.norm_sqr() - 1) <= 128 * epsilon
    is_valid = is_valid and qd.abs(tangent.norm_sqr() - 1) <= 128 * epsilon
    is_valid = is_valid and qd.abs(normal.dot(tangent)) <= 128 * epsilon
    is_valid = is_valid and center.norm() < float("inf") and root.norm() < float("inf")
    for i in range(4):
        plane = boundaries[i, :]
        length = qd.Vector([plane[0], plane[1]]).norm()
        is_valid = is_valid and length > 128 * epsilon and plane.norm() < float("inf")
        if length > 128 * epsilon:
            distance = plane[2] / length
            n_outside += gs.qd_int(distance < 0)
            if distance < minimum:
                i_edge, minimum = i, distance
    if is_valid:
        status = EdgeGeometryStatus.INTERIOR
        if n_outside > 1:
            status = EdgeGeometryStatus.CORNER
        elif n_outside == 1:
            status = EdgeGeometryStatus.PROJECTION
            selected = boundaries[i_edge, :]
            inward = -qd.Vector([selected[0], selected[1]]).normalized()
            along = qd.Vector([-inward[1], inward[0]])
            origin = -minimum * inward
            for i in range(4):
                if i != i_edge:
                    plane = boundaries[i, :]
                    direction = qd.Vector([plane[0], plane[1]])
                    denominator = direction.dot(along)
                    remaining = plane[2] - direction.dot(origin)
                    if denominator > 128 * epsilon:
                        upper = qd.min(upper, remaining / denominator)
                    elif denominator < -128 * epsilon:
                        lower = qd.max(lower, remaining / denominator)
                    elif remaining < 0:
                        is_valid = False
            is_valid = is_valid and lower < upper and qd.abs(lower) < float("inf") and upper < float("inf")
            transverse = tangent * inward[0] + normal.cross(tangent) * inward[1]
            longitudinal = normal.cross(transverse)
            basis = qd.Matrix.cols([transverse, longitudinal, normal])
            plane = qd.Matrix.rows([transverse, longitudinal])
            edge = root - minimum * transverse
            position = qd.min(upper, qd.max(lower, 0.0))
            if is_valid:
                status = EdgeGeometryStatus.ITERATIONS
            for i_iteration in range(9):
                if status == EdgeGeometryStatus.ITERATIONS:
                    target = edge + position * longitudinal
                    current, bary, point = func_projected_surface_walk(
                        current, point, target - point, normal, info, surface, epsilon
                    )
                    if current < 0:
                        status = EdgeGeometryStatus.PROJECTION
                    else:
                        gradient, hessian, is_regular = func_surface_height_derivatives(
                            current, bary, plane, normal, info, epsilon
                        )
                        slope, curvature, residual = gradient[0], hessian[1, 1], qd.abs(gradient[1])
                        if not is_regular or curvature <= 0 or slope <= 0:
                            status = EdgeGeometryStatus.CURVATURE
                        elif residual <= qd.max(1e-10, 128 * epsilon * (1 + gradient.norm())):
                            status = EdgeGeometryStatus.SUCCESS
                        elif i_iteration < 8:
                            position = qd.min(upper, qd.max(lower, position - gradient[1] / curvature))
    if status != EdgeGeometryStatus.SUCCESS:
        current = -1
    return current, point, bary, basis, slope, curvature, residual, status
