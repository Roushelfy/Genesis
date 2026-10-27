"""Prepare compact planar loading geometry and consistent physical shell interpolation."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .data import StressInfo
from .quadrilateral import func_quad_shape
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class PlanarLoadingInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    subdivisions: qd.Tensor
    epsilon: qd.Tensor
    minimum_normal: qd.Tensor
    grid_positions: qd.Tensor
    grid_shapes: qd.Tensor
    triangle_nodes: qd.Tensor
    cell_nodes: qd.Tensor
    cell_bounds_low: qd.Tensor
    cell_bounds_high: qd.Tensor
    cell_block_bounds_low: qd.Tensor
    cell_block_bounds_high: qd.Tensor
    face_bounds_low: qd.Tensor
    face_bounds_high: qd.Tensor
    parent_face: qd.Tensor
    side: qd.Tensor
    side_index: qd.Tensor
    parent_neighbor_start: qd.Tensor
    parent_neighbor_faces: qd.Tensor
    parent_neighbor_edges: qd.Tensor
    graph_curvature_bound: qd.Tensor
    geometry_error: qd.Tensor
    normal_error: qd.Tensor
    face_valid: qd.Tensor


def build_planar_geometry(surface: StressSurfaceInfo, load_info: StressInfo, subdivisions: int) -> PlanarLoadingInfo:
    """Refine both physical shell sides while retaining the original nodal displacement maps.

    Refinement uses four, eight, sixteen or thirty-two segments along each original quadrilateral edge. Shared local shape
    tables and per-face grid positions determine every planar triangle. Parent adjacency follows authored mesh edges.
    """
    if subdivisions not in (4, 8, 16, 32):
        gs.raise_exception("Planar shell loading subdivisions must be 4, 8, 16 or 32.")
    n_faces = surface.quad_vertices.shape[0]
    if n_faces == 0 or n_faces % 2 or load_info.surface_nodes.shape[0] != n_faces:
        gs.raise_exception("Planar shell loading requires both physical quadrilateral surface sides.")
    n_parents = n_faces // 2
    quads = qd_to_numpy(load_info.surface_nodes, transpose=True)[:4, :n_parents].T
    edge_faces = {}
    for i_f, quad in enumerate(quads):
        for i in range(4):
            edge = tuple(sorted((quad[i], quad[(i + 1) % 4])))
            edge_faces.setdefault(edge, []).append((i_f, i, (i + 1) % 4))
    adjacent = [[] for _ in range(n_parents)]
    for faces in edge_faces.values():
        for i_f, i_first, i_second in faces:
            adjacent[i_f].extend((i_other, i_first, i_second) for i_other, _first, _second in faces if i_other != i_f)
    starts, neighbors, shared_edges = [0], [], []
    for faces in adjacent:
        for i_other, i_first, i_second in sorted(faces):
            neighbors.append(i_other)
            shared_edges.append((i_first, i_second))
        starts.append(len(neighbors))

    cells, triangles = [], []
    for j in range(subdivisions):
        for i in range(subdivisions):
            a = j * (subdivisions + 1) + i
            b, c, d = a + 1, a + subdivisions + 2, a + subdivisions + 1
            cells.append((a, b, c, d))
            triangles.extend(((a, b, c), (a, c, d)))
    triangles = np.array(triangles, dtype=gs.np_int)
    # The inner physical chart reverses its node order. Swapping its coordinates retains common material tile IDs.
    triangles = np.stack((triangles, triangles[:, (0, 2, 1)]))
    n_grid = (subdivisions + 1) ** 2
    info = PlanarLoadingInfo(
        subdivisions=V(dtype=gs.qd_int, shape=()),
        epsilon=V(dtype=gs.qd_float, shape=()),
        minimum_normal=V(dtype=gs.qd_float, shape=()),
        grid_positions=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, n_grid)),
        grid_shapes=V_VEC(4, dtype=gs.qd_float, shape=(2, n_grid)),
        triangle_nodes=V_VEC(3, dtype=gs.qd_int, shape=triangles.shape[:2]),
        cell_nodes=V_VEC(4, dtype=gs.qd_int, shape=(len(cells),)),
        cell_bounds_low=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, len(cells))),
        cell_bounds_high=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, len(cells))),
        cell_block_bounds_low=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, (subdivisions // 4) ** 2)),
        cell_block_bounds_high=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, (subdivisions // 4) ** 2)),
        face_bounds_low=surface.face_bounds_low,
        face_bounds_high=surface.face_bounds_high,
        parent_face=V(dtype=gs.qd_int, shape=(n_faces,)),
        side=V(dtype=gs.qd_int, shape=(n_faces,)),
        side_index=V(dtype=gs.qd_int, shape=(n_faces,)),
        parent_neighbor_start=V(dtype=gs.qd_int, shape=(len(starts),)),
        parent_neighbor_faces=V_VEC(1, dtype=gs.qd_int, shape=(len(neighbors),)),
        parent_neighbor_edges=V_VEC(2, dtype=gs.qd_int, shape=(len(shared_edges),)),
        graph_curvature_bound=V(dtype=gs.qd_float, shape=(n_faces,)),
        geometry_error=V(dtype=gs.qd_float, shape=(n_faces,)),
        normal_error=V(dtype=gs.qd_float, shape=(n_faces,)),
        face_valid=V(dtype=gs.qd_bool, shape=(n_faces,)),
    )
    info.subdivisions.fill(subdivisions)
    info.epsilon.fill(float(np.finfo(gs.np_float).eps))
    info.minimum_normal.fill(float(np.finfo(gs.np_float).tiny))
    info.triangle_nodes.from_numpy(triangles)
    info.cell_nodes.from_numpy(np.array(cells, dtype=gs.np_int))
    info.parent_neighbor_start.from_numpy(np.array(starts, dtype=gs.np_int))
    if neighbors:
        info.parent_neighbor_faces.from_numpy(np.array(neighbors, dtype=gs.np_int)[:, None])
        info.parent_neighbor_edges.from_numpy(np.array(shared_edges, dtype=gs.np_int))
    kernel_planar_geometry(surface, info)
    return info


@qd.kernel
def kernel_planar_geometry(surface: StressSurfaceInfo, info: PlanarLoadingInfo):
    for i_side, i_g in qd.ndrange(2, info.grid_shapes.shape[1]):
        k = info.subdivisions[None]
        xi, eta = 2.0 * (i_g % (k + 1)) / k - 1.0, 2.0 * (i_g // (k + 1)) / k - 1.0
        if i_side == 1:
            xi, eta = eta, xi
        shape, _dxi, _deta = func_quad_shape(xi, eta)
        info.grid_shapes[i_side, i_g] = shape
    for i_f in range(info.face_valid.shape[0]):
        i_side = i_f // (info.face_valid.shape[0] // 2)
        info.parent_face[i_f] = i_f % (info.face_valid.shape[0] // 2)
        info.side_index[i_f] = i_side
        info.side[i_f] = qd.select(i_side == 0, 1, -1)
        info.face_valid[i_f] = True
        info.normal_error[i_f] = 0.0
        coefficients = surface.quad_coefficients[i_f]
        info.graph_curvature_bound[i_f] = func_quad_graph_curvature(
            coefficients[1, :],
            coefficients[2, :],
            coefficients[3, :],
            info.epsilon[None],
            info.minimum_normal[None],
        )
        info.geometry_error[i_f] = surface.quad_coefficients[i_f][3, :].norm() / (
            info.subdivisions[None] * info.subdivisions[None]
        )
    for i_f, i_g in qd.ndrange(info.grid_positions.shape[0], info.grid_positions.shape[1]):
        i_side = info.side_index[i_f]
        info.grid_positions[i_f, i_g] = info.grid_shapes[i_side, i_g] @ surface.quad_vertices[i_f]
    for i_f, i_cell in qd.ndrange(info.cell_bounds_low.shape[0], info.cell_bounds_low.shape[1]):
        nodes = info.cell_nodes[i_cell]
        i_a, i_b, i_c, i_d = nodes[0], nodes[1], nodes[2], nodes[3]
        a, b = info.grid_positions[i_f, i_a], info.grid_positions[i_f, i_b]
        c, d = info.grid_positions[i_f, i_c], info.grid_positions[i_f, i_d]
        info.cell_bounds_low[i_f, i_cell] = qd.min(a, b, c, d)
        info.cell_bounds_high[i_f, i_cell] = qd.max(a, b, c, d)
    for i_f, i_cell_block in qd.ndrange(info.cell_block_bounds_low.shape[0], info.cell_block_bounds_low.shape[1]):
        k = info.subdivisions[None]
        n_blocks_axis = k // 4
        i_first = (i_cell_block // n_blocks_axis * 4) * k + i_cell_block % n_blocks_axis * 4
        low = info.cell_bounds_low[i_f, i_first]
        high = info.cell_bounds_high[i_f, i_first]
        for i_local in range(16):
            i_cell = i_first + i_local // 4 * k + i_local % 4
            low = qd.min(low, info.cell_bounds_low[i_f, i_cell])
            high = qd.max(high, info.cell_bounds_high[i_f, i_cell])
        info.cell_block_bounds_low[i_f, i_cell_block] = low
        info.cell_block_bounds_high[i_f, i_cell_block] = high
    for i_f, i_t in qd.ndrange(info.grid_positions.shape[0], info.triangle_nodes.shape[1]):
        i_side = info.side_index[i_f]
        nodes = info.triangle_nodes[i_side, i_t]
        i_origin, i_first, i_second = nodes[0], nodes[1], nodes[2]
        origin = info.grid_positions[i_f, i_origin]
        first = info.grid_positions[i_f, i_first] - origin
        second = info.grid_positions[i_f, i_second] - origin
        normal = first.cross(second)
        if normal.norm() <= 1e-24:
            info.face_valid[i_f] = False
        else:
            mixed = surface.quad_coefficients[i_f][3, :]
            error = qd.abs(mixed.dot(normal.normalized())) / (info.subdivisions[None] * info.subdivisions[None])
            qd.atomic_max(info.normal_error[i_f], error)


@qd.func
def func_triangle_geometry(i_face: int, i_local_triangle: int, info: PlanarLoadingInfo):
    """Return physical vertices, coordinate gradients, outward normal and parent interpolation of one triangle."""
    i_side = info.side_index[i_face]
    nodes = info.triangle_nodes[i_side, i_local_triangle]
    vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
    parent_shapes = qd.Matrix.zero(gs.qd_float, 3, 4)
    for a in qd.static(range(3)):
        i_node = nodes[a]
        vertices[a, :] = info.grid_positions[i_face, i_node]
        parent_shapes[a, :] = info.grid_shapes[i_side, i_node]
    first, second = vertices[1, :] - vertices[0, :], vertices[2, :] - vertices[0, :]
    gram = qd.Matrix([[first.dot(first), first.dot(second)], [first.dot(second), second.dot(second)]])
    dual = qd.Matrix.zero(gs.qd_float, 2, 3)
    if gram.determinant() > 1e-48:
        dual = gram.inverse() @ qd.Matrix.rows([first, second])
    normal = first.cross(second)
    normal /= qd.max(normal.norm(), 1e-30)
    return vertices, dual, normal, parent_shapes


@qd.func
def func_face_overlap(i_face: int, center: qd.types.vector(3), radius: float, info: PlanarLoadingInfo):
    """Bound a sphere against the original physical face's complete corner hull."""
    separation = qd.max(info.face_bounds_low[i_face] - center, 0.0)
    separation += qd.max(center - info.face_bounds_high[i_face], 0.0)
    return separation.norm_sqr() <= radius * radius


@qd.func
def func_cell_overlap(i_face: int, i_cell: int, center: qd.types.vector(3), radius: float, info: PlanarLoadingInfo):
    """Bound a sphere against the two planar triangles of one material tile."""
    separation = qd.max(info.cell_bounds_low[i_face, i_cell] - center, 0.0)
    separation += qd.max(center - info.cell_bounds_high[i_face, i_cell], 0.0)
    return separation.norm_sqr() <= radius * radius


@qd.func
def func_cell_block_overlap(
    i_face: int, i_block: int, center: qd.types.vector(3), radius: float, info: PlanarLoadingInfo
):
    """Bound a sphere against the exact corner hull of a four-by-four block of original material tiles."""
    separation = qd.max(info.cell_block_bounds_low[i_face, i_block] - center, 0.0)
    separation += qd.max(center - info.cell_block_bounds_high[i_face, i_block], 0.0)
    return separation.norm_sqr() <= radius * radius


@qd.func
def func_quad_graph_curvature(
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    mixed: qd.types.vector(3),
    epsilon: float,
    minimum_normal: float,
):
    """Bound graph-height Hessians on a globally invertible convex Q4 projection.

    An infinite result leaves connectivity uncertified. The projected Jacobian
    has an affine determinant on the parameter square. Its positive lower
    bound and singular-value bounds control the inverse chart and height Hessian.
    """
    curvature = gs.qd_float(float("inf"))
    scale = qd.max(qd.abs(first).max(), qd.abs(second).max(), qd.abs(mixed).max())
    if scale > 0.0 and scale < float("inf"):
        # Rescaling avoids physical-length powers entering the subnormal range. An extreme aspect ratio remains uncertified.
        first_scaled, second_scaled, mixed_scaled = first / scale, second / scale, mixed / scale
        first_squared, second_squared, mixed_squared = (
            first_scaled.norm_sqr(),
            second_scaled.norm_sqr(),
            mixed_scaled.norm_sqr(),
        )
        cross = first_scaled.cross(second_scaled)
        area_squared = cross.norm_sqr()
        mixed_regular = mixed_squared >= minimum_normal or qd.abs(mixed).max() == 0.0
        if (
            first_squared >= minimum_normal
            and second_squared >= minimum_normal
            and area_squared >= minimum_normal
            and mixed_regular
        ):
            first_norm, second_norm, mixed_norm = (
                qd.sqrt(first_squared),
                qd.sqrt(second_squared),
                qd.sqrt(mixed_squared),
            )
            area = qd.sqrt(area_squared)
            # The extra chart-conditioning factor covers normalization of the cross product.
            margin = 128.0 * epsilon * (1.0 + first_norm * second_norm / area)
            normal = cross / area
            height = normal.dot(mixed_scaled)
            height_upper = qd.abs(height) + margin * mixed_norm
            tangent_upper = (mixed_scaled - height * normal).norm() + margin * mixed_norm
            determinant_lower = area - qd.abs(normal.dot(first_scaled.cross(mixed_scaled)))
            determinant_lower -= qd.abs(normal.dot(mixed_scaled.cross(second_scaled)))
            determinant_lower -= margin * (
                first_norm * second_norm + first_norm * mixed_norm + mixed_norm * second_norm
            )
            area_lower = area - margin * first_norm * second_norm
            trace = first_squared + second_squared
            discriminant = (first_squared - second_squared) ** 2 + 4.0 * first_scaled.dot(second_scaled) ** 2
            largest_upper = qd.sqrt(0.5 * (trace + qd.sqrt(qd.max(0.0, discriminant))))
            largest_upper += margin * (first_norm + second_norm)
            if margin < 0.25 and determinant_lower > 0.0 and area_lower > 0.0 and largest_upper > 0.0:
                origin_sigma_lower = area_lower / largest_upper
                weyl_lower = qd.max(0.0, origin_sigma_lower - qd.sqrt(2.0) * tangent_upper)
                frobenius_upper = qd.sqrt(
                    (first_norm * (1.0 + margin) + tangent_upper) ** 2
                    + (second_norm * (1.0 + margin) + tangent_upper) ** 2
                )
                frobenius_upper *= 1.0 + margin
                sigma_lower = qd.max(weyl_lower, determinant_lower / frobenius_upper)
                sigma_lower *= 1.0 - margin
                sigma_squared = sigma_lower * sigma_lower
                if sigma_lower > 0.0 and sigma_squared >= minimum_normal:
                    normalized = height_upper * (1.0 + qd.sqrt(2.0) * tangent_upper / sigma_lower)
                    normalized *= (1.0 + margin) / sigma_squared
                    candidate = normalized / scale
                    if candidate < float("inf") and (candidate >= minimum_normal or normalized == 0.0):
                        curvature = candidate
    return curvature


@qd.func
def func_parent_ball_connected(i_face: int, radius: float, info: PlanarLoadingInfo):
    """Certify that a positive ball footprint on one physical Q4 is connected."""
    bound = info.graph_curvature_bound[i_face]
    return radius > 0.0 and radius < float("inf") and bound >= 0.0 and radius * bound < 1.0 - 128.0 * info.epsilon[None]


@qd.func
def func_segment_ball(
    first: qd.types.vector(3),
    second: qd.types.vector(3),
    center: qd.types.vector(3),
    radius: float,
    epsilon: float,
    minimum_normal: float,
):
    """Return an open-ball intersection witness and whether its classification is resolved."""
    inside, certain = False, False
    if radius >= minimum_normal and radius < float("inf"):
        relative = (first - center) / radius
        edge = (second - first) / radius
        length_squared = edge.norm_sqr()
        scale = 1.0 + (qd.abs(first).sum() + qd.abs(second).sum() + 2.0 * qd.abs(center).sum()) / radius
        margin = 128.0 * epsilon * scale
        if length_squared > margin * margin and length_squared < float("inf") and margin < 0.25:
            fraction = qd.min(1.0, qd.max(0.0, -relative.dot(edge) / length_squared))
            distance = (relative + fraction * edge).norm()
            inside = distance < 1.0 - margin
            certain = inside or distance > 1.0 + margin
    return inside, certain


@qd.func
def func_parent_edge_ball(
    i_parent: int,
    i_neighbor: int,
    side_filter: int,
    center: qd.types.vector(3),
    radius: float,
    info: PlanarLoadingInfo,
):
    """Test the actual shared material edge on the admitted physical sides."""
    inside, uncertain = False, side_filter != 0 and side_filter != 1 and side_filter != -1
    corners = info.parent_neighbor_edges[i_neighbor]
    k = info.subdivisions[None]
    grid_corners = qd.Vector([0, k, k * (k + 2), k * (k + 1)])
    i_first_corner, i_second_corner = corners[0], corners[1]
    i_first, i_second = grid_corners[i_first_corner], grid_corners[i_second_corner]
    for i_side in range(2):
        if side_filter == 0 or side_filter == qd.select(i_side == 0, 1, -1):
            i_face = i_parent + i_side * (info.parent_neighbor_start.shape[0] - 1)
            side_inside, side_certain = func_segment_ball(
                info.grid_positions[i_face, i_first],
                info.grid_positions[i_face, i_second],
                center,
                radius,
                info.epsilon[None],
                info.minimum_normal[None],
            )
            inside = inside or side_inside
            uncertain = uncertain or not side_certain
    return inside, inside or not uncertain


@qd.func
def func_parents_ball_connected(
    i_first_parent: int,
    i_second_parent: int,
    side_filter: int,
    center: qd.types.vector(3),
    radius: float,
    info: PlanarLoadingInfo,
):
    """Resolve parent adjacency through physical shared edges inside the support ball."""
    connected, uncertain = i_first_parent == i_second_parent, False
    for i_n in range(info.parent_neighbor_start[i_first_parent], info.parent_neighbor_start[i_first_parent + 1]):
        if info.parent_neighbor_faces[i_n][0] == i_second_parent:
            inside, certain = func_parent_edge_ball(i_first_parent, i_n, side_filter, center, radius, info)
            connected = connected or inside
            uncertain = uncertain or not certain
    return connected, connected or not uncertain
