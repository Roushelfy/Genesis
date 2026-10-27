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
    grid_positions: qd.Tensor
    grid_shapes: qd.Tensor
    triangle_nodes: qd.Tensor
    cell_nodes: qd.Tensor
    cell_bounds_low: qd.Tensor
    cell_bounds_high: qd.Tensor
    face_bounds_low: qd.Tensor
    face_bounds_high: qd.Tensor
    parent_face: qd.Tensor
    side: qd.Tensor
    side_index: qd.Tensor
    parent_neighbor_start: qd.Tensor
    parent_neighbor_faces: qd.Tensor
    geometry_error: qd.Tensor
    normal_error: qd.Tensor
    face_valid: qd.Tensor


def build_planar_geometry(surface: StressSurfaceInfo, load_info: StressInfo, subdivisions: int) -> PlanarLoadingInfo:
    """Refine both physical shell sides while retaining the original nodal displacement maps.

    Refinement uses four, eight or sixteen segments along each original quadrilateral edge. Shared local shape
    tables and per-face grid positions determine every planar triangle. Parent adjacency follows authored mesh edges.
    """
    if subdivisions not in (4, 8, 16):
        gs.raise_exception("Planar shell loading subdivisions must be 4, 8 or 16.")
    n_faces = surface.quad_vertices.shape[0]
    if n_faces == 0 or n_faces % 2 or load_info.surface_nodes.shape[0] != n_faces:
        gs.raise_exception("Planar shell loading requires both physical quadrilateral surface sides.")
    n_parents = n_faces // 2
    quads = qd_to_numpy(load_info.surface_nodes, transpose=True)[:4, :n_parents].T
    edge_faces = {}
    for i_f, quad in enumerate(quads):
        for i in range(4):
            edge = tuple(sorted((quad[i], quad[(i + 1) % 4])))
            edge_faces.setdefault(edge, []).append(i_f)
    adjacent = [set() for _ in range(n_parents)]
    for faces in edge_faces.values():
        for i_f in faces:
            adjacent[i_f].update(i_other for i_other in faces if i_other != i_f)
    starts, neighbors = [0], []
    for faces in adjacent:
        neighbors.extend(sorted(faces))
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
        grid_positions=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, n_grid)),
        grid_shapes=V_VEC(4, dtype=gs.qd_float, shape=(2, n_grid)),
        triangle_nodes=V_VEC(3, dtype=gs.qd_int, shape=triangles.shape[:2]),
        cell_nodes=V_VEC(4, dtype=gs.qd_int, shape=(len(cells),)),
        cell_bounds_low=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, len(cells))),
        cell_bounds_high=V_VEC(3, dtype=gs.qd_float, shape=(n_faces, len(cells))),
        face_bounds_low=surface.face_bounds_low,
        face_bounds_high=surface.face_bounds_high,
        parent_face=V(dtype=gs.qd_int, shape=(n_faces,)),
        side=V(dtype=gs.qd_int, shape=(n_faces,)),
        side_index=V(dtype=gs.qd_int, shape=(n_faces,)),
        parent_neighbor_start=V(dtype=gs.qd_int, shape=(len(starts),)),
        parent_neighbor_faces=V_VEC(1, dtype=gs.qd_int, shape=(len(neighbors),)),
        geometry_error=V(dtype=gs.qd_float, shape=(n_faces,)),
        normal_error=V(dtype=gs.qd_float, shape=(n_faces,)),
        face_valid=V(dtype=gs.qd_bool, shape=(n_faces,)),
    )
    info.subdivisions.fill(subdivisions)
    info.triangle_nodes.from_numpy(triangles)
    info.cell_nodes.from_numpy(np.array(cells, dtype=gs.np_int))
    info.parent_neighbor_start.from_numpy(np.array(starts, dtype=gs.np_int))
    if neighbors:
        info.parent_neighbor_faces.from_numpy(np.array(neighbors, dtype=gs.np_int)[:, None])
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
def func_parent_adjacent(i_first_parent: int, i_second_parent: int, info: PlanarLoadingInfo):
    """Test material adjacency by original quadrilateral edge identity."""
    adjacent = i_first_parent == i_second_parent
    for i_n in range(info.parent_neighbor_start[i_first_parent], info.parent_neighbor_start[i_first_parent + 1]):
        adjacent = adjacent or info.parent_neighbor_faces[i_n][0] == i_second_parent
    return adjacent
