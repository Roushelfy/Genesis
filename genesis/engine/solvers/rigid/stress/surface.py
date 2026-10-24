"""Native positive quadrature of affine P2 and physical bilinear Q4 surfaces."""

import time
from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V

from .data import StressInfo
from .grid import StressGridInfo, allocate_grid, build_grid
from .quadrilateral import func_quad_shape


@dataclass(frozen=True)
class StressSurfaceInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    gauss: qd.Tensor
    gauss_weights: qd.Tensor
    positions: qd.Tensor
    weights: qd.Tensor
    shape: qd.Tensor
    face_origin: qd.Tensor
    face_edges: qd.Tensor
    face_dual: qd.Tensor
    face_normal: qd.Tensor
    face_bounds_low: qd.Tensor
    face_bounds_high: qd.Tensor
    quad_vertices: qd.Tensor
    quad_valid: qd.Tensor
    grid: StressGridInfo


class StressSurface:
    def __init__(self, quadrature: int, stress_info: StressInfo):
        started = time.perf_counter()
        n_faces = stress_info.surface_nodes.shape[0]
        n_quadrature = quadrature * quadrature
        self.info = StressSurfaceInfo(
            gauss=V(dtype=gs.qd_float, shape=(quadrature,)),
            gauss_weights=V(dtype=gs.qd_float, shape=(quadrature,)),
            positions=V_VEC(3, dtype=gs.qd_float, shape=(n_faces * n_quadrature,)),
            weights=V(dtype=gs.qd_float, shape=(n_faces * n_quadrature,)),
            shape=V_VEC(6, dtype=gs.qd_float, shape=(n_quadrature,)),
            face_origin=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_edges=V_MAT(2, 3, dtype=gs.qd_float, shape=(n_faces,)),
            face_dual=V_MAT(2, 3, dtype=gs.qd_float, shape=(n_faces,)),
            face_normal=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_bounds_low=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            face_bounds_high=V_VEC(3, dtype=gs.qd_float, shape=(n_faces,)),
            quad_vertices=V_MAT(
                4, 3, dtype=gs.qd_float, shape=(n_faces if stress_info.load_nodes.shape[0] > 0 else 0,)
            ),
            quad_valid=V(dtype=gs.qd_bool, shape=()),
            grid=allocate_grid(n_faces * n_quadrature),
        )
        kernel_gauss(self.info)
        if stress_info.load_nodes.shape[0] > 0:
            kernel_quad_surface(stress_info, self.info)
            if not self.info.quad_valid.to_numpy().item():
                gs.raise_exception("Shell physical sides require regular convex projected Q4 faces.")
        else:
            kernel_surface(stress_info, self.info)
        build_grid(self.info.positions, self.info.grid)
        qd.sync()
        self.build_seconds = time.perf_counter() - started


@qd.kernel
def kernel_gauss(surface_info: StressSurfaceInfo):
    for i in range(surface_info.gauss.shape[0]):
        n = surface_info.gauss.shape[0]
        x = -qd.cos(3.141592653589793 * (i + 0.75) / (n + 0.5))
        derivative = gs.qd_float(0.0)
        for _ in range(32):
            previous, value = gs.qd_float(0.0), gs.qd_float(1.0)
            for j in range(n):
                old = value
                value = ((2 * j + 1) * x * value - j * previous) / (j + 1)
                previous = old
            derivative = n * (x * value - previous) / (x * x - 1.0)
            x -= value / derivative
        surface_info.gauss[i] = 0.5 * (x + 1.0)
        surface_info.gauss_weights[i] = 1.0 / ((1.0 - x * x) * derivative * derivative)


@qd.kernel
def kernel_surface(stress_info: StressInfo, surface_info: StressSurfaceInfo):
    for i_f in range(stress_info.surface_nodes.shape[0]):
        i_0 = stress_info.surface_nodes[i_f, 0]
        i_1 = stress_info.surface_nodes[i_f, 1]
        i_2 = stress_info.surface_nodes[i_f, 2]
        x = stress_info.vertices[i_0]
        first, second = stress_info.vertices[i_1] - x, stress_info.vertices[i_2] - x
        gram = qd.Matrix([[first.dot(first), first.dot(second)], [second.dot(first), second.dot(second)]])
        surface_info.face_origin[i_f] = x
        surface_info.face_edges[i_f] = qd.Matrix.rows([first, second])
        surface_info.face_bounds_low[i_f] = qd.min(x, x + first, x + second)
        surface_info.face_bounds_high[i_f] = qd.max(x, x + first, x + second)
        surface_info.face_dual[i_f] = gram.inverse() @ qd.Matrix.rows([first, second])
        normal = first.cross(second).normalized()
        i_opposite = stress_info.surface_opposite_vertices[i_f]
        if normal.dot(stress_info.vertices[i_opposite] - x) > 0.0:
            normal = -normal
        surface_info.face_normal[i_f] = normal
        for i_q in range(surface_info.shape.shape[0]):
            n = surface_info.gauss.shape[0]
            u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
            bary = qd.Vector([1.0 - u, u * (1.0 - v), u * v])
            i_sample = i_f * surface_info.shape.shape[0] + i_q
            surface_info.positions[i_sample] = x + bary[1] * first + bary[2] * second
            surface_info.weights[i_sample] = (
                first.cross(second).norm()
                * u
                * surface_info.gauss_weights[i_q // n]
                * surface_info.gauss_weights[i_q % n]
            )
    for i_q in range(surface_info.shape.shape[0]):
        n = surface_info.gauss.shape[0]
        u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
        bary = qd.Vector([1.0 - u, u * (1.0 - v), u * v])
        surface_info.shape[i_q] = qd.Vector(
            [
                bary[0] * (2.0 * bary[0] - 1.0),
                bary[1] * (2.0 * bary[1] - 1.0),
                bary[2] * (2.0 * bary[2] - 1.0),
                4.0 * bary[0] * bary[1],
                4.0 * bary[0] * bary[2],
                4.0 * bary[1] * bary[2],
            ]
        )


@qd.kernel
def kernel_quad_surface(stress_info: StressInfo, surface_info: StressSurfaceInfo):
    for _ in range(1):
        surface_info.quad_valid[None] = True
    for i_f in range(stress_info.surface_nodes.shape[0]):
        vertices = qd.Matrix.zero(gs.qd_float, 4, 3)
        for i in qd.static(range(4)):
            i_n = stress_info.surface_nodes[i_f, i]
            vertices[i, :] = stress_info.vertices[i_n]
        origin = 0.25 * (vertices[0, :] + vertices[1, :] + vertices[2, :] + vertices[3, :])
        first = 0.25 * (-vertices[0, :] + vertices[1, :] + vertices[2, :] - vertices[3, :])
        second = 0.25 * (-vertices[0, :] - vertices[1, :] + vertices[2, :] + vertices[3, :])
        normal = first.cross(second).normalized()
        tangent = first.normalized()
        surface_info.quad_vertices[i_f] = vertices
        surface_info.face_origin[i_f] = origin
        surface_info.face_edges[i_f] = qd.Matrix.rows([2.0 * first, 2.0 * second])
        surface_info.face_dual[i_f] = qd.Matrix.rows([tangent, normal.cross(tangent)])
        surface_info.face_normal[i_f] = normal
        surface_info.face_bounds_low[i_f] = qd.min(vertices[0, :], vertices[1, :], vertices[2, :], vertices[3, :])
        surface_info.face_bounds_high[i_f] = qd.max(vertices[0, :], vertices[1, :], vertices[2, :], vertices[3, :])
        for i in range(4):
            edge = vertices[(i + 1) % 4, :] - vertices[i, :]
            following = vertices[(i + 2) % 4, :] - vertices[(i + 1) % 4, :]
            if edge.cross(following).dot(normal) <= 1e-12 * edge.norm() * following.norm():
                surface_info.quad_valid[None] = False
        for i_q in range(surface_info.shape.shape[0]):
            n = surface_info.gauss.shape[0]
            u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
            shape, dxi, deta = func_quad_shape(2.0 * u - 1.0, 2.0 * v - 1.0)
            i_sample = i_f * surface_info.shape.shape[0] + i_q
            surface_info.positions[i_sample] = shape @ vertices
            surface_info.weights[i_sample] = (
                4.0
                * (dxi @ vertices).cross(deta @ vertices).norm()
                * surface_info.gauss_weights[i_q // n]
                * surface_info.gauss_weights[i_q % n]
            )
    for i_q in range(surface_info.shape.shape[0]):
        n = surface_info.gauss.shape[0]
        u, v = surface_info.gauss[i_q // n], surface_info.gauss[i_q % n]
        surface_info.shape[i_q] = qd.Vector([(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v, 0.0, 0.0])


@qd.func
def func_closest_triangle(i_f: int, point: qd.types.vector(3), surface_info: StressSurfaceInfo):
    """Return the closest actual triangle point and its barycentric coordinates."""
    origin = surface_info.face_origin[i_f]
    edges = surface_info.face_edges[i_f]
    local = surface_info.face_dual[i_f] @ (point - origin)
    bary = qd.Vector([1.0 - local.sum(), local[0], local[1]])
    position = origin + local @ edges
    if bary[0] < 0.0 or bary[1] < 0.0 or bary[2] < 0.0:
        nearest = gs.qd_float(float("inf"))
        vertices = qd.Matrix.rows([origin, origin + edges[0, :], origin + edges[1, :]])
        for a in range(3):
            b = (a + 1) % 3
            edge = vertices[b, :] - vertices[a, :]
            fraction = qd.min(1.0, qd.max(0.0, edge.dot(point - vertices[a, :]) / edge.dot(edge)))
            candidate = vertices[a, :] + fraction * edge
            squared = (candidate - point).dot(candidate - point)
            if squared < nearest:
                nearest = squared
                position = candidate
                bary = qd.Vector.zero(gs.qd_float, 3)
                bary[a], bary[b] = 1.0 - fraction, fraction
    return position, bary
