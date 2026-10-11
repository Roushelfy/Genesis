"""Shared intrinsic surface paths through material-boundary edge ports."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .data import StressInfo
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class StressPathInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    positions: qd.Tensor
    face_ports: qd.Tensor
    starts: qd.Tensor
    neighbors: qd.Tensor
    lengths: qd.Tensor
    distances: qd.Tensor


class StressSurfacePaths:
    """Share fixed-geometry upper bounds on intrinsic distance across environments.

    Each graph edge is a segment contained in one actual boundary triangle.
    Edge ports use mesh identities, so separate nearby sheets never connect.
    A checked coplanar triangle walk supplies exact straight distances where
    possible. Port paths otherwise overestimate continuous intrinsic distance.
    """

    def __init__(self, stress_info: StressInfo, subdivisions: int):
        if subdivisions < 1:
            gs.raise_exception("Surface edge subdivisions must be positive.")
        triangles = qd_to_numpy(stress_info.surface_nodes, transpose=True)[:3].T
        vertices, vertex_inverse = np.unique(triangles, return_inverse=True)
        edges, edge_inverse = np.unique(
            np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1),
            axis=0,
            return_inverse=True,
        )
        n_inner = subdivisions - 1
        inner_ports = len(vertices) + edge_inverse[:, None] * n_inner + np.arange(n_inner)
        face_ports = np.concatenate(
            (vertex_inverse.reshape((-1, 3)), inner_ports.reshape((len(triangles), -1))), axis=1
        )
        endpoints = np.concatenate((np.repeat(vertices[:, None], 2, axis=1), np.repeat(edges, n_inner, axis=0)))
        fractions = np.concatenate(
            (np.zeros(len(vertices), dtype=gs.np_int), np.tile(np.arange(1, subdivisions), len(edges)))
        )
        pairs = np.stack(np.broadcast_arrays(face_ports[:, :, None], face_ports[:, None, :]), axis=-1).reshape((-1, 2))
        pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
        starts = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=len(endpoints)))]
        n_ports = len(endpoints)
        self.info = StressPathInfo(
            positions=V_VEC(3, dtype=gs.qd_float, shape=(n_ports,)),
            face_ports=V(dtype=gs.qd_int, shape=face_ports.shape),
            starts=V(dtype=gs.qd_int, shape=(n_ports + 1,)),
            neighbors=V(dtype=gs.qd_int, shape=(len(pairs),)),
            lengths=V(dtype=gs.qd_float, shape=(len(pairs),)),
            distances=V(dtype=gs.qd_float, shape=(n_ports, n_ports)),
        )
        self.info.face_ports.from_numpy(face_ports.astype(gs.np_int, copy=False))
        self.info.starts.from_numpy(starts.astype(gs.np_int, copy=False))
        self.info.neighbors.from_numpy(pairs[:, 1].astype(gs.np_int, copy=False))
        visited = V(dtype=gs.qd_bool, shape=(n_ports, n_ports))
        kernel_port_geometry(
            subdivisions,
            endpoints.astype(gs.np_int, copy=False),
            fractions.astype(gs.np_int, copy=False),
            stress_info,
            self.info,
        )
        kernel_port_lengths(self.info)
        kernel_port_distances(visited, self.info)
        qd.sync()


@qd.kernel
def kernel_port_geometry(
    subdivisions: int,
    endpoints: qd.types.ndarray(),
    fractions: qd.types.ndarray(),
    stress_info: StressInfo,
    path_info: StressPathInfo,
):
    for i_p in range(path_info.positions.shape[0]):
        i_first, i_second = endpoints[i_p, 0], endpoints[i_p, 1]
        fraction = gs.qd_float(fractions[i_p]) / subdivisions
        path_info.positions[i_p] = (1.0 - fraction) * stress_info.vertices[i_first] + fraction * stress_info.vertices[
            i_second
        ]


@qd.kernel
def kernel_port_lengths(path_info: StressPathInfo):
    for i_p in range(path_info.positions.shape[0]):
        for i_e in range(path_info.starts[i_p], path_info.starts[i_p + 1]):
            i_neighbor = path_info.neighbors[i_e]
            path_info.lengths[i_e] = (path_info.positions[i_neighbor] - path_info.positions[i_p]).norm()


@qd.kernel
def kernel_port_distances(visited: qd.Tensor, path_info: StressPathInfo):
    for i_source in range(path_info.positions.shape[0]):
        for i_p in range(path_info.positions.shape[0]):
            path_info.distances[i_source, i_p] = float("inf")
            visited[i_source, i_p] = False
        path_info.distances[i_source, i_source] = 0.0
        for _ in range(path_info.positions.shape[0]):
            nearest = -1
            distance = gs.qd_float(float("inf"))
            for i_p in range(path_info.positions.shape[0]):
                if not visited[i_source, i_p] and path_info.distances[i_source, i_p] < distance:
                    nearest = i_p
                    distance = path_info.distances[i_source, i_p]
            if nearest >= 0:
                visited[i_source, nearest] = True
                for i_e in range(path_info.starts[nearest], path_info.starts[nearest + 1]):
                    i_neighbor = path_info.neighbors[i_e]
                    candidate = distance + path_info.lengths[i_e]
                    path_info.distances[i_source, i_neighbor] = qd.min(
                        path_info.distances[i_source, i_neighbor], candidate
                    )


@qd.func
def func_surface_segment(
    i_anchor: int,
    i_target: int,
    anchor: qd.types.vector(3),
    target: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
):
    """Verify that a straight segment reaches its target through actual coplanar triangles."""
    direction = target - anchor
    scale = qd.max(direction.norm(), 1e-30)
    normal = surface_info.face_normal[i_anchor]
    tolerance = gs.qd_float(2e-12)
    if qd.static(gs.qd_float == qd.f32):
        tolerance = 2e-5
    current, parameter = i_anchor, gs.qd_float(0.0)
    active, accepted = True, False
    for _ in range(stress_info.surface_nodes.shape[0]):
        if active:
            origin = surface_info.face_origin[current]
            coplanar = (surface_info.face_normal[current] - normal).norm() <= tolerance
            coplanar = coplanar and qd.abs(normal.dot(anchor - origin)) <= tolerance * scale
            coplanar = coplanar and qd.abs(normal.dot(direction)) <= tolerance * scale
            start = surface_info.face_dual[current] @ (anchor - origin)
            delta = surface_info.face_dual[current] @ direction
            bary = qd.Vector([1.0 - start.sum(), start[0], start[1]])
            derivative = qd.Vector([-delta.sum(), delta[0], delta[1]])
            contained = True
            for a in qd.static(range(3)):
                contained = contained and bary[a] + parameter * derivative[a] >= -tolerance
            if not coplanar or not contained:
                active = False
            else:
                end_contained = True
                for a in qd.static(range(3)):
                    end_contained = end_contained and bary[a] + derivative[a] >= -tolerance
                if end_contained:
                    accepted = current == i_target
                    active = False
                else:
                    crossing = gs.qd_float(float("inf"))
                    edge = -1
                    ties = 0
                    for a in qd.static(range(3)):
                        if derivative[a] < -tolerance:
                            candidate = -bary[a] / derivative[a]
                            if candidate < crossing - tolerance:
                                crossing = candidate
                                edge = (a + 1) % 3
                                ties = 1
                            elif qd.abs(candidate - crossing) <= tolerance:
                                ties += 1
                    if edge < 0 or ties != 1 or crossing <= parameter + tolerance or crossing >= 1.0:
                        active = False
                    else:
                        neighbor = stress_info.surface_neighbors[current, edge]
                        if neighbor < 0:
                            active = False
                        else:
                            current = neighbor
                            parameter = crossing
    return accepted


@qd.func
def func_surface_distance(
    i_anchor: int,
    i_target: int,
    anchor: qd.types.vector(3),
    target: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    straight = func_surface_segment(i_anchor, i_target, anchor, target, stress_info, surface_info)
    distance = gs.qd_float(float("inf"))
    if straight:
        distance = (target - anchor).norm()
    else:
        for i_a in range(path_info.face_ports.shape[1]):
            i_first = path_info.face_ports[i_anchor, i_a]
            first = (path_info.positions[i_first] - anchor).norm()
            for i_t in range(path_info.face_ports.shape[1]):
                i_second = path_info.face_ports[i_target, i_t]
                candidate = first + path_info.distances[i_first, i_second]
                candidate += (target - path_info.positions[i_second]).norm()
                distance = qd.min(distance, candidate)
    return distance, straight
