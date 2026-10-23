"""Bind a collision mesh vertex to a uniquely matched reference boundary feature."""

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils import geom
from genesis.utils.array_class import V_VEC, DataKind, V

from .contact import StressContactState
from .data import StressInfo
from .provenance import StressSourceState

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


class StressSourceVertexStatus(IntEnum):
    SUCCESS = 0
    UNMAPPED = 1
    AMBIGUOUS = 2


@dataclass(frozen=True)
class StressSourceVertexInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    geom: qd.Tensor
    positions: qd.Tensor
    reference_vertex: qd.Tensor
    reference_face: qd.Tensor
    barycentric: qd.Tensor
    offset: qd.Tensor
    status: qd.Tensor
    search_min: qd.Tensor
    search_max: qd.Tensor


class StressSourceVertices:
    """Validate declared mesh/reference correspondence, then retain its feature identity.

    The geometry tolerance is a declared bound on corresponding coordinates,
    independent of contact radius and final wrench/residual/stress budgets.
    Runtime binding requires the actual witness to equal a collision vertex
    of the same geom to floating-point roundoff. Nearby general points retain
    the ordinary bounded reference-surface resolver.
    """

    def __init__(self, link: "RigidLink", stress_info: StressInfo, geometry_tolerance: float):
        if not isfinite(geometry_tolerance) or geometry_tolerance <= 0.0:
            gs.raise_exception("Mesh/reference vertex correspondence requires a positive geometry tolerance.")
        geoms = tuple(geometry for geometry in link.geoms if geometry.type == gs.GEOM_TYPE.MESH)
        n_vertices = sum(geometry.n_verts for geometry in geoms)
        self.info: StressSourceVertexInfo | None = None
        if not n_vertices:
            return
        self.info = StressSourceVertexInfo(
            geom=V(dtype=gs.qd_int, shape=(n_vertices,)),
            positions=V_VEC(3, dtype=gs.qd_float, shape=(n_vertices,)),
            reference_vertex=V(dtype=gs.qd_int, shape=(n_vertices,)),
            reference_face=V(dtype=gs.qd_int, shape=(n_vertices,)),
            barycentric=V_VEC(3, dtype=gs.qd_float, shape=(n_vertices,)),
            offset=V(dtype=gs.qd_float, shape=(n_vertices,)),
            status=V(dtype=gs.qd_int, shape=(n_vertices,)),
            search_min=V_VEC(3, dtype=gs.qd_float, shape=((n_vertices + 63) // 64,)),
            search_max=V_VEC(3, dtype=gs.qd_float, shape=((n_vertices + 63) // 64,)),
        )
        start = 0
        for geometry in geoms:
            kernel_source_vertex_positions(
                start,
                geometry.idx,
                link.idx,
                geometry.init_verts,
                geometry.init_pos,
                geometry.init_quat,
                link.entity.solver._links_offset_pos,
                link.entity.solver._links_offset_quat,
                self.info,
                link.entity.solver._links_offset_quat.ndim == 3,
            )
            start += geometry.n_verts
        kernel_source_vertex_mapping(stress_info, self.info, geometry_tolerance)
        kernel_source_vertex_bounds(self.info)


@qd.kernel
def kernel_source_vertex_positions(
    i_start: int,
    i_geom: int,
    i_link: int,
    vertices: qd.types.ndarray(),
    position: qd.types.ndarray(),
    quaternion: qd.types.ndarray(),
    links_offset_pos: qd.types.ndarray(),
    links_offset_quat: qd.types.ndarray(),
    source_vertex_info: StressSourceVertexInfo,
    batch_offsets: qd.template(),
):
    for i_v in range(vertices.shape[0]):
        point = qd.Vector.zero(gs.qd_float, 3)
        origin = qd.Vector.zero(gs.qd_float, 3)
        rotation = qd.Vector.zero(gs.qd_float, 4)
        offset_pos = qd.Vector.zero(gs.qd_float, 3)
        offset_quat = qd.Vector.zero(gs.qd_float, 4)
        for a in qd.static(range(3)):
            point[a], origin[a] = vertices[i_v, a], position[a]
            if qd.static(batch_offsets):
                offset_pos[a] = links_offset_pos[0, i_link, a]
            else:
                offset_pos[a] = links_offset_pos[i_link, a]
        for a in qd.static(range(4)):
            rotation[a] = quaternion[a]
            if qd.static(batch_offsets):
                offset_quat[a] = links_offset_quat[0, i_link, a]
            else:
                offset_quat[a] = links_offset_quat[i_link, a]
        source_vertex_info.geom[i_start + i_v] = i_geom
        point = origin + geom.qd_transform_by_quat(point, rotation)
        source_vertex_info.positions[i_start + i_v] = offset_pos + geom.qd_transform_by_quat(point, offset_quat)


@qd.kernel
def kernel_source_vertex_mapping(
    stress_info: StressInfo, source_vertex_info: StressSourceVertexInfo, geometry_tolerance: float
):
    for i_v in range(source_vertex_info.positions.shape[0]):
        point = source_vertex_info.positions[i_v]
        nearest, face, corner = -1, -1, -1
        distance = gs.qd_float(float("inf"))
        for i_f in range(stress_info.surface_nodes.shape[0]):
            for a in qd.static(range(6)):
                i_node = stress_info.surface_nodes[i_f, a]
                candidate = (stress_info.vertices[i_node] - point).norm()
                if candidate < distance:
                    distance, nearest, face, corner = candidate, i_node, i_f, a
        source_vertex_info.status[i_v] = StressSourceVertexStatus.UNMAPPED
        source_vertex_info.reference_vertex[i_v] = -1
        source_vertex_info.reference_face[i_v] = -1
        source_vertex_info.barycentric[i_v] = qd.Vector([float("nan")] * 3)
        source_vertex_info.offset[i_v] = distance
        if distance <= geometry_tolerance:
            is_ambiguous = False
            for i_f in range(stress_info.surface_nodes.shape[0]):
                for a in qd.static(range(6)):
                    i_node = stress_info.surface_nodes[i_f, a]
                    if i_node != nearest and (stress_info.vertices[i_node] - point).norm() <= geometry_tolerance:
                        is_ambiguous = True
            if is_ambiguous:
                source_vertex_info.status[i_v] = StressSourceVertexStatus.AMBIGUOUS
            else:
                barycentric = qd.Vector.zero(gs.qd_float, 3)
                if corner < 3:
                    barycentric[corner] = 1.0
                else:
                    first = 0 if corner < 5 else 1
                    second = 1 if corner == 3 else 2
                    barycentric[first], barycentric[second] = 0.5, 0.5
                source_vertex_info.status[i_v] = StressSourceVertexStatus.SUCCESS
                source_vertex_info.reference_vertex[i_v] = nearest
                source_vertex_info.reference_face[i_v] = face
                source_vertex_info.barycentric[i_v] = barycentric


@qd.kernel
def kernel_source_vertex_bounds(source_vertex_info: StressSourceVertexInfo):
    for i_block in range(source_vertex_info.search_min.shape[0]):
        lower = qd.Vector([float("inf")] * 3)
        upper = qd.Vector([-float("inf")] * 3)
        for i_v in range(64 * i_block, qd.min(64 * (i_block + 1), source_vertex_info.positions.shape[0])):
            lower = qd.min(lower, source_vertex_info.positions[i_v])
            upper = qd.max(upper, source_vertex_info.positions[i_v])
        source_vertex_info.search_min[i_block], source_vertex_info.search_max[i_block] = lower, upper


@qd.kernel
def kernel_bind_source_vertices(
    contact_state: StressContactState,
    source_state: StressSourceState,
    source_vertex_info: StressSourceVertexInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        source_state.has_vertex_ambiguity[i_c, i_b] = False
        if contact_state.valid[i_c, i_b]:
            point = source_state.witness[i_c, i_b]
            tolerance = 64.0 * epsilon * (point.norm() + contact_state.frame_position[i_b].norm())
            nearest = -1
            reference_vertex = -1
            is_ambiguous = False
            for i_block in range(source_vertex_info.search_min.shape[0]):
                lower = qd.max(
                    source_vertex_info.search_min[i_block] - point,
                    qd.max(point - source_vertex_info.search_max[i_block], 0.0),
                )
                if lower.norm_sqr() <= tolerance * tolerance:
                    for i_v in range(64 * i_block, qd.min(64 * (i_block + 1), source_vertex_info.positions.shape[0])):
                        if (
                            source_vertex_info.geom[i_v] == source_state.geom[i_c, i_b]
                            and (point - source_vertex_info.positions[i_v]).norm() <= tolerance
                        ):
                            status = source_vertex_info.status[i_v]
                            is_ambiguous = is_ambiguous or status == StressSourceVertexStatus.AMBIGUOUS
                            if status == StressSourceVertexStatus.SUCCESS:
                                i_reference = source_vertex_info.reference_vertex[i_v]
                                is_ambiguous = is_ambiguous or (nearest >= 0 and i_reference != reference_vertex)
                                nearest, reference_vertex = i_v, i_reference
            if is_ambiguous:
                source_state.has_vertex_ambiguity[i_c, i_b] = True
            elif nearest >= 0:
                source_state.witness_face[i_c, i_b] = source_vertex_info.reference_face[nearest]
                source_state.witness_barycentric[i_c, i_b] = source_vertex_info.barycentric[nearest]
