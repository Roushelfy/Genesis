"""Resolve witnessed reference-boundary sources without changing a rigid wrench."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V

from .contact import StressContactState
from .surface import StressSurfaceInfo


class StressSourceStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    AMBIGUOUS = 2
    UNRESOLVED = 3


@dataclass(frozen=True)
class StressSourceState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    geom: qd.Tensor
    pair: qd.Tensor
    witness: qd.Tensor
    witness_face: qd.Tensor
    witness_barycentric: qd.Tensor
    uncertainty: qd.Tensor
    free_moment: qd.Tensor
    anchor_moment: qd.Tensor
    barycentric: qd.Tensor
    status: qd.Tensor


def create_sources(n_contacts: int, n_envs: int) -> StressSourceState:
    shape = (n_contacts, n_envs)
    sources = StressSourceState(
        geom=V(dtype=gs.qd_int, shape=shape),
        pair=V(dtype=gs.qd_int, shape=shape),
        witness=V_VEC(3, dtype=gs.qd_float, shape=shape),
        witness_face=V(dtype=gs.qd_int, shape=shape),
        witness_barycentric=V_VEC(3, dtype=gs.qd_float, shape=shape),
        uncertainty=V(dtype=gs.qd_float, shape=shape),
        free_moment=V_VEC(3, dtype=gs.qd_float, shape=shape),
        anchor_moment=V_VEC(3, dtype=gs.qd_float, shape=shape),
        barycentric=V_VEC(3, dtype=gs.qd_float, shape=shape),
        status=V(dtype=gs.qd_int, shape=shape),
    )
    sources.geom.fill(-1)
    sources.pair.fill(-1)
    sources.witness_face.fill(-1)
    sources.uncertainty.fill(0)
    sources.free_moment.fill(0)
    return sources


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


@qd.kernel
def kernel_resolve_sources(
    projection_limit: float,
    source_epsilon: float,
    contact_state: StressContactState,
    source_state: StressSourceState,
    surface_info: StressSurfaceInfo,
):
    func_resolve_sources(projection_limit, source_epsilon, contact_state, source_state, surface_info)


@qd.func
def func_resolve_sources(
    projection_limit: float,
    source_epsilon: float,
    contact_state: StressContactState,
    source_state: StressSourceState,
    surface_info: StressSurfaceInfo,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        source_state.status[i_c, i_b] = StressSourceStatus.SUCCESS
        contact_state.center[i_c, i_b] = qd.Vector([float("nan")] * 3)
        contact_state.anchor_face[i_c, i_b] = -1
        source_state.barycentric[i_c, i_b] = qd.Vector([float("nan")] * 3)
        source_state.anchor_moment[i_c, i_b] = qd.Vector([float("nan")] * 3)
        if contact_state.valid[i_c, i_b]:
            witness = source_state.witness[i_c, i_b]
            uncertainty = source_state.uncertainty[i_c, i_b]
            tolerance = 32.0 * source_epsilon * qd.max(witness.norm(), projection_limit)
            valid = source_state.geom[i_c, i_b] >= 0 and source_state.pair[i_c, i_b] >= 0
            valid = valid and projection_limit > 0.0 and uncertainty >= 0.0 and source_epsilon > 0.0
            finite = witness.sum() + projection_limit + uncertainty + source_epsilon
            finite += source_state.free_moment[i_c, i_b].sum()
            finite += contact_state.position[i_c, i_b].sum() + contact_state.force[i_c, i_b].sum()
            valid = valid and qd.abs(finite) < qd.math.inf
            nearest_face = source_state.witness_face[i_c, i_b]
            position = qd.Vector.zero(gs.qd_float, 3)
            bary = qd.Vector.zero(gs.qd_float, 3)
            distance = gs.qd_float(float("inf"))
            if valid:
                if nearest_face >= 0:
                    valid = nearest_face < surface_info.face_origin.shape[0]
                    if valid:
                        bary = source_state.witness_barycentric[i_c, i_b]
                        valid = qd.abs(bary.sum() - 1.0) <= 32.0 * source_epsilon
                        for a in qd.static(range(3)):
                            valid = valid and bary[a] >= 0.0 and qd.abs(bary[a]) < qd.math.inf
                        position = (
                            surface_info.face_origin[nearest_face] + bary[1:3] @ surface_info.face_edges[nearest_face]
                        )
                        distance = (position - witness).norm()
                else:
                    for i_f in range(surface_info.face_origin.shape[0]):
                        candidate, candidate_bary = func_closest_triangle(i_f, witness, surface_info)
                        candidate_distance = (candidate - witness).norm()
                        if candidate_distance < distance:
                            distance = candidate_distance
                            nearest_face = i_f
                            position, bary = candidate, candidate_bary
            if not valid:
                source_state.status[i_c, i_b] = StressSourceStatus.INVALID_INPUT
            elif nearest_face < 0 or distance > projection_limit:
                source_state.status[i_c, i_b] = StressSourceStatus.UNRESOLVED
            else:
                if source_state.witness_face[i_c, i_b] < 0:
                    for i_f in range(surface_info.face_origin.shape[0]):
                        candidate, candidate_bary = func_closest_triangle(i_f, witness, surface_info)
                        # An uncertainty ball must still select one physical anchor.
                        # Adjacent edge/corner faces sharing that anchor are equivalent sources.
                        if (candidate - witness).norm() <= distance + 2.0 * uncertainty + tolerance and (
                            candidate - position
                        ).norm() > tolerance:
                            source_state.status[i_c, i_b] = StressSourceStatus.AMBIGUOUS
                if source_state.status[i_c, i_b] == StressSourceStatus.SUCCESS:
                    contact_state.center[i_c, i_b] = position
                    contact_state.anchor_face[i_c, i_b] = nearest_face
                    source_state.barycentric[i_c, i_b] = bary
                    source_state.anchor_moment[i_c, i_b] = source_state.free_moment[i_c, i_b]
                    source_state.anchor_moment[i_c, i_b] += (contact_state.position[i_c, i_b] - position).cross(
                        contact_state.force[i_c, i_b]
                    )
