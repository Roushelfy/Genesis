"""Resolve witnessed reference-boundary sources without changing a rigid wrench."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom
from genesis.utils.array_class import V_VEC, DataKind, V

from .contact import StressContactState
from .data import StressInfo
from .paths import func_surface_segment
from .surface import StressSurfaceInfo, func_quadratic_edge_closest, func_surface_geometry


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
    method: qd.Tensor
    vertex: qd.Tensor
    has_vertex_ambiguity: qd.Tensor
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
        method=V(dtype=gs.qd_int, shape=shape),
        vertex=V(dtype=gs.qd_int, shape=shape),
        has_vertex_ambiguity=V(dtype=gs.qd_bool, shape=shape),
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
    sources.vertex.fill(-1)
    sources.witness_face.fill(-1)
    sources.uncertainty.fill(0)
    sources.free_moment.fill(0)
    return sources


@qd.kernel
def kernel_capture_rigid_sources(
    i_l: int,
    contact_state: StressContactState,
    source_state: StressSourceState,
    collider_state: array_class.ColliderState,
):
    func_capture_rigid_sources(i_l, contact_state, source_state, collider_state)


@qd.func
def func_capture_rigid_sources(
    i_l: int,
    contact_state: StressContactState,
    source_state: StressSourceState,
    collider_state: array_class.ColliderState,
):
    """Capture both geom sides in the existing association's material frame.

    The caller declares source uncertainty separately; a producer method is not
    a certificate of material-boundary accuracy. Enabled contact moment modes
    still require their actual torque, as enforced by the stress feature.
    """
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b]:
            source_state.geom[i_c, i_b] = -1
            source_state.pair[i_c, i_b] = -1
            source_state.method[i_c, i_b] = array_class.ContactWitnessKind.NORMAL_OFFSET
            source_state.vertex[i_c, i_b] = -1
            source_state.has_vertex_ambiguity[i_c, i_b] = False
            source_state.witness_face[i_c, i_b] = -1
            source_state.witness_barycentric[i_c, i_b] = qd.Vector([float("nan")] * 3)
            source_state.witness[i_c, i_b] = qd.Vector([float("nan")] * 3)
            source_state.free_moment[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
            i_contact = collider_state.contact_sort_idx[i_c, i_b]
            i_a = collider_state.contact_data.link_a[i_contact, i_b]
            i_b_link = collider_state.contact_data.link_b[i_contact, i_b]
            if i_l == i_a or i_l == i_b_link:
                i_geom = collider_state.contact_data.geom_b[i_contact, i_b]
                point = collider_state.contact_data.witness.point_b[i_contact, i_b]
                i_vertex = collider_state.contact_data.witness.vertex_b[i_contact, i_b]
                if i_l == i_a:
                    i_geom = collider_state.contact_data.geom_a[i_contact, i_b]
                    point = collider_state.contact_data.witness.point_a[i_contact, i_b]
                    i_vertex = collider_state.contact_data.witness.vertex_a[i_contact, i_b]
                source_state.geom[i_c, i_b] = i_geom
                source_state.pair[i_c, i_b] = collider_state.contact_data.pair_idx[i_contact, i_b]
                source_state.method[i_c, i_b] = collider_state.contact_data.witness.method[i_contact, i_b]
                source_state.vertex[i_c, i_b] = i_vertex
                source_state.witness[i_c, i_b] = geom.qd_inv_transform_by_quat(
                    point - contact_state.frame_position[i_b], contact_state.frame_quaternion[i_b]
                )


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


@qd.func
def func_material_source_point(
    i_f: int,
    point: qd.types.vector(3),
    known_bary: qd.types.vector(3),
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
    is_bound: bool,
):
    """Resolve a bound coordinate or the closest point on a material face."""
    position, bary = qd.Vector.zero(gs.qd_float, 3), known_bary
    if is_bound:
        position = surface_info.face_origin[i_f] + bary[1:3] @ surface_info.face_edges[i_f]
    else:
        position, bary = func_closest_triangle(i_f, point, surface_info)
    if stress_info.volume_points.shape[0] > 0:
        is_regular = True
        for i_iteration in range(qd.select(is_bound, 1, 13)):
            if is_regular:
                position, tangent, area = func_surface_geometry(i_f, bary, stress_info)
                gram = tangent.transpose() @ tangent
                is_regular = gram.determinant() > 0 and gram.norm() < float("inf")
                if is_regular and not is_bound and i_iteration < 12:
                    update = gram.inverse() @ (tangent.transpose() @ (position - point))
                    bary += qd.Vector([update.sum(), -update[0], -update[1]])
        distance = gs.qd_float(float("inf"))
        if is_regular and bary.min() >= -64 * epsilon and bary.max() <= 1 + 64 * epsilon:
            distance = (position - point).norm_sqr()
        for i_edge in range(qd.select(is_bound, 0, 3)):
            a, b = i_edge, (i_edge + 1) % 3
            i_first, i_second = stress_info.surface_nodes[i_f, a], stress_info.surface_nodes[i_f, b]
            i_midpoint = stress_info.surface_nodes[i_f, qd.select(i_edge == 0, 3, qd.select(i_edge == 1, 5, 4))]
            first, midpoint, second = (
                stress_info.vertices[i_first],
                stress_info.vertices[i_midpoint],
                stress_info.vertices[i_second],
            )
            # The quadratic edge lies in its Bezier control hull. Pad its box for evaluation roundoff.
            control = 2 * midpoint - 0.5 * (first + second)
            padding = 128 * epsilon * (qd.abs(first) + qd.abs(midpoint) + qd.abs(second) + qd.abs(point))
            outside = qd.max(
                qd.min(first, control, second) - point - padding,
                point - qd.max(first, control, second) - padding,
                0.0,
            )
            if outside.norm_sqr() <= distance:
                candidate, fraction = func_quadratic_edge_closest(point, first, midpoint, second)
                squared = (candidate - point).norm_sqr()
                if squared < distance:
                    position, distance = candidate, squared
                    bary = qd.Vector.zero(gs.qd_float, 3)
                    bary[a], bary[b] = 1 - fraction, fraction
    return position, bary


@qd.func
def func_source_distance_bound(i_f: int, point: qd.types.vector(3), surface_info: StressSurfaceInfo):
    """Bound squared distance to a face by its immutable enclosing box."""
    outside = qd.max(surface_info.face_bounds_low[i_f] - point, point - surface_info.face_bounds_high[i_f], 0.0)
    return outside.dot(outside)


@qd.func
def func_same_surface_feature(
    first_face: int,
    second_face: int,
    position: qd.types.vector(3),
    tolerance: float,
    stress_info: StressInfo,
):
    """Compare shared vertex/edge identity using the existing position budget.

    Fixed dimensionless barycentric thresholds depend on triangle conditioning
    and misclassify roundoff near a real edge. Coincident separately indexed
    sheets remain distinct: distance is checked only on their common topology.
    """
    count = 0
    first_corner, second_corner = -1, -1
    first, second = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    for a in qd.static(range(3)):
        vertex = stress_info.surface_nodes[first_face, a]
        shared = False
        for b in qd.static(range(3)):
            shared = shared or vertex == stress_info.surface_nodes[second_face, b]
        if shared:
            if count == 0:
                first = stress_info.vertices[vertex]
                first_corner = a
            elif count == 1:
                second = stress_info.vertices[vertex]
                second_corner = a
            count += 1
    equivalent = count == 3
    if count == 1:
        equivalent = (position - first).norm() <= tolerance
    elif count == 2:
        edge = second - first
        coordinate = qd.min(1.0, qd.max(0.0, (position - first).dot(edge) / edge.dot(edge)))
        closest = first + coordinate * edge
        if stress_info.volume_points.shape[0] > 0:
            i_edge = qd.select(first_corner == 0, qd.select(second_corner == 1, 3, 4), 5)
            i_midpoint = stress_info.surface_nodes[first_face, i_edge]
            closest, coordinate = func_quadratic_edge_closest(position, first, stress_info.vertices[i_midpoint], second)
        equivalent = (position - closest).norm() <= tolerance
    return equivalent


@qd.kernel
def kernel_resolve_sources(
    projection_limit: float,
    source_epsilon: float,
    contact_state: StressContactState,
    source_state: StressSourceState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
):
    func_resolve_sources(projection_limit, source_epsilon, contact_state, source_state, stress_info, surface_info)


@qd.func
def func_resolve_sources(
    projection_limit: float,
    source_epsilon: float,
    contact_state: StressContactState,
    source_state: StressSourceState,
    stress_info: StressInfo,
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
            # The witness was transformed from world coordinates. Its local
            # magnitude alone misses subtraction/rotation roundoff when the
            # body is far from the world origin; standalone sources have a
            # zero frame position and retain their existing geometric budget.
            operand_scale = qd.max(witness.norm(), projection_limit) + contact_state.frame_position[i_b].norm()
            tolerance = 32.0 * source_epsilon * operand_scale
            valid = source_state.geom[i_c, i_b] >= 0 and source_state.pair[i_c, i_b] >= 0
            valid = valid and projection_limit > 0.0 and uncertainty >= 0.0 and source_epsilon > 0.0
            finite = witness.sum() + projection_limit + uncertainty + source_epsilon
            finite += source_state.free_moment[i_c, i_b].sum()
            finite += contact_state.position[i_c, i_b].sum() + contact_state.force[i_c, i_b].sum()
            valid = valid and qd.abs(finite) < qd.math.inf
            nearest_face = source_state.witness_face[i_c, i_b]
            bound_face = nearest_face
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
            if valid:
                # One shared query body serves nearest-point and ambiguity
                # passes, so curved projection is compiled only once.
                for i_pass in range(2):
                    for i_candidate in range(qd.select(bound_face >= 0, 1, surface_info.face_origin.shape[0])):
                        i_f = qd.select(bound_face >= 0, bound_face, i_candidate)
                        reach = distance + tolerance + qd.select(i_pass == 1, 2 * uncertainty, 0.0)
                        is_query = i_pass == 0
                        if i_pass == 1:
                            is_query = bound_face < 0 and nearest_face >= 0 and distance <= projection_limit
                        if bound_face < 0:
                            is_query = (
                                is_query and func_source_distance_bound(i_f, witness, surface_info) <= reach * reach
                            )
                        if is_query:
                            candidate, candidate_bary = func_material_source_point(
                                i_f,
                                witness,
                                source_state.witness_barycentric[i_c, i_b],
                                stress_info,
                                surface_info,
                                source_epsilon,
                                bound_face >= 0,
                            )
                            candidate_distance = (candidate - witness).norm()
                            separation = (candidate - position).norm()
                            is_equivalent = False
                            is_comparable = nearest_face >= 0 and separation <= tolerance
                            if i_pass == 0:
                                is_comparable = (
                                    is_comparable
                                    and candidate_distance < distance
                                    and (distance - candidate_distance <= tolerance)
                                )
                            if is_comparable:
                                is_equivalent = func_same_surface_feature(
                                    nearest_face, i_f, position, tolerance, stress_info
                                )
                            if i_pass == 0:
                                # Equivalent shared features retain their canonical first face.
                                if candidate_distance < distance and not is_equivalent:
                                    distance, nearest_face = candidate_distance, i_f
                                    position, bary = candidate, candidate_bary
                            elif candidate_distance <= reach:
                                if separation > tolerance and separation <= 2 * uncertainty + tolerance:
                                    is_equivalent = func_surface_segment(
                                        nearest_face, i_f, position, candidate, stress_info, surface_info
                                    )
                                if not is_equivalent:
                                    source_state.status[i_c, i_b] = StressSourceStatus.AMBIGUOUS
            if source_state.has_vertex_ambiguity[i_c, i_b]:
                source_state.status[i_c, i_b] = StressSourceStatus.AMBIGUOUS
            elif not valid:
                source_state.status[i_c, i_b] = StressSourceStatus.INVALID_INPUT
            elif nearest_face < 0 or distance > projection_limit:
                source_state.status[i_c, i_b] = StressSourceStatus.UNRESOLVED
            else:
                if source_state.status[i_c, i_b] == StressSourceStatus.SUCCESS:
                    contact_state.center[i_c, i_b] = position
                    contact_state.anchor_face[i_c, i_b] = nearest_face
                    source_state.barycentric[i_c, i_b] = bary
                    source_state.anchor_moment[i_c, i_b] = source_state.free_moment[i_c, i_b]
                    source_state.anchor_moment[i_c, i_b] += (contact_state.position[i_c, i_b] - position).cross(
                        contact_state.force[i_c, i_b]
                    )
