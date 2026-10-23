"""Current local contact minima and complete region wrenches for Hertz estimates."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils import array_class, geom
from genesis.utils.array_class import DataKind, V, V_MAT, V_VEC

from .contact import StressContactState
from .curvature import CurvatureStatus, StressCurvatureInfo, func_surface_curvature, func_closest_surface
from .normal_roots import func_normal_root
from .pair_geometry import PairGeometryStatus, func_opponent_boundary
from .provenance import StressSourceState


class HertzRegionStatus(IntEnum):
    SUCCESS = 0
    UNLOADED = 1
    GEOMETRY = 2
    CURVATURE = 3
    ITERATIONS = 4
    SOURCE = 5


@dataclass(frozen=True)
class StressHertzRegionState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    center: qd.Tensor
    normal: qd.Tensor
    tangent: qd.Tensor
    curvature: qd.Tensor
    gap: qd.Tensor
    normal_residual: qd.Tensor
    geometry_kind: qd.Tensor
    opponent_link: qd.Tensor
    opponent_type: qd.Tensor
    leader: qd.Tensor
    status: qd.Tensor
    force: qd.Tensor
    moment: qd.Tensor
    source_count: qd.Tensor
    is_valid: qd.Tensor
    # Rectangle half-planes a*x + b*y <= c use the region's tangent/bitangent frame at contact evaluation time.
    boundary_planes: qd.Tensor
    n_boundaries: qd.Tensor


class StressHertzRegions:
    """Find local geometric minima before combining solved contact forces.

    Planar opponents use direct inversion of the fixed normal field. Other
    pairs use six bounded two-dimensional Newton updates with signed rest
    curvature. Converged roots on the same link pair are coalesced by
    their geometric resolution, independently of source force or Hertz radius.
    Separate minima remain separate regions. All original forces and moments
    are retained; a pressure law still must represent their spatial wrench.

    Smooth-normal mesh roots are explicitly approximate geometry. These inputs
    do not certify contact pressure, a continuous union boundary, or friction
    history. Unresolved sources invalidate their environment's region result.
    An opposing box uses the face most opposed to the initial material normal;
    the face stays fixed during the local solve. This prevents rigid overlap
    from selecting a nearer side face. Edge-clipped roots remain unresolved.
    """

    def __init__(self, contacts: StressContactState):
        shape = contacts.valid.shape
        self.state = StressHertzRegionState(
            center=V_VEC(3, dtype=gs.qd_float, shape=shape),
            normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
            tangent=V_VEC(3, dtype=gs.qd_float, shape=shape),
            curvature=V_MAT(2, 2, dtype=gs.qd_float, shape=shape),
            gap=V(dtype=gs.qd_float, shape=shape),
            normal_residual=V(dtype=gs.qd_float, shape=shape),
            geometry_kind=V(dtype=gs.qd_int, shape=shape),
            opponent_link=V(dtype=gs.qd_int, shape=shape),
            opponent_type=V(dtype=gs.qd_int, shape=shape),
            leader=V(dtype=gs.qd_int, shape=shape),
            status=V(dtype=gs.qd_int, shape=shape),
            force=V_VEC(3, dtype=gs.qd_float, shape=shape),
            moment=V_VEC(3, dtype=gs.qd_float, shape=shape),
            source_count=V(dtype=gs.qd_int, shape=shape),
            is_valid=V(dtype=gs.qd_bool, shape=(shape[1],)),
            boundary_planes=V_MAT(4, 3, dtype=gs.qd_float, shape=shape),
            n_boundaries=V(dtype=gs.qd_int, shape=shape),
        )

    def update(self, link, contacts, sources, dyn_state, collider, dyn_info, curvature_info, epsilon):
        kernel_hertz_roots(link, dyn_state, collider, contacts, sources, self.state, dyn_info, curvature_info, epsilon)
        kernel_hertz_planar(dyn_state, collider, contacts, sources, self.state, dyn_info, curvature_info, epsilon)
        kernel_hertz_regions(contacts, sources, self.state, collider, epsilon)


@qd.kernel
def kernel_hertz_roots(
    i_l: int,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    region_state: StressHertzRegionState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        region_state.boundary_planes[i_c, i_b] = qd.Matrix.zero(gs.qd_float, 4, 3)
        region_state.n_boundaries[i_c, i_b] = 0
        status = gs.qd_int(HertzRegionStatus.UNLOADED)
        center, normal, tangent = (
            qd.Vector.zero(gs.qd_float, 3),
            qd.Vector.zero(gs.qd_float, 3),
            qd.Vector.zero(gs.qd_float, 3),
        )
        curvature = qd.Matrix.zero(gs.qd_float, 2, 2)
        gap, residual, geometry_kind, i_opponent = gs.qd_float(0.0), gs.qd_float(0.0), 0, -1
        opponent_type = -1
        is_finite = True
        for i in qd.static(range(3)):
            force = contact_state.force[i_c, i_b][i]
            position = contact_state.position[i_c, i_b][i]
            witness = source_state.witness[i_c, i_b][i]
            moment = source_state.free_moment[i_c, i_b][i]
            is_finite = is_finite and not qd.math.isnan(force) and not qd.math.isinf(force)
            is_finite = is_finite and not qd.math.isnan(position) and not qd.math.isinf(position)
            is_finite = is_finite and not qd.math.isnan(witness) and not qd.math.isinf(witness)
            is_finite = is_finite and not qd.math.isnan(moment) and not qd.math.isinf(moment)
        if contact_state.valid[i_c, i_b] and not is_finite:
            status = HertzRegionStatus.SOURCE
        if contact_state.valid[i_c, i_b] and is_finite and contact_state.force[i_c, i_b].norm() > 0.0:
            status = HertzRegionStatus.SOURCE
            i_contact = collider_state.contact_sort_idx[i_c, i_b]
            i_ga = collider_state.contact_data.geom_a[i_contact, i_b]
            i_gb = collider_state.contact_data.geom_b[i_contact, i_b]
            if collider_state.contact_data.link_b[i_contact, i_b] == i_l:
                i_ga, i_gb = i_gb, i_ga
            is_valid = i_ga >= 0 and i_gb >= 0
            is_valid = is_valid and i_ga < dyn_info.geoms.link_idx.shape[0] and i_gb < dyn_info.geoms.link_idx.shape[0]
            is_valid = is_valid and source_state.geom[i_c, i_b] == i_ga and source_state.pair[i_c, i_b] >= 0
            is_valid = is_valid and source_state.pair[i_c, i_b] == collider_state.contact_data.pair_idx[i_contact, i_b]
            is_valid = is_valid and (
                collider_state.contact_data.link_a[i_contact, i_b] == i_l
                or collider_state.contact_data.link_b[i_contact, i_b] == i_l
            )
            if is_valid:
                i_opponent = dyn_info.geoms.link_idx[i_gb]
                opponent_type = dyn_info.geoms.type[i_gb]
                point = geom.qd_transform_by_trans_quat(
                    source_state.witness[i_c, i_b],
                    contact_state.frame_position[i_b],
                    contact_state.frame_quaternion[i_b],
                )
                status = HertzRegionStatus.ITERATIONS
                is_planar = (
                    dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.MESH or dyn_info.geoms.type[i_ga] == gs.GEOM_TYPE.SPHERE
                ) and (dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.PLANE or dyn_info.geoms.type[i_gb] == gs.GEOM_TYPE.BOX)
                i_box_face = -1
                for i_iteration in range(qd.select(is_planar, 0, 6)):
                    if status == HertzRegionStatus.ITERATIONS:
                        positions = qd.Matrix.zero(gs.qd_float, 2, 3)
                        normals = qd.Matrix.zero(gs.qd_float, 2, 3)
                        operators = qd.Matrix.zero(gs.qd_float, 6, 3)
                        for i_side in range(2):
                            i_g = i_ga
                            if i_side == 1:
                                i_g = i_gb
                            query = point
                            if i_side == 1 and dyn_info.geoms.type[i_g] == gs.GEOM_TYPE.BOX:
                                if i_box_face < 0:
                                    opposed = geom.qd_inv_transform_by_quat(
                                        -normals[0, :], dyn_state.geoms.quat[i_g, i_b]
                                    )
                                    i_axis = 0
                                    for i in qd.static(range(1, 3)):
                                        if qd.abs(opposed[i]) > qd.abs(opposed[i_axis]):
                                            i_axis = i
                                    i_box_face = 2 * i_axis + gs.qd_int(opposed[i_axis] < 0.0)
                                query = func_box_face_point(i_g, i_b, i_box_face, point, dyn_state, dyn_info)
                            candidate, boundary_normal, boundary_gap, i_f, boundary_status = func_opponent_boundary(
                                i_g, i_b, query, dyn_state, dyn_info, epsilon
                            )
                            direction, operator, curvature_status = func_surface_curvature(
                                i_g, i_f, i_b, candidate, dyn_state, dyn_info, curvature_info, epsilon
                            )
                            positions[i_side, :], normals[i_side, :] = candidate, direction
                            for i, j in qd.static(qd.ndrange(3, 3)):
                                operators[3 * i_side + i, j] = operator[i, j]
                            geometry_kind = qd.max(geometry_kind, curvature_status)
                            if (
                                boundary_status == PairGeometryStatus.UNSUPPORTED
                                or curvature_status >= CurvatureStatus.AMBIGUOUS
                            ):
                                status = HertzRegionStatus.GEOMETRY
                        first, second = normals[0, :], normals[1, :]
                        if first.dot(second) >= -0.5:
                            status = HertzRegionStatus.GEOMETRY
                        if status == HertzRegionStatus.ITERATIONS:
                            normal = (first - second).normalized()
                            i_axis = 0
                            for i in qd.static(range(1, 3)):
                                if qd.abs(normal[i]) < qd.abs(normal[i_axis]):
                                    i_axis = i
                            unit = qd.Vector.zero(gs.qd_float, 3)
                            unit[i_axis] = 1.0
                            tangent = normal.cross(unit).normalized()
                            frame = qd.Matrix.rows([tangent, normal.cross(tangent)])
                            relative = qd.Matrix.zero(gs.qd_float, 3, 3)
                            for i, j in qd.static(qd.ndrange(3, 3)):
                                relative[i, j] = operators[i, j] + operators[3 + i, j]
                            curvature = frame @ relative @ frame.transpose()
                            gradient = frame @ (first + second) / first.dot(normal)
                            residual = gradient.norm()
                            center = positions[0, :]
                            gap = (positions[1, :] - center).dot(normal)
                            if (
                                curvature.trace() <= 0.0
                                or curvature.determinant() <= 128.0 * epsilon * curvature.trace() ** 2
                            ):
                                status = HertzRegionStatus.CURVATURE
                            elif residual < 1e-7:
                                status = HertzRegionStatus.SUCCESS
                            else:
                                step = -(curvature.inverse() @ gradient)
                                # Bound the angular change of the local quadratic approximation.
                                fraction = qd.min(1.0, 0.25 / qd.max(step.norm() * curvature.norm(), 1e-30))
                                point = 0.5 * (positions[0, :] + positions[1, :]) + fraction * (
                                    frame.transpose() @ step
                                )
        region_state.center[i_c, i_b], region_state.normal[i_c, i_b], region_state.tangent[i_c, i_b] = (
            center,
            normal,
            tangent,
        )
        region_state.curvature[i_c, i_b], region_state.gap[i_c, i_b] = curvature, gap
        region_state.normal_residual[i_c, i_b], region_state.geometry_kind[i_c, i_b] = residual, geometry_kind
        region_state.opponent_link[i_c, i_b], region_state.status[i_c, i_b] = i_opponent, status
        region_state.opponent_type[i_c, i_b] = opponent_type


@qd.kernel
def kernel_hertz_planar(
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    region_state: StressHertzRegionState,
    dyn_info: array_class.DynInfo,
    curvature_info: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if region_state.status[i_c, i_b] == HertzRegionStatus.ITERATIONS:
            i_own = source_state.geom[i_c, i_b]
            i_contact = collider_state.contact_sort_idx[i_c, i_b]
            i_other = collider_state.contact_data.geom_a[i_contact, i_b]
            if i_other == i_own:
                i_other = collider_state.contact_data.geom_b[i_contact, i_b]
            own_kind, other_kind = dyn_info.geoms.type[i_own], dyn_info.geoms.type[i_other]
            is_planar = (own_kind == gs.GEOM_TYPE.MESH or own_kind == gs.GEOM_TYPE.SPHERE) and (
                other_kind == gs.GEOM_TYPE.PLANE or other_kind == gs.GEOM_TYPE.BOX
            )
            if is_planar:
                seed = geom.qd_transform_by_trans_quat(
                    source_state.witness[i_c, i_b],
                    contact_state.frame_position[i_b],
                    contact_state.frame_quaternion[i_b],
                )
                local_normal = qd.Vector([dyn_info.geoms.data[i_other][i] for i in qd.static(range(3))]).normalized()
                plane_point = dyn_state.geoms.pos[i_other, i_b]
                i_box_axis = -1
                if other_kind == gs.GEOM_TYPE.BOX:
                    position, i_f = seed, -1
                    if own_kind == gs.GEOM_TYPE.MESH:
                        position, i_f = func_closest_surface(
                            i_own, i_b, seed, dyn_state, dyn_info, curvature_info, epsilon
                        )
                    else:
                        position, boundary_normal, boundary_gap, i_f, boundary_status = func_opponent_boundary(
                            i_own, i_b, seed, dyn_state, dyn_info, epsilon
                        )
                    own_normal, own_tensor, curvature_status = func_surface_curvature(
                        i_own, i_f, i_b, position, dyn_state, dyn_info, curvature_info, epsilon
                    )
                    opposed = geom.qd_inv_transform_by_quat(-own_normal, dyn_state.geoms.quat[i_other, i_b])
                    i_box_axis = 0
                    for i in qd.static(range(1, 3)):
                        if qd.abs(opposed[i]) > qd.abs(opposed[i_box_axis]):
                            i_box_axis = i
                    local_normal = qd.Vector.zero(gs.qd_float, 3)
                    local_normal[i_box_axis] = qd.select(opposed[i_box_axis] >= 0.0, 1.0, -1.0)
                    plane_point += geom.qd_transform_by_quat(
                        0.5 * dyn_info.geoms.data[i_other][i_box_axis] * local_normal,
                        dyn_state.geoms.quat[i_other, i_b],
                    )
                normal = -geom.qd_transform_by_quat(local_normal, dyn_state.geoms.quat[i_other, i_b])
                center, operator, i_face, residual, geometry_kind = func_normal_root(
                    i_own, i_b, normal, seed, dyn_state, dyn_info, curvature_info, epsilon
                )
                status = gs.qd_int(HertzRegionStatus.SUCCESS)
                if geometry_kind >= CurvatureStatus.AMBIGUOUS:
                    status = HertzRegionStatus.GEOMETRY
                elif residual >= 1e-7:
                    status = HertzRegionStatus.ITERATIONS
                i_axis = 0
                for i in qd.static(range(1, 3)):
                    if qd.abs(normal[i]) < qd.abs(normal[i_axis]):
                        i_axis = i
                unit = qd.Vector.zero(gs.qd_float, 3)
                unit[i_axis] = 1.0
                tangent = normal.cross(unit).normalized()
                frame = qd.Matrix.rows([tangent, normal.cross(tangent)])
                curvature = frame @ operator @ frame.transpose()
                if status == HertzRegionStatus.SUCCESS:
                    if curvature.trace() <= 0.0 or curvature.determinant() <= 128.0 * epsilon * curvature.trace() ** 2:
                        status = HertzRegionStatus.CURVATURE
                    if i_box_axis >= 0:
                        local_center = geom.qd_inv_transform_by_trans_quat(
                            center, dyn_state.geoms.pos[i_other, i_b], dyn_state.geoms.quat[i_other, i_b]
                        )
                        half = 0.5 * qd.Vector([dyn_info.geoms.data[i_other][i] for i in qd.static(range(3))])
                        local_frame = frame @ geom.qd_quat_to_R(dyn_state.geoms.quat[i_other, i_b], epsilon)
                        i_boundary = 0
                        for i in qd.static(range(3)):
                            if i != i_box_axis:
                                for j in qd.static(range(2)):
                                    sign = 1.0 - 2.0 * j
                                    region_state.boundary_planes[i_c, i_b][i_boundary, 0] = sign * local_frame[0, i]
                                    region_state.boundary_planes[i_c, i_b][i_boundary, 1] = sign * local_frame[1, i]
                                    region_state.boundary_planes[i_c, i_b][i_boundary, 2] = (
                                        half[i] - sign * local_center[i]
                                    )
                                    i_boundary += 1
                                if qd.abs(local_center[i]) >= half[i] - 128.0 * epsilon * half.norm():
                                    status = HertzRegionStatus.GEOMETRY
                                    geometry_kind = CurvatureStatus.AMBIGUOUS
                        region_state.n_boundaries[i_c, i_b] = i_boundary
                region_state.center[i_c, i_b] = center
                region_state.normal[i_c, i_b] = normal
                region_state.tangent[i_c, i_b] = tangent
                region_state.curvature[i_c, i_b] = curvature
                region_state.gap[i_c, i_b] = (plane_point - center).dot(normal)
                region_state.normal_residual[i_c, i_b] = residual
                region_state.geometry_kind[i_c, i_b] = geometry_kind
                region_state.status[i_c, i_b] = status


@qd.func
def func_box_face_point(
    i_g: int,
    i_b: int,
    i_face: int,
    position: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
):
    local = geom.qd_inv_transform_by_trans_quat(position, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b])
    half = 0.5 * qd.Vector([dyn_info.geoms.data[i_g][i] for i in qd.static(range(3))])
    local = qd.min(half, qd.max(-half, local))
    i_axis = i_face // 2
    local[i_axis] = qd.select(i_face % 2 == 0, 1.0, -1.0) * half[i_axis]
    return geom.qd_transform_by_trans_quat(local, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b])


@qd.kernel
def kernel_hertz_regions(
    contact_state: StressContactState,
    source_state: StressSourceState,
    region_state: StressHertzRegionState,
    collider_state: array_class.ColliderState,
    epsilon: float,
):
    for i_b in range(region_state.is_valid.shape[0]):
        region_state.is_valid[i_b] = True
    # Every slot must clear on release, while only current contacts need the
    # serial transitive grouping and deterministic wrench accumulation.
    for i_c, i_b in qd.ndrange(region_state.status.shape[0], region_state.status.shape[1]):
        region_state.force[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        region_state.moment[i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        region_state.source_count[i_c, i_b] = 0
        region_state.leader[i_c, i_b] = -1
    for i_b in range(region_state.is_valid.shape[0]):
        count = qd.min(collider_state.n_contacts[i_b], region_state.status.shape[0])
        for i_c in range(count):
            status = region_state.status[i_c, i_b]
            if status != HertzRegionStatus.SUCCESS and status != HertzRegionStatus.UNLOADED:
                region_state.is_valid[i_b] = False
            if status == HertzRegionStatus.SUCCESS:
                i_leader = i_c
                curvature = region_state.curvature[i_c, i_b]
                resolution = 1e-6 * curvature.trace() / curvature.determinant()
                center = region_state.center[i_c, i_b]
                resolution += 128.0 * epsilon * center.norm()
                for j_c in range(i_c):
                    if region_state.status[j_c, i_b] == HertzRegionStatus.SUCCESS:
                        if region_state.opponent_link[j_c, i_b] == region_state.opponent_link[i_c, i_b]:
                            if (region_state.center[j_c, i_b] - center).norm() <= resolution:
                                i_leader = qd.min(i_leader, region_state.leader[j_c, i_b])
                region_state.leader[i_c, i_b] = i_leader
        for i_c in range(count):
            i_leader = region_state.leader[i_c, i_b]
            if i_leader >= 0 and region_state.is_valid[i_b]:
                rotation = contact_state.frame_quaternion[i_b]
                force = geom.qd_transform_by_quat(contact_state.force[i_c, i_b], rotation)
                position = geom.qd_transform_by_trans_quat(
                    contact_state.position[i_c, i_b], contact_state.frame_position[i_b], rotation
                )
                moment = (position - region_state.center[i_leader, i_b]).cross(force)
                moment += geom.qd_transform_by_quat(source_state.free_moment[i_c, i_b], rotation)
                region_state.force[i_leader, i_b] += force
                region_state.moment[i_leader, i_b] += moment
                region_state.source_count[i_leader, i_b] += 1
