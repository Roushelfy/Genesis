"""Current material-boundary geometry for grouped physical contact prediction."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.abd.forward_dynamics import func_vel_at_point
from genesis.utils import array_class, geom
from genesis.utils.array_class import DataKind, V, V_VEC
from genesis.utils.misc import qd_to_numpy

from .contact import StressContactState
from .provenance import StressSourceState
from .surface import StressSurfaceInfo


class PairGeometryStatus(IntEnum):
    SUCCESS = 0
    BACK_FACING = 1
    OCCLUDED = 2
    AMBIGUOUS = 3
    UNSUPPORTED = 4


@dataclass(frozen=True)
class StressPairIndex:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    opponent_link: qd.Tensor
    leader: qd.Tensor
    group: qd.Tensor
    env_count: qd.Tensor
    env_start: qd.Tensor
    invalid: qd.Tensor
    count: qd.Tensor


@dataclass(frozen=True)
class StressPairState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    environment: qd.Tensor
    opponent_link: qd.Tensor
    source_count: qd.Tensor
    force: qd.Tensor
    moment: qd.Tensor


@dataclass(frozen=True)
class StressPairSamples:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    opponent_position: qd.Tensor
    normal: qd.Tensor
    relative_velocity: qd.Tensor
    gap: qd.Tensor
    projected_area: qd.Tensor
    opponent_geom: qd.Tensor
    opponent_face: qd.Tensor
    status: qd.Tensor


class StressPairGeometry:
    """Retain complete pair wrenches and query actual opposing surfaces at current poses.

    These are geometric candidates, not admitted pressure support. No collision
    penetration is interpreted as physical indentation, and no radius is assigned.
    Groups use link pairs, so collision decomposition cannot duplicate support.
    Separate support islands can coexist in one pair. The pressure model must
    retain their coupling and separately enforce physical and load acceptance.
    Distance selection follows material-facing and visibility checks, including
    opposing-link occlusion. Mesh queries examine each boundary triangle.
    """

    def __init__(self, contacts: StressContactState):
        n_envs = contacts.valid.shape[1]
        self.index = StressPairIndex(
            opponent_link=V(dtype=gs.qd_int, shape=contacts.valid.shape),
            leader=V(dtype=gs.qd_int, shape=contacts.valid.shape),
            group=V(dtype=gs.qd_int, shape=contacts.valid.shape),
            env_count=V(dtype=gs.qd_int, shape=(n_envs,)),
            env_start=V(dtype=gs.qd_int, shape=(n_envs,)),
            invalid=V(dtype=gs.qd_bool, shape=(n_envs,)),
            count=V(dtype=gs.qd_int, shape=()),
        )
        self.pairs: StressPairState | None = None
        self.samples: StressPairSamples | None = None

    def update(
        self,
        link: int,
        contacts: StressContactState,
        sources: StressSourceState,
        surface: StressSurfaceInfo,
        dyn_info: array_class.DynInfo,
        dyn_state: array_class.DynState,
        collider: array_class.ColliderState,
        epsilon: float,
    ) -> None:
        kernel_pair_indices(link, collider, contacts, sources, self.index, dyn_info)
        count = int(qd_to_numpy(self.index.count, transpose=True))
        if not count:
            self.pairs = None
            self.samples = None
            return
        if (
            self.pairs is None
            or self.pairs.environment.shape[0] != count
            or self.samples.gap.shape[0] != surface.positions.shape[0]
        ):
            self.pairs = StressPairState(
                environment=V(dtype=gs.qd_int, shape=(count,)),
                opponent_link=V(dtype=gs.qd_int, shape=(count,)),
                source_count=V(dtype=gs.qd_int, shape=(count,)),
                force=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
                moment=V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            )
            shape = (surface.positions.shape[0], count)
            self.samples = StressPairSamples(
                opponent_position=V_VEC(3, dtype=gs.qd_float, shape=shape),
                normal=V_VEC(3, dtype=gs.qd_float, shape=shape),
                relative_velocity=V_VEC(3, dtype=gs.qd_float, shape=shape),
                gap=V(dtype=gs.qd_float, shape=shape),
                projected_area=V(dtype=gs.qd_float, shape=shape),
                opponent_geom=V(dtype=gs.qd_int, shape=shape),
                opponent_face=V(dtype=gs.qd_int, shape=shape),
                status=V(dtype=gs.qd_int, shape=shape),
            )
        kernel_pair_wrenches(contacts, sources, self.index, self.pairs)
        kernel_pair_samples(link, dyn_state, contacts, self.pairs, self.samples, dyn_info, surface, epsilon)


@qd.kernel
def kernel_pair_indices(
    link: int,
    collider_state: array_class.ColliderState,
    contact_state: StressContactState,
    source_state: StressSourceState,
    pair_index: StressPairIndex,
    dyn_info: array_class.DynInfo,
):
    for i_b in range(pair_index.env_count.shape[0]):
        pair_index.env_count[i_b] = 0
        pair_index.invalid[i_b] = False
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        pair_index.opponent_link[i_c, i_b] = -1
        pair_index.leader[i_c, i_b] = -1
        pair_index.group[i_c, i_b] = -1
        if contact_state.valid[i_c, i_b]:
            admitted = i_c < collider_state.n_contacts[i_b]
            if admitted:
                i_contact = collider_state.contact_sort_idx[i_c, i_b]
                own, other = -1, -1
                if collider_state.contact_data.link_a[i_contact, i_b] == link:
                    own = collider_state.contact_data.geom_a[i_contact, i_b]
                    other = collider_state.contact_data.geom_b[i_contact, i_b]
                elif collider_state.contact_data.link_b[i_contact, i_b] == link:
                    own = collider_state.contact_data.geom_b[i_contact, i_b]
                    other = collider_state.contact_data.geom_a[i_contact, i_b]
                admitted = own >= 0 and other >= 0 and source_state.geom[i_c, i_b] == own
                admitted = admitted and source_state.pair[i_c, i_b] >= 0
                admitted = (
                    admitted and source_state.pair[i_c, i_b] == collider_state.contact_data.pair_idx[i_contact, i_b]
                )
                for a in qd.static(range(3)):
                    admitted = admitted and not qd.math.isnan(contact_state.force[i_c, i_b][a])
                    admitted = admitted and not qd.math.isinf(contact_state.force[i_c, i_b][a])
                    admitted = admitted and not qd.math.isnan(contact_state.position[i_c, i_b][a])
                    admitted = admitted and not qd.math.isinf(contact_state.position[i_c, i_b][a])
                    admitted = admitted and not qd.math.isnan(source_state.free_moment[i_c, i_b][a])
                    admitted = admitted and not qd.math.isinf(source_state.free_moment[i_c, i_b][a])
                if admitted:
                    pair_index.opponent_link[i_c, i_b] = dyn_info.geoms.link_idx[other]
            if not admitted:
                pair_index.invalid[i_b] = True
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if pair_index.opponent_link[i_c, i_b] >= 0 and not pair_index.invalid[i_b]:
            leader = i_c
            for previous in range(i_c):
                if pair_index.opponent_link[previous, i_b] == pair_index.opponent_link[i_c, i_b]:
                    leader = qd.min(leader, previous)
            pair_index.leader[i_c, i_b] = leader
            if leader == i_c:
                qd.atomic_add(pair_index.env_count[i_b], 1)
    for _ in range(1):
        total = 0
        for i_b in range(pair_index.env_count.shape[0]):
            pair_index.env_start[i_b] = total
            total += pair_index.env_count[i_b]
        pair_index.count[None] = total
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        leader = pair_index.leader[i_c, i_b]
        if leader >= 0:
            group = pair_index.env_start[i_b]
            for previous in range(leader):
                if pair_index.leader[previous, i_b] == previous:
                    group += 1
            pair_index.group[i_c, i_b] = group


@qd.kernel
def kernel_pair_wrenches(
    contact_state: StressContactState,
    source_state: StressSourceState,
    pair_index: StressPairIndex,
    pair_state: StressPairState,
):
    for i_p in range(pair_state.environment.shape[0]):
        pair_state.force[i_p] = qd.Vector.zero(gs.qd_float, 3)
        pair_state.moment[i_p] = qd.Vector.zero(gs.qd_float, 3)
        pair_state.source_count[i_p] = 0
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        i_p = pair_index.group[i_c, i_b]
        if i_p >= 0:
            if pair_index.leader[i_c, i_b] == i_c:
                pair_state.environment[i_p] = i_b
                pair_state.opponent_link[i_p] = pair_index.opponent_link[i_c, i_b]
            force = contact_state.force[i_c, i_b]
            moment = contact_state.position[i_c, i_b].cross(force) + source_state.free_moment[i_c, i_b]
            for a in qd.static(range(3)):
                qd.atomic_add(pair_state.force[i_p][a], force[a])
                qd.atomic_add(pair_state.moment[i_p][a], moment[a])
            qd.atomic_add(pair_state.source_count[i_p], 1)


@qd.func
def func_closest_face(
    point: qd.types.vector(3), first: qd.types.vector(3), second: qd.types.vector(3), third: qd.types.vector(3)
):
    edge_a, edge_b = second - first, third - first
    gram = qd.Matrix([[edge_a.dot(edge_a), edge_a.dot(edge_b)], [edge_a.dot(edge_b), edge_b.dot(edge_b)]])
    local = gram.inverse() @ qd.Vector([edge_a.dot(point - first), edge_b.dot(point - first)])
    closest = first + local[0] * edge_a + local[1] * edge_b
    if local[0] < 0 or local[1] < 0 or local.sum() > 1:
        distance = gs.qd_float(float("inf"))
        vertices = qd.Matrix.rows([first, second, third])
        for a in qd.static(range(3)):
            b = (a + 1) % 3
            edge = vertices[b, :] - vertices[a, :]
            weight = qd.min(1.0, qd.max(0.0, (point - vertices[a, :]).dot(edge) / edge.dot(edge)))
            candidate = vertices[a, :] + weight * edge
            squared = (point - candidate).norm_sqr()
            if squared < distance:
                distance, closest = squared, candidate
    return closest


@qd.func
def func_opponent_candidate(
    i_g: int,
    i_b: int,
    i_f: int,
    point: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    epsilon: float,
):
    local = geom.qd_inv_transform_by_trans_quat(point, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b])
    kind = dyn_info.geoms.type[i_g]
    closest, normal = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    gap, face = gs.qd_float(float("nan")), -1
    status = gs.qd_int(PairGeometryStatus.SUCCESS)
    if kind == gs.GEOM_TYPE.PLANE:
        normal = qd.Vector([dyn_info.geoms.data[i_g][a] for a in qd.static(range(3))]).normalized()
        gap = local.dot(normal)
        closest = local - gap * normal
    elif kind == gs.GEOM_TYPE.SPHERE or kind == gs.GEOM_TYPE.CAPSULE:
        center = qd.Vector.zero(gs.qd_float, 3)
        if kind == gs.GEOM_TYPE.CAPSULE:
            half = 0.5 * dyn_info.geoms.data[i_g][1]
            center[2] = qd.min(half, qd.max(-half, local[2]))
        radial = local - center
        radius = dyn_info.geoms.data[i_g][0]
        if radial.norm() > 128.0 * epsilon * radius:
            normal = radial.normalized()
            gap = radial.norm() - radius
            closest = center + radius * normal
        else:
            status = PairGeometryStatus.AMBIGUOUS
    elif kind == gs.GEOM_TYPE.CYLINDER:
        radius, half = dyn_info.geoms.data[i_g][0], 0.5 * dyn_info.geoms.data[i_g][1]
        radial = qd.Vector([local[0], local[1], 0.0])
        rho = radial.norm()
        tolerance = 128.0 * epsilon * qd.max(radius, half)
        closest = local
        closest[2] = qd.min(half, qd.max(-half, local[2]))
        if rho > radius:
            closest[0], closest[1] = local[0] * radius / rho, local[1] * radius / rho
        delta = local - closest
        if delta.norm() > tolerance:
            normal, gap = delta.normalized(), delta.norm()
        else:
            side_gap, cap_gap = radius - rho, half - qd.abs(local[2])
            if qd.abs(side_gap - cap_gap) <= tolerance:
                status = PairGeometryStatus.AMBIGUOUS
            elif cap_gap < side_gap and qd.abs(local[2]) <= tolerance:
                status = PairGeometryStatus.AMBIGUOUS
            elif cap_gap < side_gap:
                normal[2] = qd.select(local[2] >= 0.0, 1.0, -1.0)
                gap = -cap_gap
                closest = local - gap * normal
            elif rho > tolerance:
                normal = radial / rho
                gap = -side_gap
                closest = local - gap * normal
            else:
                status = PairGeometryStatus.AMBIGUOUS
    elif kind == gs.GEOM_TYPE.BOX:
        half = 0.5 * qd.Vector([dyn_info.geoms.data[i_g][a] for a in qd.static(range(3))])
        closest = qd.min(half, qd.max(-half, local))
        delta = local - closest
        tolerance = 128.0 * epsilon * half.norm()
        if delta.norm() > tolerance:
            normal = delta.normalized()
            gap = delta.norm()
        else:
            distances = half - qd.abs(local)
            axis = 0
            for a in qd.static(range(1, 3)):
                if distances[a] < distances[axis]:
                    axis = a
            count = 0
            for a in qd.static(range(3)):
                count += gs.qd_int(qd.abs(distances[a] - distances[axis]) <= tolerance)
            if count == 1 and qd.abs(local[axis]) > tolerance:
                normal[axis] = qd.select(local[axis] >= 0.0, 1.0, -1.0)
                gap = -distances[axis]
                closest = local - gap * normal
            else:
                status = PairGeometryStatus.AMBIGUOUS
    elif kind == gs.GEOM_TYPE.MESH:
        indices = dyn_info.faces.verts_idx[i_f]
        i_a, i_c, i_d = indices[0], indices[1], indices[2]
        first, second, third = dyn_info.verts.init_pos[i_a], dyn_info.verts.init_pos[i_c], dyn_info.verts.init_pos[i_d]
        cross = (second - first).cross(third - first)
        if cross.norm() > 0.0:
            closest = func_closest_face(local, first, second, third)
            delta = local - closest
            distance = delta.norm()
            normal = cross.normalized()
            scale = (dyn_state.geoms.aabb_max[i_g, i_b] - dyn_state.geoms.aabb_min[i_g, i_b]).norm()
            if distance > 128.0 * epsilon * scale:
                normal = qd.select(delta.dot(normal) >= 0.0, 1.0, -1.0) * delta / distance
            gap, face = delta.dot(normal), i_f
        else:
            status = PairGeometryStatus.UNSUPPORTED
    else:
        status = PairGeometryStatus.UNSUPPORTED
    world_point = geom.qd_transform_by_trans_quat(
        closest, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
    )
    world_normal = geom.qd_transform_by_quat(normal, dyn_state.geoms.quat[i_g, i_b])
    return world_point, world_normal, gap, face, status


@qd.func
def func_opponent_boundary(
    i_g: int,
    i_b: int,
    point: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    epsilon: float,
):
    closest, normal = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
    minimum, gap, face = gs.qd_float(float("inf")), gs.qd_float(float("nan")), -1
    status = gs.qd_int(PairGeometryStatus.UNSUPPORTED)
    count = 1
    is_mesh = dyn_info.geoms.type[i_g] == gs.GEOM_TYPE.MESH
    if is_mesh:
        count = dyn_info.geoms.face_end[i_g] - dyn_info.geoms.face_start[i_g]
    for index in range(count):
        i_f = -1
        if is_mesh:
            i_f = dyn_info.geoms.face_start[i_g] + index
        candidate, direction, distance, candidate_face, flag = func_opponent_candidate(
            i_g, i_b, i_f, point, dyn_state, dyn_info, epsilon
        )
        if not is_mesh:
            closest, normal, gap, face, status = candidate, direction, distance, candidate_face, flag
        elif flag == PairGeometryStatus.SUCCESS:
            size = (dyn_state.geoms.aabb_max[i_g, i_b] - dyn_state.geoms.aabb_min[i_g, i_b]).norm()
            tolerance = 128.0 * epsilon * size
            if qd.abs(distance) < minimum - tolerance:
                minimum, closest, normal, gap, face, status = (
                    qd.abs(distance),
                    candidate,
                    direction,
                    distance,
                    candidate_face,
                    flag,
                )
            elif qd.abs(qd.abs(distance) - minimum) <= tolerance:
                if (candidate - closest).norm() > tolerance or (direction - normal).norm() > 128.0 * epsilon:
                    status = PairGeometryStatus.AMBIGUOUS
    return closest, normal, gap, face, status


@qd.func
def func_material_occluded(
    own_face: int,
    point: qd.types.vector(3),
    target: qd.types.vector(3),
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    segment = target - point
    blocked = False
    for i_f in range(surface_info.face_normal.shape[0]):
        if i_f != own_face:
            normal = surface_info.face_normal[i_f]
            denominator = segment.dot(normal)
            if qd.abs(denominator) > 128.0 * epsilon * segment.norm():
                fraction = (surface_info.face_origin[i_f] - point).dot(normal) / denominator
                if fraction > 128.0 * epsilon and fraction < 1.0 - 128.0 * epsilon:
                    intersection = point + fraction * segment
                    uv = surface_info.face_dual[i_f] @ (intersection - surface_info.face_origin[i_f])
                    if uv[0] >= -128.0 * epsilon and uv[1] >= -128.0 * epsilon and uv.sum() <= 1.0 + 128.0 * epsilon:
                        blocked = True
    return blocked


@qd.func
def func_opponent_occluded(
    link: int,
    i_b: int,
    point: qd.types.vector(3),
    target: qd.types.vector(3),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    epsilon: float,
):
    """Test the open segment against the opposing link's collision geometry."""
    blocked = False
    for i_g in range(dyn_info.geoms.type.shape[0]):
        if (
            not blocked
            and dyn_info.geoms.link_idx[i_g] == link
            and (dyn_info.geoms.contype[i_g] != 0 or dyn_info.geoms.conaffinity[i_g] != 0)
        ):
            local = geom.qd_inv_transform_by_trans_quat(
                point, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
            )
            end = geom.qd_inv_transform_by_trans_quat(
                target, dyn_state.geoms.pos[i_g, i_b], dyn_state.geoms.quat[i_g, i_b]
            )
            segment = end - local
            kind = dyn_info.geoms.type[i_g]
            scale = segment.norm()
            if kind != gs.GEOM_TYPE.PLANE:
                scale = qd.max(scale, (dyn_state.geoms.aabb_max[i_g, i_b] - dyn_state.geoms.aabb_min[i_g, i_b]).norm())
            tolerance = 128.0 * epsilon * scale
            lower, upper = gs.qd_float(128.0 * epsilon), gs.qd_float(1.0 - 128.0 * epsilon)
            if kind == gs.GEOM_TYPE.PLANE:
                normal = qd.Vector([dyn_info.geoms.data[i_g][a] for a in qd.static(range(3))]).normalized()
                blocked = qd.min(local.dot(normal), end.dot(normal)) < -tolerance
            elif kind == gs.GEOM_TYPE.BOX:
                half = 0.5 * qd.Vector([dyn_info.geoms.data[i_g][a] for a in qd.static(range(3))])
                for a in qd.static(range(3)):
                    if qd.abs(segment[a]) > tolerance:
                        first = (-half[a] + tolerance - local[a]) / segment[a]
                        last = (half[a] - tolerance - local[a]) / segment[a]
                        lower, upper = qd.max(lower, qd.min(first, last)), qd.min(upper, qd.max(first, last))
                    elif qd.abs(local[a]) >= half[a] - tolerance:
                        upper = -1.0
                blocked = lower < upper
            elif kind == gs.GEOM_TYPE.SPHERE or kind == gs.GEOM_TYPE.CAPSULE or kind == gs.GEOM_TYPE.CYLINDER:
                radius = dyn_info.geoms.data[i_g][0] - tolerance
                half = gs.qd_float(0.0)
                if kind != gs.GEOM_TYPE.SPHERE:
                    half = 0.5 * dyn_info.geoms.data[i_g][1]
                    if qd.abs(segment[2]) > tolerance:
                        first = (-half + tolerance - local[2]) / segment[2]
                        last = (half - tolerance - local[2]) / segment[2]
                        lower, upper = qd.max(lower, qd.min(first, last)), qd.min(upper, qd.max(first, last))
                    elif qd.abs(local[2]) >= half - tolerance:
                        upper = -1.0
                    if lower < upper:
                        square = segment[0] ** 2 + segment[1] ** 2
                        fraction = lower
                        if square > 0.0:
                            fraction = qd.max(
                                lower, qd.min(upper, -(local[0] * segment[0] + local[1] * segment[1]) / square)
                            )
                        radial = local + fraction * segment
                        blocked = radial[0] ** 2 + radial[1] ** 2 < radius**2
                if kind != gs.GEOM_TYPE.CYLINDER and segment.norm_sqr() > 0.0:
                    count = 1
                    if kind == gs.GEOM_TYPE.CAPSULE:
                        count = 2
                    for cap in range(count):
                        center = qd.Vector([0.0, 0.0, qd.select(cap == 0, -half, half)])
                        fraction = qd.max(
                            128.0 * epsilon,
                            qd.min(1.0 - 128.0 * epsilon, -(local - center).dot(segment) / segment.norm_sqr()),
                        )
                        blocked = blocked or (local + fraction * segment - center).norm_sqr() < radius**2
            elif kind == gs.GEOM_TYPE.MESH:
                for i_f in range(dyn_info.geoms.face_start[i_g], dyn_info.geoms.face_end[i_g]):
                    indices = dyn_info.faces.verts_idx[i_f]
                    i_a, i_c, i_d = indices[0], indices[1], indices[2]
                    first, second, third = (
                        dyn_info.verts.init_pos[i_a],
                        dyn_info.verts.init_pos[i_c],
                        dyn_info.verts.init_pos[i_d],
                    )
                    edge_a, edge_b = second - first, third - first
                    cross = edge_a.cross(edge_b)
                    denominator = segment.dot(cross)
                    if qd.abs(denominator) > 128.0 * epsilon * segment.norm() * cross.norm():
                        fraction = (first - local).dot(cross) / denominator
                        if fraction > lower and fraction < upper:
                            relative = local + fraction * segment - first
                            gram = qd.Matrix(
                                [[edge_a.dot(edge_a), edge_a.dot(edge_b)], [edge_a.dot(edge_b), edge_b.dot(edge_b)]]
                            )
                            uv = gram.inverse() @ qd.Vector([edge_a.dot(relative), edge_b.dot(relative)])
                            if (
                                uv[0] >= -128.0 * epsilon
                                and uv[1] >= -128.0 * epsilon
                                and uv.sum() <= 1.0 + 128.0 * epsilon
                            ):
                                blocked = True
    return blocked


@qd.kernel
def kernel_pair_samples(
    link: int,
    dyn_state: array_class.DynState,
    contact_state: StressContactState,
    pair_state: StressPairState,
    samples: StressPairSamples,
    dyn_info: array_class.DynInfo,
    surface_info: StressSurfaceInfo,
    epsilon: float,
):
    for i_s, i_p in qd.ndrange(surface_info.positions.shape[0], pair_state.environment.shape[0]):
        i_b = pair_state.environment[i_p]
        other_link = pair_state.opponent_link[i_p]
        own_face = i_s // surface_info.shape.shape[0]
        local = surface_info.positions[i_s]
        quaternion = contact_state.frame_quaternion[i_b]
        point = geom.qd_transform_by_trans_quat(local, contact_state.frame_position[i_b], quaternion)
        target, normal = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        gap, face, selected_geom = gs.qd_float(float("inf")), -1, -1
        status = gs.qd_int(PairGeometryStatus.UNSUPPORTED)
        inside_count, component_count = 0, 0
        found, unresolved = False, False
        material_normal = geom.qd_transform_by_quat(surface_info.face_normal[own_face], quaternion)
        for opponent in range(dyn_info.geoms.type.shape[0]):
            if dyn_info.geoms.link_idx[opponent] == other_link and (
                dyn_info.geoms.contype[opponent] != 0 or dyn_info.geoms.conaffinity[opponent] != 0
            ):
                component_count += 1
                is_mesh = dyn_info.geoms.type[opponent] == gs.GEOM_TYPE.MESH
                count = 1
                if is_mesh:
                    count = dyn_info.geoms.face_end[opponent] - dyn_info.geoms.face_start[opponent]
                size = surface_info.face_edges[own_face][0, :].norm()
                if dyn_info.geoms.type[opponent] != gs.GEOM_TYPE.PLANE:
                    size = qd.max(
                        size, (dyn_state.geoms.aabb_max[opponent, i_b] - dyn_state.geoms.aabb_min[opponent, i_b]).norm()
                    )
                nearest = gs.qd_float(float("inf"))
                nearest_gap = gs.qd_float(float("nan"))
                nearest_found, sign_ambiguous = False, False
                # The first pass establishes the material side. The second
                # selects an accessible candidate on that side, never a mesh's
                # hidden back face interpreted as negative penetration.
                for query in range(2 * count):
                    candidate_face = -1
                    if is_mesh:
                        candidate_face = dyn_info.geoms.face_start[opponent] + query % count
                    candidate, direction, distance, candidate_face, candidate_status = func_opponent_candidate(
                        opponent, i_b, candidate_face, point, dyn_state, dyn_info, epsilon
                    )
                    tolerance = 128.0 * epsilon * qd.max(size, qd.abs(distance))
                    if query < count:
                        if candidate_status == PairGeometryStatus.SUCCESS:
                            if qd.abs(distance) < nearest - tolerance:
                                nearest, nearest_gap = qd.abs(distance), distance
                                nearest_found, sign_ambiguous = True, False
                            elif qd.abs(qd.abs(distance) - nearest) <= tolerance:
                                sign_ambiguous = sign_ambiguous or distance * nearest_gap < -(tolerance**2)
                        elif not is_mesh:
                            unresolved = True
                    elif nearest_found and not sign_ambiguous and candidate_status == PairGeometryStatus.SUCCESS:
                        same_side = (nearest_gap >= 0.0 and distance >= -tolerance) or (
                            nearest_gap < 0.0 and distance <= tolerance
                        )
                        if same_side:
                            projection = -direction.dot(material_normal)
                            if projection <= 128.0 * epsilon:
                                candidate_status = PairGeometryStatus.BACK_FACING
                            elif distance > 0.0 and (not found or qd.abs(distance) <= qd.abs(gap) + tolerance):
                                candidate_local = geom.qd_inv_transform_by_trans_quat(
                                    candidate, contact_state.frame_position[i_b], quaternion
                                )
                                if func_material_occluded(own_face, local, candidate_local, surface_info, epsilon):
                                    candidate_status = PairGeometryStatus.OCCLUDED
                                elif func_opponent_occluded(
                                    other_link, i_b, point, candidate, dyn_state, dyn_info, epsilon
                                ):
                                    candidate_status = PairGeometryStatus.OCCLUDED
                            eligible = candidate_status == PairGeometryStatus.SUCCESS
                            replace = not found and qd.abs(distance) < qd.abs(gap) - tolerance
                            replace = replace or (
                                eligible and (not found or qd.abs(distance) < qd.abs(gap) - tolerance)
                            )
                            if replace:
                                target, normal, gap, face, status = (
                                    candidate,
                                    direction,
                                    distance,
                                    candidate_face,
                                    candidate_status,
                                )
                                selected_geom = opponent
                                found = eligible
                            elif eligible and found and qd.abs(qd.abs(distance) - qd.abs(gap)) <= tolerance:
                                if (candidate - target).norm() > tolerance or (
                                    direction - normal
                                ).norm() > 128.0 * epsilon:
                                    status = PairGeometryStatus.AMBIGUOUS
                if not nearest_found or sign_ambiguous:
                    unresolved = True
                elif nearest_gap < 0.0:
                    inside_count += 1
        if unresolved:
            status = PairGeometryStatus.UNSUPPORTED
        if inside_count and component_count > 1:
            # Component interiors can hide union-boundary intersections that
            # are not the closest point of any original component face.
            status = PairGeometryStatus.AMBIGUOUS
        normal = geom.qd_inv_transform_by_quat(normal, quaternion)
        target_local = geom.qd_inv_transform_by_trans_quat(target, contact_state.frame_position[i_b], quaternion)
        projection = -normal.dot(surface_info.face_normal[own_face])
        samples.opponent_position[i_s, i_p] = target_local
        samples.normal[i_s, i_p] = normal
        samples.gap[i_s, i_p] = gap
        samples.opponent_face[i_s, i_p] = face
        samples.opponent_geom[i_s, i_p] = selected_geom
        samples.status[i_s, i_p] = status
        samples.projected_area[i_s, i_p] = qd.select(
            status == PairGeometryStatus.SUCCESS, surface_info.weights[i_s] * projection, 0.0
        )
        velocity = func_vel_at_point(other_link, i_b, target, dyn_state.links) - func_vel_at_point(
            link, i_b, point, dyn_state.links
        )
        samples.relative_velocity[i_s, i_p] = geom.qd_inv_transform_by_quat(velocity, quaternion)
