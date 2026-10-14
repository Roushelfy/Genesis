"""QD triangle-ray clipping to the physical Euclidean ball before intrinsic path acceptance.

Experimental chart storage reuses the existing typed shared-support anchor/radius.
No physical input, topology, profile law or original-point wrench changes.
"""

import quadrants as qd
from genesis.engine.solvers.rigid.stress import footprint, traction
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.footprint import (
    StressFootprintState,
    StressFootprintStatus,
    func_footprint_point_distance,
)
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo
from genesis.engine.solvers.rigid.stress.provenance import func_closest_triangle
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionInfo, StressTractionState

import genesis as gs

_native_bary = traction.func_surface_fan_bary


@qd.func
def func_fan_cap(i_sample: int, i_c: int, i_b: int, state: StressTractionState, info: StressTractionInfo):
    n = info.gauss.shape[0]
    i_face = i_sample // (3 * n * n)
    i_fan = (i_sample // (n * n)) % 3
    v = info.gauss[i_sample % n] ** state.fan_coordinate_power[None]
    origin, edges = info.face_origin[i_face], info.face_edges[i_face]
    center_bary = state.fan_barycentric[i_face, i_c, i_b]
    edge_bary = qd.Vector.zero(gs.qd_float, 3)
    edge_bary[i_fan], edge_bary[(i_fan + 1) % 3] = 1.0 - v, v
    center = origin + center_bary[1] * edges[0, :] + center_bary[2] * edges[1, :]
    edge = origin + edge_bary[1] * edges[0, :] + edge_bary[2] * edges[1, :]
    direction = edge - center
    delta = center - state.support.center[i_c, i_b]
    radius = state.support.radius[i_c, i_b]
    a, b, c = direction.dot(direction), delta.dot(direction), delta.dot(delta) - radius * radius
    cap = gs.qd_float(0.0)
    if a > 0.0 and c < 0.0:
        root = qd.sqrt(qd.max(0.0, b * b - a * c))
        cap = (-b + root) / a
        if b >= 0.0 and b + root > 0.0:
            cap = -c / (b + root)
        cap = qd.min(1.0, qd.max(0.0, cap))
    return cap


@qd.func
def func_clipped_fan_bary(i_sample: int, i_c: int, i_b: int, state: StressTractionState, info: StressTractionInfo):
    i_face = 0
    bary = qd.Vector.zero(gs.qd_float, 3)
    if i_sample >= 3 * info.positions.shape[0]:
        i_face, bary = _native_bary(i_sample, i_c, i_b, state, info)
    else:
        n = info.gauss.shape[0]
        i_face = i_sample // (3 * n * n)
        i_fan, local = (i_sample // (n * n)) % 3, i_sample % (n * n)
        power = state.fan_coordinate_power[None]
        u, v = info.gauss[local // n] ** power, info.gauss[local % n] ** power
        u *= func_fan_cap(i_sample, i_c, i_b, state, info)
        bary = (1.0 - u) * state.fan_barycentric[i_face, i_c, i_b]
        bary[i_fan] += u * (1.0 - v)
        bary[(i_fan + 1) % 3] += u * v
    return i_face, bary


@qd.kernel
def kernel_clipped_profiles(
    contact_state: StressContactState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
    traction_info: StressTractionInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        traction_state.sample_count[i_c, i_b] = 0
        traction_state.circle.face[i_c, i_b] = -1
        traction_state.circle.radius[i_c, i_b] = 0.0
        traction_state.support.center[i_c, i_b] = contact_state.center[i_c, i_b]
        traction_state.support.radius[i_c, i_b] = contact_state.radius[i_c, i_b]
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            i_face = contact_state.anchor_face[i_c, i_b]
            origin, edges = surface_info.face_origin[i_face], surface_info.face_edges[i_face]
            anchor, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            normal = -surface_info.face_normal[i_face]
            roundoff = 128.0 * epsilon * (anchor.norm() + origin.norm() + edges[0, :].norm() + edges[1, :].norm())
            inside = qd.abs((anchor - origin).dot(normal)) <= roundoff
            vertices = qd.Matrix.rows([origin, origin + edges[0, :], origin + edges[1, :]])
            for a in range(3):
                edge = vertices[(a + 1) % 3, :] - vertices[a, :]
                direction = normal.cross(edge).normalized()
                if direction.dot(vertices[(a + 2) % 3, :] - vertices[a, :]) < 0.0:
                    direction = -direction
                inside = inside and (anchor - vertices[a, :]).dot(direction) >= radius + roundoff
            if inside:
                traction_state.circle.face[i_c, i_b] = i_face
                traction_state.circle.radius[i_c, i_b] = radius
    for i_f, i_c, i_b in qd.ndrange(
        surface_info.face_origin.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        bary = qd.Vector.zero(gs.qd_float, 3)
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            _, bary = func_closest_triangle(i_f, contact_state.center[i_c, i_b], surface_info)
        traction_state.fan_barycentric[i_f, i_c, i_b] = bary
    for i_q, i_c, i_b in qd.ndrange(
        traction_state.profile.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            anchor, radius = contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b]
            n = surface_info.gauss.shape[0]
            i_circle = traction_state.circle.face[i_c, i_b]
            i_sample, is_candidate = i_q, False
            if i_circle >= 0:
                is_candidate = i_q < n * n
                i_sample = 3 * surface_info.positions.shape[0] + i_q
            else:
                i_face = i_q // (3 * n * n)
                center_bary = traction_state.fan_barycentric[i_face, i_c, i_b]
                origin, edges = surface_info.face_origin[i_face], surface_info.face_edges[i_face]
                center = origin + center_bary[1] * edges[0, :] + center_bary[2] * edges[1, :]
                is_candidate = (center - anchor).norm() < radius
            if is_candidate:
                i_face, bary = func_clipped_fan_bary(i_sample, i_c, i_b, traction_state, traction_info)
                origin, edges = surface_info.face_origin[i_face], surface_info.face_edges[i_face]
                position = origin + bary[1] * edges[0, :] + bary[2] * edges[1, :]
                weight, distance = gs.qd_float(0.0), gs.qd_float(float("inf"))
                local = i_q % (n * n)
                u, v = surface_info.gauss[local // n], surface_info.gauss[local % n]
                power = traction_state.fan_coordinate_power[None]
                if i_circle >= 0:
                    qd.atomic_add(footprint_state.candidates[i_c, i_b], 1)
                    distance = radius * u**power
                    weight = 2.0 * qd.math.pi * radius * radius * power * u ** (2 * power - 1)
                else:
                    i_fan = (i_q // (n * n)) % 3
                    center_bary = traction_state.fan_barycentric[i_face, i_c, i_b]
                    area = center_bary[(i_fan + 2) % 3] * edges[0, :].cross(edges[1, :]).norm()
                    if area > 0.0 and (position - anchor).norm() < radius:
                        qd.atomic_add(footprint_state.candidates[i_c, i_b], 1)
                        distance, _ = func_footprint_point_distance(
                            i_face,
                            i_c,
                            i_b,
                            position,
                            contact_state,
                            footprint_state,
                            stress_info,
                            surface_info,
                            path_info,
                        )
                        cap = func_fan_cap(i_sample, i_c, i_b, traction_state, traction_info)
                        weight = area * power * power * u ** (2 * power - 1) * v ** (power - 1) * cap * cap
                if weight > 0.0 and distance < radius:
                    weight *= surface_info.gauss_weights[local // n] * surface_info.gauss_weights[local % n]
                    relative = distance / radius
                    profile = weight * qd.exp(-0.5 * (relative / 0.45) ** 2) * (1.0 - relative * relative) ** 2
                    slot = qd.atomic_add(traction_state.sample_count[i_c, i_b], 1)
                    traction_state.sample_indices[slot, i_c, i_b] = i_sample
                    traction_state.profile[slot, i_c, i_b] = profile
                    qd.atomic_add(footprint_state.samples[i_c, i_b], 1)
                    qd.atomic_max(footprint_state.maximum_distance[i_c, i_b], distance)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if (
            contact_state.valid[i_c, i_b]
            and footprint_state.status[i_c, i_b] == 0
            and footprint_state.samples[i_c, i_b] == 0
        ):
            footprint_state.status[i_c, i_b] = StressFootprintStatus.EMPTY_SUPPORT


def install():
    traction.func_surface_fan_bary = func_clipped_fan_bary
    footprint.func_surface_fan_bary = func_clipped_fan_bary
