"""Native finite profiles reached through actual material-boundary paths."""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, V

from .contact import StressContactState
from .data import StressInfo
from .paths import StressPathInfo, func_surface_segment
from .provenance import StressSourceState, StressSourceStatus
from .surface import StressSurfaceInfo
from .traction import StressTractionState, create_tractions


class StressFootprintStatus(IntEnum):
    SUCCESS = 0
    INVALID_SOURCE = 1
    INVALID_INPUT = 2
    EMPTY_SUPPORT = 3


@dataclass(frozen=True)
class StressFootprintState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    port_distance: qd.Tensor
    status: qd.Tensor
    candidates: qd.Tensor
    samples: qd.Tensor
    maximum_distance: qd.Tensor


def create_footprints(n_ports: int, n_contacts: int, n_envs: int) -> StressFootprintState:
    shape = (n_contacts, n_envs)
    return StressFootprintState(
        port_distance=V(dtype=gs.qd_float, shape=(n_ports, n_contacts, n_envs)),
        status=V(dtype=gs.qd_int, shape=shape),
        candidates=V(dtype=gs.qd_int, shape=shape),
        samples=V(dtype=gs.qd_int, shape=shape),
        maximum_distance=V(dtype=gs.qd_float, shape=shape),
    )


@qd.kernel
def kernel_footprint_ports(
    contact_state: StressContactState,
    source_state: StressSourceState,
    footprint_state: StressFootprintState,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        footprint_state.status[i_c, i_b] = StressFootprintStatus.SUCCESS
        footprint_state.candidates[i_c, i_b] = 0
        footprint_state.samples[i_c, i_b] = 0
        footprint_state.maximum_distance[i_c, i_b] = 0.0
        if contact_state.valid[i_c, i_b]:
            i_anchor = contact_state.anchor_face[i_c, i_b]
            is_resolved = source_state.status[i_c, i_b] == StressSourceStatus.SUCCESS
            is_resolved = is_resolved and 0 <= i_anchor < surface_info.face_origin.shape[0]
            if not is_resolved:
                footprint_state.status[i_c, i_b] = StressFootprintStatus.INVALID_SOURCE
            else:
                radius = contact_state.radius[i_c, i_b]
                anchor = contact_state.center[i_c, i_b]
                is_valid = radius > 0.0 and radius < qd.math.inf and qd.abs(anchor.sum()) < qd.math.inf
                if not is_valid:
                    footprint_state.status[i_c, i_b] = StressFootprintStatus.INVALID_INPUT
    for i_p, i_c, i_b in qd.ndrange(
        path_info.positions.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        distance = gs.qd_float(float("inf"))
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            i_anchor = contact_state.anchor_face[i_c, i_b]
            anchor = contact_state.center[i_c, i_b]
            for i_a in range(path_info.face_ports.shape[1]):
                i_first = path_info.face_ports[i_anchor, i_a]
                candidate = (path_info.positions[i_first] - anchor).norm() + path_info.distances[i_first, i_p]
                distance = qd.min(distance, candidate)
        footprint_state.port_distance[i_p, i_c, i_b] = distance


@qd.func
def func_footprint_distance(
    i_q: int,
    i_c: int,
    i_b: int,
    contact_state: StressContactState,
    footprint_state: StressFootprintState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    """Return a witnessed surface-path upper bound, with a checked planar shortcut."""
    i_face = i_q // surface_info.shape.shape[0]
    target = surface_info.positions[i_q]
    return func_footprint_point_distance(
        i_face, i_c, i_b, target, contact_state, footprint_state, stress_info, surface_info, path_info
    )


@qd.func
def func_footprint_point_distance(
    i_face: int,
    i_c: int,
    i_b: int,
    target: qd.types.vector(3),
    contact_state: StressContactState,
    footprint_state: StressFootprintState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    i_anchor = contact_state.anchor_face[i_c, i_b]
    anchor = contact_state.center[i_c, i_b]
    is_straight = func_surface_segment(i_anchor, i_face, anchor, target, stress_info, surface_info)
    distance = gs.qd_float(float("inf"))
    if is_straight:
        distance = (target - anchor).norm()
    else:
        for i_t in range(path_info.face_ports.shape[1]):
            i_port = path_info.face_ports[i_face, i_t]
            candidate = footprint_state.port_distance[i_port, i_c, i_b]
            candidate += (target - path_info.positions[i_port]).norm()
            distance = qd.min(distance, candidate)
    return distance, is_straight


def create_anchor_fan_tractions(surface_info: StressSurfaceInfo, n_contacts: int, n_envs: int) -> StressTractionState:
    """Integrate the witnessed face as three finite triangles incident to its anchor.

    Other faces retain the same shared quadrature. This local partition adds
    two face-sized rules, preserves physical radius/profile and uses no point
    load. The dense proof layout remains separate from live contact packing.
    """
    n_samples = surface_info.positions.shape[0] + 2 * surface_info.shape.shape[0]
    return create_tractions(n_samples, n_contacts, n_envs, local_samples=True)


@qd.kernel
def kernel_anchor_fan_profiles(
    contact_state: StressContactState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    for i_q, i_c, i_b in qd.ndrange(
        traction_state.profile.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        traction_state.profile[i_q, i_c, i_b] = 0.0
        traction_state.sample_faces[i_q, i_c, i_b] = -1
        traction_state.sample_positions[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        traction_state.sample_normals[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 3)
        traction_state.sample_shape[i_q, i_c, i_b] = qd.Vector.zero(gs.qd_float, 6)
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            n_quadrature = surface_info.shape.shape[0]
            n_shared = surface_info.positions.shape[0]
            i_anchor = contact_state.anchor_face[i_c, i_b]
            i_face, i_sample, i_fan = i_q // n_quadrature, i_q % n_quadrature, 0
            if i_q >= n_shared:
                i_face = i_anchor
                i_fan = 1 + (i_q - n_shared) // n_quadrature
            anchor = contact_state.center[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            normal = -surface_info.face_normal[i_face]
            position = qd.Vector.zero(gs.qd_float, 3)
            basis = qd.Vector.zero(gs.qd_float, 6)
            area_weight = gs.qd_float(0.0)
            if i_face == i_anchor:
                n = surface_info.gauss.shape[0]
                u, v = surface_info.gauss[i_sample // n], surface_info.gauss[i_sample % n]
                local = surface_info.face_dual[i_face] @ (anchor - surface_info.face_origin[i_face])
                anchor_bary = qd.Vector([1.0 - local[0] - local[1], local[0], local[1]])
                bary = (1.0 - u) * anchor_bary
                bary[i_fan] += u * (1.0 - v)
                bary[(i_fan + 1) % 3] += u * v
                i_first = stress_info.surface_nodes[i_face, i_fan]
                i_second = stress_info.surface_nodes[i_face, (i_fan + 1) % 3]
                first, second = stress_info.vertices[i_first], stress_info.vertices[i_second]
                position = (1.0 - u) * anchor + u * (1.0 - v) * first + u * v * second
                area_weight = (first - anchor).cross(second - anchor).norm() * u
                area_weight *= surface_info.gauss_weights[i_sample // n] * surface_info.gauss_weights[i_sample % n]
                basis = qd.Vector(
                    [
                        bary[0] * (2.0 * bary[0] - 1.0),
                        bary[1] * (2.0 * bary[1] - 1.0),
                        bary[2] * (2.0 * bary[2] - 1.0),
                        4.0 * bary[0] * bary[1],
                        4.0 * bary[0] * bary[2],
                        4.0 * bary[1] * bary[2],
                    ]
                )
            else:
                position = surface_info.positions[i_q]
                basis = surface_info.shape[i_sample]
                area_weight = surface_info.weights[i_q]
            traction_state.sample_positions[i_q, i_c, i_b] = position
            traction_state.sample_normals[i_q, i_c, i_b] = normal
            traction_state.sample_faces[i_q, i_c, i_b] = i_face
            traction_state.sample_shape[i_q, i_c, i_b] = basis
            if area_weight > 0.0 and (position - anchor).norm() < radius:
                qd.atomic_add(footprint_state.candidates[i_c, i_b], 1)
                distance, _ = func_footprint_point_distance(
                    i_face, i_c, i_b, position, contact_state, footprint_state, stress_info, surface_info, path_info
                )
                if distance < radius:
                    relative = distance / radius
                    profile = area_weight * qd.exp(-0.5 * (relative / 0.45) ** 2)
                    profile *= (1.0 - relative * relative) ** 2
                    traction_state.profile[i_q, i_c, i_b] = profile
                    qd.atomic_add(footprint_state.samples[i_c, i_b], 1)
                    qd.atomic_max(footprint_state.maximum_distance[i_c, i_b], distance)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if (
            contact_state.valid[i_c, i_b]
            and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS
            and footprint_state.samples[i_c, i_b] == 0
        ):
            footprint_state.status[i_c, i_b] = StressFootprintStatus.EMPTY_SUPPORT


@qd.kernel
def kernel_footprint_profiles(
    contact_state: StressContactState,
    footprint_state: StressFootprintState,
    traction_state: StressTractionState,
    stress_info: StressInfo,
    surface_info: StressSurfaceInfo,
    path_info: StressPathInfo,
):
    """Use all radius-reachable faces, with true local cones rather than a normal mask.

    The declared profile uses the certified port-path distance; a fully covered
    planar segment uses its exact length. Port paths can overestimate continuous
    distance. Euclidean distance only prunes candidates. This general baseline
    stores the supplied full quadrature profile; live packing remains separate.
    """
    for i_q, i_c, i_b in qd.ndrange(
        surface_info.positions.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]
    ):
        traction_state.profile[i_q, i_c, i_b] = 0.0
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            anchor = contact_state.center[i_c, i_b]
            radius = contact_state.radius[i_c, i_b]
            if (surface_info.positions[i_q] - anchor).norm() < radius:
                qd.atomic_add(footprint_state.candidates[i_c, i_b], 1)
                distance, _ = func_footprint_distance(
                    i_q, i_c, i_b, contact_state, footprint_state, stress_info, surface_info, path_info
                )
                if distance < radius:
                    relative = distance / radius
                    profile = surface_info.weights[i_q] * qd.exp(-0.5 * (relative / 0.45) ** 2)
                    profile *= (1.0 - relative * relative) ** 2
                    traction_state.profile[i_q, i_c, i_b] = profile
                    qd.atomic_add(footprint_state.samples[i_c, i_b], 1)
                    qd.atomic_max(footprint_state.maximum_distance[i_c, i_b], distance)
    for i_c, i_b in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if (
            contact_state.valid[i_c, i_b]
            and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS
            and footprint_state.samples[i_c, i_b] == 0
        ):
            footprint_state.status[i_c, i_b] = StressFootprintStatus.EMPTY_SUPPORT
