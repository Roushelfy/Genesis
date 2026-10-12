"""Repartition unresolved surface fans around their original wrench lines."""

import quadrants as qd

import genesis as gs

from .contact import StressContactState
from .data import StressInfo
from .footprint import StressFootprintState, StressFootprintStatus, func_footprint_point_distance
from .paths import StressPathInfo
from .provenance import StressSourceState, func_closest_triangle
from .surface import StressSurfaceInfo
from .traction import (
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    func_surface_fan_bary,
    func_traction_constraint_active,
    func_traction_support_range,
)


@qd.kernel
def kernel_select_failed_fans(
    contacts: StressContactState, footprints: StressFootprintState, state: StressTractionState
):
    """Refit every constraint sharing a changed support; preserve successful unrelated groups."""
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        selected, has_fan = False, False
        if contacts.valid[i_c, i_b]:
            start, end = func_traction_support_range(i_c, i_b, state)
            for i_s in range(start, end):
                if contacts.valid[i_s, i_b] and footprints.status[i_s, i_b] == StressFootprintStatus.SUCCESS:
                    has_fan = has_fan or state.circle.face[i_s, i_b] < 0
                    if func_traction_constraint_active(i_s, i_b, state):
                        unresolved = state.status[i_s, i_b] == StressTractionStatus.NUMERICAL_FAILURE
                        unresolved = (
                            unresolved or state.status[i_s, i_b] == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
                        )
                        selected = selected or unresolved
        state.fan_refined[i_c, i_b] = selected and has_fan


@qd.func
def func_refine_source_fan(
    i_c: int,
    i_b: int,
    epsilon: float,
    contacts: StressContactState,
    sources: StressSourceState,
    footprints: StressFootprintState,
    state: StressTractionState,
    stress: StressInfo,
    surface: StressSurfaceInfo,
    paths: StressPathInfo,
    info: StressTractionInfo,
):
    anchor, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
    force, moment = contacts.force[i_c, i_b], sources.anchor_moment[i_c, i_b]
    if qd.static(len(state.priors.force.shape) == 2):
        force, moment = state.priors.force[i_c, i_b], state.priors.anchor_moment[i_c, i_b]
    state.sample_count[i_c, i_b] = 0
    footprints.samples[i_c, i_b] = 0
    footprints.candidates[i_c, i_b] = 0
    footprints.maximum_distance[i_c, i_b] = 0.0
    for i_f in range(surface.face_origin.shape[0]):
        closest, bary = func_closest_triangle(i_f, anchor, surface)
        if (closest - anchor).norm() < radius and force.norm() > 0.0:
            # This line has the original wrench's transverse lever arm. It
            # selects integration charts, never a replacement application point.
            focus = anchor + force.cross(moment) / force.dot(force)
            normal = surface.face_normal[i_f]
            denominator = force.dot(normal)
            if qd.abs(denominator) > 64.0 * epsilon * force.norm():
                focus += force * (surface.face_origin[i_f] - focus).dot(normal) / denominator
            _, bary = func_closest_triangle(i_f, focus, surface)
        state.fan_barycentric[i_f, i_c, i_b] = bary
    n = surface.gauss.shape[0]
    for i_q in range(state.profile.shape[0]):
        i_f = i_q // (3 * n * n)
        closest, _ = func_closest_triangle(i_f, anchor, surface)
        # Pruning uses the original anchor, irrespective of the new chart center.
        if (closest - anchor).norm() < radius:
            i_face, bary = func_surface_fan_bary(i_q, i_c, i_b, state, info)
            origin, edges = surface.face_origin[i_face], surface.face_edges[i_face]
            position = origin + bary[1] * edges[0, :] + bary[2] * edges[1, :]
            i_fan, local = (i_q // (n * n)) % 3, i_q % (n * n)
            center_bary = state.fan_barycentric[i_face, i_c, i_b]
            area = center_bary[(i_fan + 2) % 3] * edges[0, :].cross(edges[1, :]).norm()
            if area > 0.0 and (position - anchor).norm() < radius:
                footprints.candidates[i_c, i_b] += 1
                distance, _ = func_footprint_point_distance(
                    i_face, i_c, i_b, position, contacts, footprints, stress, surface, paths
                )
                if distance < radius:
                    u, v = surface.gauss[local // n], surface.gauss[local % n]
                    power = state.fan_coordinate_power[None]
                    weight = area * u**power * power * power * u ** (power - 1) * v ** (power - 1)
                    weight *= surface.gauss_weights[local // n] * surface.gauss_weights[local % n]
                    relative = distance / radius
                    profile = weight * qd.exp(-0.5 * (relative / 0.45) ** 2) * (1.0 - relative**2) ** 2
                    slot = state.sample_count[i_c, i_b]
                    state.sample_count[i_c, i_b] += 1
                    state.sample_indices[slot, i_c, i_b] = i_q
                    state.profile[slot, i_c, i_b] = profile
                    footprints.samples[i_c, i_b] += 1
                    footprints.maximum_distance[i_c, i_b] = qd.max(footprints.maximum_distance[i_c, i_b], distance)


@qd.kernel
def kernel_refine_failed_fans(
    contacts: StressContactState,
    sources: StressSourceState,
    footprints: StressFootprintState,
    state: StressTractionState,
    stress: StressInfo,
    surface: StressSurfaceInfo,
    paths: StressPathInfo,
    info: StressTractionInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if state.fan_refined[i_c, i_b] and state.circle.face[i_c, i_b] < 0:
            func_refine_source_fan(
                i_c, i_b, epsilon, contacts, sources, footprints, state, stress, surface, paths, info
            )
