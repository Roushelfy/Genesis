"""Diagnostic Quadrants Q10 repartition; all original geometry/profile/constraints retained."""

import quadrants as qd
import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState, func_footprint_point_distance
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState, func_closest_triangle
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionInfo, StressTractionState, func_surface_fan_bary

@qd.kernel
def kernel_focus_fans(contacts: StressContactState, sources: StressSourceState,
                      footprints: StressFootprintState, state: StressTractionState,
                      stress: StressInfo, surface: StressSurfaceInfo, paths: StressPathInfo, info: StressTractionInfo):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        state.sample_count[i_c, i_b] = 0
        footprints.samples[i_c, i_b] = 0
    for i_f, i_c, i_b in qd.ndrange(surface.face_origin.shape[0], contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b]:
            anchor, force = contacts.center[i_c, i_b], contacts.force[i_c, i_b]
            closest, bary = func_closest_triangle(i_f, anchor, surface)
            if (closest - anchor).norm() < contacts.radius[i_c, i_b] and force.norm() > 0.0:
                focus = anchor + force.cross(sources.anchor_moment[i_c, i_b]) / force.dot(force)
                normal = surface.face_edges[i_f][0, :].cross(surface.face_edges[i_f][1, :]).normalized()
                denominator = force.dot(normal)
                if qd.abs(denominator) > 1e-14 * force.norm():
                    focus += force * (surface.face_origin[i_f] - focus).dot(normal) / denominator
                _, bary = func_closest_triangle(i_f, focus, surface)
            state.fan_barycentric[i_f, i_c, i_b] = bary
    for i_q, i_c, i_b in qd.ndrange(state.profile.shape[0], contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b]:
            anchor, radius = contacts.center[i_c, i_b], contacts.radius[i_c, i_b]
            n = surface.gauss.shape[0]
            i_f = i_q // (3 * n * n)
            closest, _ = func_closest_triangle(i_f, anchor, surface)
            if (closest - anchor).norm() < radius:
                i_face, bary = func_surface_fan_bary(i_q, i_c, i_b, state, info)
                origin, edges = surface.face_origin[i_face], surface.face_edges[i_face]
                position = origin + bary[1] * edges[0, :] + bary[2] * edges[1, :]
                i_fan, local = (i_q // (n * n)) % 3, i_q % (n * n)
                center_bary = state.fan_barycentric[i_face, i_c, i_b]
                area = center_bary[(i_fan + 2) % 3] * edges[0, :].cross(edges[1, :]).norm()
                if area > 0.0 and (position - anchor).norm() < radius:
                    distance, _ = func_footprint_point_distance(i_face, i_c, i_b, position, contacts, footprints,
                                                               stress, surface, paths)
                    if distance < radius:
                        u, v = surface.gauss[local // n], surface.gauss[local % n]
                        power = state.fan_coordinate_power[None]
                        weight = area * u**power * power * power * u**(power - 1) * v**(power - 1)
                        weight *= surface.gauss_weights[local // n] * surface.gauss_weights[local % n]
                        relative = distance / radius
                        profile = weight * qd.exp(-0.5 * (relative / 0.45)**2) * (1.0 - relative**2)**2
                        i_slot = qd.atomic_add(state.sample_count[i_c, i_b], 1)
                        state.sample_indices[i_slot, i_c, i_b] = i_q
                        state.profile[i_slot, i_c, i_b] = profile
                        qd.atomic_add(footprints.samples[i_c, i_b], 1)
