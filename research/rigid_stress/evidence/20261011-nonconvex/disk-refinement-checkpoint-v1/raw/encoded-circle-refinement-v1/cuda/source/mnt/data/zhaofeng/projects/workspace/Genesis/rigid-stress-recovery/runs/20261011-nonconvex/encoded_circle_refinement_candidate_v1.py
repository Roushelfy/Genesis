"""Shared-geometry encoded disk panels with native device-masked refinement and fitting retry."""

from dataclasses import dataclass, replace

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress import footprint, traction
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.traction import (
    StressTractionCircle,
    StressTractionInfo,
    StressTractionState,
    StressTractionStatus,
    func_traction_support_range,
)
from genesis.utils.array_class import V_VEC, V

_native_bary = traction.func_surface_fan_bary


@dataclass(frozen=True)
class RefinedCircle(StressTractionCircle):
    refined: qd.Tensor
    focus: qd.Tensor


def extend_chart(state: StressTractionState) -> StressTractionState:
    chart = RefinedCircle(
        face=state.circle.face,
        radius=state.circle.radius,
        refined=V(dtype=gs.qd_bool, shape=state.circle.face.shape),
        focus=V_VEC(2, dtype=gs.qd_float, shape=state.circle.face.shape),
    )
    chart.refined.fill(False)
    chart.focus.fill(0.0)
    return replace(state, circle=chart)


@qd.func
def func_circle_panel(i_q: int, i_c: int, i_b: int, state: StressTractionState, info: StressTractionInfo):
    fraction, theta = state.circle.focus[i_c, i_b][0], state.circle.focus[i_c, i_b][1]
    half_width = qd.min(0.5 * qd.math.pi, qd.max(0.1, qd.sqrt(8.0 * (1.0 - fraction))))
    radial_edges = qd.Vector([0.0, qd.max(0.0, fraction - 8.0 * (1.0 - fraction)), 0.5 * (1.0 + fraction), 1.0])
    angular_edges = qd.Vector([theta - qd.math.pi, theta - half_width, theta + half_width, theta + qd.math.pi])
    n = info.gauss.shape[0]
    panel, local = i_q // (n * n), i_q % (n * n)
    i_r, i_t = panel // 3, panel % 3
    u, v = info.gauss[local // n], info.gauss[local % n]
    width_r = radial_edges[i_r + 1] - radial_edges[i_r]
    width_t = angular_edges[i_t + 1] - angular_edges[i_t]
    distance = radial_edges[i_r] + width_r * u
    angle = angular_edges[i_t] + width_t * v
    return distance, angle, width_r * width_t


@qd.func
def func_encoded_fan_bary(i_sample: int, i_c: int, i_b: int, state: StressTractionState, info: StressTractionInfo):
    i_face = 0
    bary = qd.Vector.zero(gs.qd_float, 3)
    circle_start = 3 * info.positions.shape[0]
    if i_sample >= circle_start and state.circle.refined[i_c, i_b]:
        i_face = state.circle.face[i_c, i_b]
        distance, angle, _ = func_circle_panel(i_sample - circle_start, i_c, i_b, state, info)
        edges = info.face_edges[i_face]
        normal = info.inward_normals[i_face * info.gauss.shape[0] ** 2]
        length = edges[0, :].norm()
        tangent = edges[0, :] / length
        second = normal.cross(tangent)
        distance *= state.circle.radius[i_c, i_b]
        dv = distance * qd.sin(angle) / edges[1, :].dot(second)
        du = (distance * qd.cos(angle) - edges[1, :].dot(tangent) * dv) / length
        bary = state.fan_barycentric[i_face, i_c, i_b] + qd.Vector([-du - dv, du, dv])
    else:
        i_face, bary = _native_bary(i_sample, i_c, i_b, state, info)
    return i_face, bary


@qd.kernel
def kernel_refine_failed_disks(
    contacts: StressContactState,
    sources: StressSourceState,
    footprints: StressFootprintState,
    state: StressTractionState,
    surface: StressSurfaceInfo,
    info: StressTractionInfo,
):
    """Refine unresolved single-source verified disks while retaining their original physical inputs.

    Shared pair support and nonpositive compression keep their original diagnostic route. Only the numerical
    integration partitions change. Successful fits keep their profile, indices, dual and error state untouched.
    """
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        first, last = func_traction_support_range(i_c, i_b, state)
        unresolved = state.status[i_c, i_b] == StressTractionStatus.NUMERICAL_FAILURE
        unresolved = unresolved or state.status[i_c, i_b] == StressTractionStatus.SAMPLED_WRENCH_INFEASIBLE
        i_face = state.circle.face[i_c, i_b]
        if contacts.valid[i_c, i_b] and unresolved and first == i_c and last == i_c + 1 and i_face >= 0:
            normal = -surface.face_normal[i_face]
            compression = contacts.force[i_c, i_b].dot(normal)
            if compression > 0.0:
                centroid = normal.cross(sources.anchor_moment[i_c, i_b]) / (compression * state.circle.radius[i_c, i_b])
                fraction = centroid.norm()
                if fraction < 1.0 and 9 * surface.gauss.shape[0] ** 2 <= state.profile.shape[0]:
                    tangent = surface.face_edges[i_face][0, :].normalized()
                    second = normal.cross(tangent)
                    state.circle.focus[i_c, i_b] = qd.Vector(
                        [
                            qd.min(1.0 - 1e-12, fraction),
                            qd.atan2(centroid.dot(second), centroid.dot(tangent)),
                        ]
                    )
                    state.circle.refined[i_c, i_b] = True
                    state.sample_count[i_c, i_b] = 9 * surface.gauss.shape[0] ** 2
                    footprints.samples[i_c, i_b] = state.sample_count[i_c, i_b]
                    footprints.candidates[i_c, i_b] = state.sample_count[i_c, i_b]
                    footprints.maximum_distance[i_c, i_b] = 0.0
    for i_q, i_c, i_b in qd.ndrange(9 * surface.gauss.shape[0] ** 2, contacts.valid.shape[0], contacts.valid.shape[1]):
        if state.circle.refined[i_c, i_b]:
            distance, _, jacobian = func_circle_panel(i_q, i_c, i_b, state, info)
            local = i_q % surface.gauss.shape[0] ** 2
            radius = state.circle.radius[i_c, i_b]
            profile = radius * radius * distance * jacobian
            profile *= (
                surface.gauss_weights[local // surface.gauss.shape[0]]
                * surface.gauss_weights[local % surface.gauss.shape[0]]
            )
            profile *= qd.exp(-0.5 * (distance / 0.45) ** 2) * (1.0 - distance * distance) ** 2
            state.profile[i_q, i_c, i_b] = profile
            state.sample_indices[i_q, i_c, i_b] = 3 * info.positions.shape[0] + i_q
            qd.atomic_max(footprints.maximum_distance[i_c, i_b], radius * distance)


@qd.kernel
def kernel_clear_chart(state: StressTractionState):
    for i_c, i_b in qd.ndrange(state.circle.refined.shape[0], state.circle.refined.shape[1]):
        state.circle.refined[i_c, i_b] = False
        state.circle.focus[i_c, i_b] = qd.Vector.zero(gs.qd_float, 2)


def install_geometry():
    """Install only the experimental encoded geometry; production files stay unchanged."""
    traction.func_surface_fan_bary = func_encoded_fan_bary
    footprint.func_surface_fan_bary = func_encoded_fan_bary
