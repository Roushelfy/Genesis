"""Unselected native fan-frontier launch experiment. Same geometry,samples,profiles and acceptance."""

from dataclasses import dataclass
from typing import ClassVar
import weakref
import quadrants as qd
import genesis as gs
from genesis.utils.array_class import V, DataKind
from genesis.utils.misc import qd_to_numpy
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.engine.solvers.rigid.stress.paths import StressPathInfo
from genesis.engine.solvers.rigid.stress.traction import StressTractionState, StressTractionInfo, func_surface_fan_bary
from genesis.engine.solvers.rigid.stress.footprint import StressFootprintState, StressFootprintStatus, func_footprint_point_distance
from genesis.engine.solvers.rigid.stress.provenance import func_closest_triangle

@dataclass(frozen=True)
class FanFrontier:
    kind: ClassVar[DataKind] = DataKind.SCRATCH
    indices: qd.Tensor
    count: qd.Tensor
    maximum: qd.Tensor

workspaces = {}

def workspace(tractions):
    key = id(tractions)
    if key not in workspaces:
        shape = tractions.fan_barycentric.shape
        value = FanFrontier(indices=V(dtype=gs.qd_int,shape=shape),count=V(dtype=gs.qd_int,shape=shape[1:]),maximum=V(dtype=gs.qd_int,shape=()))
        workspaces[key] = (weakref.ref(tractions, lambda _ref: workspaces.pop(key,None)),value)
    return workspaces[key][1]

@qd.kernel
def kernel_fan_frontier(contact_state: StressContactState, footprint_state: StressFootprintState, traction_state: StressTractionState, stress_info: StressInfo, surface_info: StressSurfaceInfo, path_info: StressPathInfo, traction_info: StressTractionInfo, epsilon: float, frontier: FanFrontier):
    for _ in range(1):
        frontier.maximum[None] = 0
    for (i_c, i_b) in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        traction_state.sample_count[i_c, i_b] = 0
        traction_state.circle.face[i_c, i_b] = -1
        traction_state.circle.radius[i_c, i_b] = 0.0
        traction_state.circle.refined[i_c, i_b] = False
        traction_state.circle.focus[i_c, i_b] = qd.Vector.zero(gs.qd_float, 2)
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            i_face = contact_state.anchor_face[i_c, i_b]
            (origin, edges) = (surface_info.face_origin[i_face], surface_info.face_edges[i_face])
            (anchor, radius) = (contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b])
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
    for (i_f, i_c, i_b) in qd.ndrange(surface_info.face_origin.shape[0], contact_state.valid.shape[0], contact_state.valid.shape[1]):
        bary = qd.Vector.zero(gs.qd_float, 3)
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            (_, bary) = func_closest_triangle(i_f, contact_state.center[i_c, i_b], surface_info)
        traction_state.fan_barycentric[i_f, i_c, i_b] = bary
        candidate = False
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
            circle = traction_state.circle.face[i_c, i_b]
            if circle >= 0:
                candidate = i_f == circle
            else:
                (origin, edges) = (surface_info.face_origin[i_f], surface_info.face_edges[i_f])
                center = origin + bary[1] * edges[0, :] + bary[2] * edges[1, :]
                candidate = (center - contact_state.center[i_c, i_b]).norm() < contact_state.radius[i_c, i_b]
        frontier.indices[i_f, i_c, i_b] = i_f if candidate else -1
    for (i_c, i_b) in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        count = 0
        for i_f in range(surface_info.face_origin.shape[0]):
            face = frontier.indices[i_f, i_c, i_b]
            if face >= 0:
                frontier.indices[count, i_c, i_b] = face
                count += 1
        frontier.count[i_c, i_b] = count
        qd.atomic_max(frontier.maximum[None], count)

@qd.kernel
def kernel_fan_frontier_samples(contact_state: StressContactState, footprint_state: StressFootprintState, traction_state: StressTractionState, stress_info: StressInfo, surface_info: StressSurfaceInfo, path_info: StressPathInfo, traction_info: StressTractionInfo, epsilon: float, frontier: FanFrontier, maximum: int):
    for (i_task, i_c, i_b) in qd.ndrange(maximum * 3 * surface_info.gauss.shape[0] ** 2, contact_state.valid.shape[0], contact_state.valid.shape[1]):
        n_local = 3 * surface_info.gauss.shape[0] ** 2
        i_work = i_task // n_local
        if i_work < frontier.count[i_c, i_b]:
            i_q = frontier.indices[i_work, i_c, i_b] * n_local + i_task % n_local
            if traction_state.circle.face[i_c, i_b] >= 0:
                i_q = i_task % n_local
            if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS:
                (anchor, radius) = (contact_state.center[i_c, i_b], contact_state.radius[i_c, i_b])
                n = surface_info.gauss.shape[0]
                i_circle = traction_state.circle.face[i_c, i_b]
                i_sample = i_q
                is_candidate = False
                if i_circle >= 0:
                    is_candidate = i_q < n * n
                    i_sample = 3 * surface_info.positions.shape[0] + i_q
                else:
                    i_face = i_q // (3 * n * n)
                    center_bary = traction_state.fan_barycentric[i_face, i_c, i_b]
                    (origin, edges) = (surface_info.face_origin[i_face], surface_info.face_edges[i_face])
                    center = origin + center_bary[1] * edges[0, :] + center_bary[2] * edges[1, :]
                    is_candidate = (center - anchor).norm() < radius
                if is_candidate:
                    (i_face, bary) = func_surface_fan_bary(i_sample, i_c, i_b, traction_state, traction_info)
                    (origin, edges) = (surface_info.face_origin[i_face], surface_info.face_edges[i_face])
                    position = origin + bary[1] * edges[0, :] + bary[2] * edges[1, :]
                    (weight, distance) = (gs.qd_float(0.0), gs.qd_float(float('inf')))
                    i_local = i_q % (n * n)
                    (u, v) = (surface_info.gauss[i_local // n], surface_info.gauss[i_local % n])
                    power = traction_state.fan_coordinate_power[None]
                    if i_circle >= 0:
                        qd.atomic_add(footprint_state.candidates[i_c, i_b], 1)
                        distance = radius * u ** power
                        weight = 2.0 * qd.math.pi * radius * radius * power * u ** (2 * power - 1)
                    else:
                        i_fan = i_q // (n * n) % 3
                        center_bary = traction_state.fan_barycentric[i_face, i_c, i_b]
                        area = center_bary[(i_fan + 2) % 3] * edges[0, :].cross(edges[1, :]).norm()
                        if area > 0.0 and (position - anchor).norm() < radius:
                            qd.atomic_add(footprint_state.candidates[i_c, i_b], 1)
                            (distance, _) = func_footprint_point_distance(i_face, i_c, i_b, position, contact_state, footprint_state, stress_info, surface_info, path_info)
                            weight = area * u ** power * power * power * u ** (power - 1) * v ** (power - 1)
                    if weight > 0.0 and distance < radius:
                        weight *= surface_info.gauss_weights[i_local // n] * surface_info.gauss_weights[i_local % n]
                        relative = distance / radius
                        profile = weight * qd.exp(-0.5 * (relative / 0.45) ** 2) * (1.0 - relative * relative) ** 2
                        i_slot = qd.atomic_add(traction_state.sample_count[i_c, i_b], 1)
                        traction_state.sample_indices[i_slot, i_c, i_b] = i_sample
                        traction_state.profile[i_slot, i_c, i_b] = profile
                        qd.atomic_add(footprint_state.samples[i_c, i_b], 1)
                        qd.atomic_max(footprint_state.maximum_distance[i_c, i_b], distance)
    for (i_c, i_b) in qd.ndrange(contact_state.valid.shape[0], contact_state.valid.shape[1]):
        if contact_state.valid[i_c, i_b] and footprint_state.status[i_c, i_b] == StressFootprintStatus.SUCCESS and (footprint_state.samples[i_c, i_b] == 0):
            footprint_state.status[i_c, i_b] = StressFootprintStatus.EMPTY_SUPPORT


def kernel_surface_fan_profiles(contacts,footprints,tractions,model,surface,paths,info,epsilon):
    frontier = workspace(tractions)
    kernel_fan_frontier(contacts,footprints,tractions,model,surface,paths,info,epsilon,frontier)
    maximum = int(qd_to_numpy(frontier.maximum,transpose=True))
    kernel_fan_frontier_samples(contacts,footprints,tractions,model,surface,paths,info,epsilon,frontier,maximum)
