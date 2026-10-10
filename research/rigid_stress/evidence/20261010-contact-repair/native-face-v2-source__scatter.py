"""Bounded face-parallel contact integration with complete native overflow fallback."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V

from .contact import StressContactState, func_contact_sample, func_force_frame, func_pack_contacts, func_scatter_warp
from .data import StressInfo, StressState
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class StressScatterWorkspace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    tasks: qd.Tensor
    count: qd.Tensor
    wrench: qd.Tensor
    overflow_calls: qd.Tensor


def create_scatter_workspace(n_envs: int, tasks_per_env: int) -> StressScatterWorkspace:
    # Every task uses one warp. Bound the thread index as well as allocated storage.
    capacity = min(n_envs * tasks_per_env, 2147483647 // 32)
    workspace = StressScatterWorkspace(
        tasks=V_VEC(2, dtype=gs.qd_int, shape=(capacity,)),
        count=V(dtype=qd.i64, shape=()),
        wrench=V_VEC(6, dtype=gs.qd_float, shape=(capacity,)),
        overflow_calls=V(dtype=qd.i64, shape=()),
    )
    workspace.count.fill(0)
    workspace.overflow_calls.fill(0)
    return workspace


@qd.func
def func_scatter_faces(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
):
    admitted = (
        qd.i64(contacts.active_count[None]) * surface.face_origin.shape[0] <= 2147483647
        and contacts.active_count[None] <= workspace.wrench.shape[0]
    )
    workspace.count[None] = qd.select(admitted, 0, workspace.tasks.shape[0] + 1)
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.active.shape[0]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for slot in range(qd.select(admitted, contacts.active_count[None], 0)):
        workspace.wrench[slot] = qd.Vector.zero(gs.qd_float, 6)
    for task in range(qd.select(admitted, contacts.active_count[None] * surface.face_origin.shape[0], 0)):
        slot, face = task // surface.face_origin.shape[0], task % surface.face_origin.shape[0]
        pair = contacts.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            low, high = surface.face_bounds_low[face], surface.face_bounds_high[face]
            if qd.static(not cached_bounds):
                origin = surface.face_origin[face]
                first = origin + surface.face_edges[face][0, :]
                second = origin + surface.face_edges[face][1, :]
                low, high = qd.min(origin, first, second), qd.max(origin, first, second)
            center = contacts.center[i_c, i_b]
            distance = qd.max(low - center, 0.0) + qd.max(center - high, 0.0)
            if distance.dot(distance) < contacts.radius[i_c, i_b] ** 2:
                index = qd.atomic_add(workspace.count[None], 1)
                if index < workspace.tasks.shape[0]:
                    workspace.tasks[index] = qd.Vector([slot, face])
    qd.loop_config(block_dim=128)
    for i_thread in range(
        qd.i32(qd.select(workspace.count[None] <= workspace.tasks.shape[0], workspace.count[None] * 32, 0))
    ):
        lane, task = i_thread % 32, i_thread // 32
        slot, face = workspace.tasks[task][0], workspace.tasks[task][1]
        pair = contacts.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        force = contacts.force[i_c, i_b]
        first, second = func_force_frame(force)
        n_q = surface.shape.shape[0]
        start, count = face * n_q, n_q
        if contacts.is_refined[i_c, i_b] and face == contacts.anchor_face[i_c, i_b]:
            start, count = surface.positions.shape[0], 7 * n_q
        total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        load = qd.Matrix.zero(gs.qd_float, 6, 3)
        for chunk in range((count + 31) // 32):
            sample = chunk * 32 + lane
            if sample < count:
                coordinates, weight, position, shape, _face = func_contact_sample(
                    start + sample, i_c, i_b, first, second, contacts, surface
                )
                sample_force = weight * qd.max(0.0, coordinates.dot(contacts.coefficient[i_c, i_b])) * force
                total += sample_force
                moment += (position - contacts.position[i_c, i_b]).cross(sample_force)
                load += shape.outer_product(sample_force)
        for a, b in qd.static(qd.ndrange(6, 3)):
            load[a, b] = qd.simt.subgroup.reduce_all_add(load[a, b])
        for a in qd.static(range(3)):
            total[a] = qd.simt.subgroup.reduce_all_add(total[a])
            moment[a] = qd.simt.subgroup.reduce_all_add(moment[a])
        if lane == 0:
            for i_local in range(6):
                i_n = info.surface_nodes[face, i_local]
                for a in qd.static(range(3)):
                    qd.atomic_add(state.force[i_n, i_b][a], load[i_local, a])
            for a in qd.static(range(3)):
                qd.atomic_add(workspace.wrench[slot][a], total[a])
                qd.atomic_add(workspace.wrench[slot][a + 3], moment[a])
    for slot in range(qd.select(workspace.count[None] <= workspace.tasks.shape[0], contacts.active_count[None], 0)):
        pair = contacts.active_pairs[slot]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            accumulated = workspace.wrench[slot]
            total = qd.Vector([accumulated[a] for a in qd.static(range(3))])
            moment = qd.Vector([accumulated[a + 3] for a in qd.static(range(3))])
            force = contacts.force[i_c, i_b]
            contacts.force_error[i_c, i_b] = (total - force).norm()
            contacts.moment_error[i_c, i_b] = moment.norm()
            if (total - force).norm() > 1e-8 * force.norm() or moment.norm() > 1e-8 * force.norm() * contacts.radius[
                i_c, i_b
            ]:
                contacts.status[i_c, i_b] = 4
    overflow = workspace.count[None] > workspace.tasks.shape[0]
    if overflow:
        workspace.overflow_calls[None] += 1
    # The incomplete task buffer contributes no loads. Overflow starts from zero
    # and integrates every original contact/face/sample, without a host transfer.
    func_scatter_warp(contacts, state, info, surface, cached_bounds, overflow)


@qd.kernel(graph=True)
def kernel_scatter_faces(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
):
    func_pack_contacts(contacts)
    func_scatter_faces(contacts, state, info, surface, workspace, cached_bounds)
