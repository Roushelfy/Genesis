"""Bounded face-parallel contact integration with complete native overflow fallback."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V

from .contact import (
    StressContactState,
    func_contact_sample,
    func_force_frame,
    func_pack_contacts,
    func_pressure_correct_warp,
    func_pressure_initial,
    func_pressure_warp,
    func_refine_contacts,
    func_scatter_warp,
)
from .data import StressInfo, StressState
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class StressScatterWorkspace:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    tasks: qd.Tensor
    count: qd.Tensor
    wrench: qd.Tensor
    overflow_calls: qd.Tensor
    pressure_gram: qd.Tensor
    load_moments: qd.Tensor
    affine_pressure: qd.Tensor


def create_scatter_workspace(n_envs: int, tasks_per_env: int, reuse_moments: bool = False) -> StressScatterWorkspace:
    # Every task uses one warp. Bound the thread index as well as allocated storage.
    capacity = min(n_envs * tasks_per_env, 2147483647 // 32)
    workspace = StressScatterWorkspace(
        tasks=V_VEC(2, dtype=gs.qd_int, shape=(capacity,)),
        count=V(dtype=qd.i64, shape=()),
        wrench=V_VEC(6, dtype=gs.qd_float, shape=(capacity,)),
        overflow_calls=V(dtype=qd.i64, shape=()),
        pressure_gram=V_MAT(3, 3, dtype=gs.qd_float, shape=(capacity if reuse_moments else 0,)),
        load_moments=V_MAT(6, 3, dtype=gs.qd_float, shape=(capacity if reuse_moments else 0,)),
        # The ndarray backend rejects empty scalar tensors. An unused one-byte
        # placeholder keeps disabled caching compatible with that backend.
        affine_pressure=V(dtype=gs.qd_bool, shape=(max(1, capacity if reuse_moments else 0),)),
    )
    workspace.count.fill(0)
    workspace.overflow_calls.fill(0)
    return workspace


@qd.func
def func_prepare_scatter(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
):
    is_admitted = (
        qd.i64(contacts.active_count[None]) * surface.face_origin.shape[0] <= 2147483647
        and contacts.active_count[None] <= workspace.wrench.shape[0]
    )
    workspace.count[None] = qd.select(is_admitted, 0, workspace.tasks.shape[0] + 1)
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.active.shape[0]):
        state.force[i_n, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_pair in range(qd.select(is_admitted, contacts.active_count[None], 0)):
        workspace.wrench[i_pair] = qd.Vector.zero(gs.qd_float, 6)
    for i_t in range(qd.select(is_admitted, contacts.active_count[None] * surface.face_origin.shape[0], 0)):
        i_pair, i_f = i_t // surface.face_origin.shape[0], i_t % surface.face_origin.shape[0]
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            low, high = surface.face_bounds_low[i_f], surface.face_bounds_high[i_f]
            if qd.static(not cached_bounds):
                origin = surface.face_origin[i_f]
                first = origin + surface.face_edges[i_f][0, :]
                second = origin + surface.face_edges[i_f][1, :]
                low, high = qd.min(origin, first, second), qd.max(origin, first, second)
            center = contacts.center[i_c, i_b]
            distance = qd.max(low - center, 0.0) + qd.max(center - high, 0.0)
            if distance.dot(distance) < contacts.radius[i_c, i_b] ** 2:
                i_t_out = qd.atomic_add(workspace.count[None], 1)
                if i_t_out < workspace.tasks.shape[0]:
                    workspace.tasks[i_t_out] = qd.Vector([i_pair, i_f])


@qd.func
def func_pressure_moments(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
):
    # Task identities stay fixed until scatter; repacking changes atomic order.
    func_prepare_scatter(contacts, state, info, surface, workspace, cached_bounds)
    is_admitted = workspace.count[None] <= workspace.tasks.shape[0]
    for i_pair in range(qd.select(is_admitted, contacts.active_count[None], 0)):
        workspace.pressure_gram[i_pair] = qd.Matrix.zero(gs.qd_float, 3, 3)
        workspace.affine_pressure[i_pair] = False
        pair = contacts.active_pairs[i_pair]
        contacts.candidate_count[pair[0], pair[1]] = 0
    qd.loop_config(block_dim=128)
    for i_thread in range(qd.i32(qd.select(is_admitted, workspace.count[None] * 32, 0))):
        lane, i_t = i_thread % 32, i_thread // 32
        i_pair, i_f = workspace.tasks[i_t][0], workspace.tasks[i_t][1]
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        first, second = func_force_frame(contacts.force[i_c, i_b])
        gram = qd.Matrix.zero(gs.qd_float, 3, 3)
        load = qd.Matrix.zero(gs.qd_float, 6, 3)
        candidates = 0
        n_q = surface.shape.shape[0]
        for chunk in range((n_q + 31) // 32):
            i_q = chunk * 32 + lane
            if i_q < n_q:
                coordinates, weight, _position, shape, _face = func_contact_sample(
                    i_f * n_q + i_q, i_c, i_b, first, second, contacts, surface
                )
                gram += weight * coordinates.outer_product(coordinates)
                load += shape.outer_product(weight * coordinates)
                candidates += gs.qd_int(weight > 0.0)
        for a, b in qd.static(qd.ndrange(3, 3)):
            gram[a, b] = qd.simt.subgroup.reduce_all_add(gram[a, b])
        for a, b in qd.static(qd.ndrange(6, 3)):
            load[a, b] = qd.simt.subgroup.reduce_all_add(load[a, b])
        candidates = qd.simt.subgroup.reduce_all_add(candidates)
        if lane == 0:
            workspace.load_moments[i_t] = load
            for a, b in qd.static(qd.ndrange(3, 3)):
                qd.atomic_add(workspace.pressure_gram[i_pair][a, b], gram[a, b])
            qd.atomic_add(contacts.candidate_count[i_c, i_b], candidates)
    for i_pair in range(qd.select(is_admitted, contacts.active_count[None], 0)):
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        coefficient, weight_sum, status = func_pressure_initial(workspace.pressure_gram[i_pair])
        contacts.coefficient[i_c, i_b] = coefficient
        contacts.weight_sum[i_c, i_b] = weight_sum
        contacts.status[i_c, i_b] = status
        workspace.affine_pressure[i_pair] = status == 0
    # Overflow recomputes the complete original pressure, with no cache accesses.
    func_pressure_warp(contacts, surface, False, not is_admitted)


@qd.kernel(graph=True)
def kernel_pressure_moments(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
):
    func_pack_contacts(contacts)
    func_pressure_moments(contacts, state, info, surface, workspace, cached_bounds)
    func_pressure_correct_warp(contacts, surface)
    func_refine_contacts(contacts)
    func_pressure_warp(contacts, surface, True)
    func_pressure_correct_warp(contacts, surface)


@qd.func
def func_scatter_faces(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
    reuse_moments: qd.template() = False,
):
    if qd.static(not reuse_moments):
        func_prepare_scatter(contacts, state, info, surface, workspace, cached_bounds)
    qd.loop_config(block_dim=128)
    for i_thread in range(
        qd.i32(qd.select(workspace.count[None] <= workspace.tasks.shape[0], workspace.count[None] * 32, 0))
    ):
        i_lane, i_t = i_thread % 32, i_thread // 32
        i_pair, i_f = workspace.tasks[i_t][0], workspace.tasks[i_t][1]
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        force = contacts.force[i_c, i_b]
        first, second = func_force_frame(force)
        n_q = surface.shape.shape[0]
        i_q_start, n_q_current = i_f * n_q, n_q
        if contacts.is_refined[i_c, i_b] and i_f == contacts.anchor_face[i_c, i_b]:
            i_q_start, n_q_current = surface.positions.shape[0], 7 * n_q
        if qd.static(reuse_moments):
            n_q_current = qd.select(contacts.status[i_c, i_b] == 0, n_q_current, 0)
        total, moment = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
        load = qd.Matrix.zero(gs.qd_float, 6, 3)
        use_moments = False
        if qd.static(reuse_moments):
            use_moments = (
                workspace.affine_pressure[i_pair]
                and not contacts.is_refined[i_c, i_b]
                and contacts.status[i_c, i_b] == 0
            )
        if use_moments:
            if i_lane < 6:
                row = qd.Vector([workspace.load_moments[i_t][i_lane, a] for a in qd.static(range(3))])
                nodal_force = row.dot(contacts.coefficient[i_c, i_b]) * force
                load[i_lane, :] = nodal_force
                total = nodal_force
                node = info.surface_nodes[i_f, i_lane]
                # P2 partition of unity and linear completeness preserve the wrench.
                moment = (info.vertices[node] - contacts.position[i_c, i_b]).cross(nodal_force)
        else:
            for i_q_block in range((n_q_current + 31) // 32):
                i_q_ = i_q_block * 32 + i_lane
                if i_q_ < n_q_current:
                    coordinates, weight, position, shape, _face = func_contact_sample(
                        i_q_start + i_q_, i_c, i_b, first, second, contacts, surface
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
        if i_lane == 0:
            for i_local in range(6):
                i_n = info.surface_nodes[i_f, i_local]
                for a in qd.static(range(3)):
                    qd.atomic_add(state.force[i_n, i_b][a], load[i_local, a])
            for a in qd.static(range(3)):
                qd.atomic_add(workspace.wrench[i_pair][a], total[a])
                qd.atomic_add(workspace.wrench[i_pair][a + 3], moment[a])
    for i_pair in range(qd.select(workspace.count[None] <= workspace.tasks.shape[0], contacts.active_count[None], 0)):
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            accumulated = workspace.wrench[i_pair]
            total = qd.Vector([accumulated[a] for a in qd.static(range(3))])
            moment = qd.Vector([accumulated[a + 3] for a in qd.static(range(3))])
            force = contacts.force[i_c, i_b]
            contacts.force_error[i_c, i_b] = (total - force).norm()
            contacts.moment_error[i_c, i_b] = moment.norm()
            if (total - force).norm() > 1e-8 * force.norm() or moment.norm() > 1e-8 * force.norm() * contacts.radius[
                i_c, i_b
            ]:
                contacts.status[i_c, i_b] = 4
    is_overflow = workspace.count[None] > workspace.tasks.shape[0]
    if is_overflow:
        workspace.overflow_calls[None] += 1
    # The incomplete task buffer contributes no loads. Overflow starts from zero
    # and integrates every original contact/face/sample, without a host transfer.
    func_scatter_warp(contacts, state, info, surface, cached_bounds, is_overflow)


def kernel_scatter_faces(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: bool,
    reuse_moments: bool = False,
):
    kernel_scatter_faces_impl(contacts, state, info, surface, workspace, cached_bounds, reuse_moments)


@qd.kernel(graph=True)
def kernel_scatter_faces_impl(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
    reuse_moments: qd.template(),
):
    if qd.static(not reuse_moments):
        func_pack_contacts(contacts)
    func_scatter_faces(contacts, state, info, surface, workspace, cached_bounds, reuse_moments)


@qd.kernel(graph=True)
def kernel_fit_scatter_moments(
    contacts: StressContactState,
    state: StressState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    workspace: StressScatterWorkspace,
    cached_bounds: qd.template(),
):
    func_pack_contacts(contacts)
    func_pressure_moments(contacts, state, info, surface, workspace, cached_bounds)
    func_pressure_correct_warp(contacts, surface)
    func_refine_contacts(contacts)
    func_pressure_warp(contacts, surface, True)
    func_pressure_correct_warp(contacts, surface)
    func_scatter_faces(contacts, state, info, surface, workspace, cached_bounds, True)
