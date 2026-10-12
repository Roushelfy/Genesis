"""Coalesced node reduction for complete rigid inertia relief."""

import quadrants as qd

import genesis as gs

from .contact import StressContactState
from .data import StressInfo, StressState
from .scatter import StressScatterWorkspace


@qd.kernel
def kernel_relief_projection(stress_info: StressInfo):
    for i_n in range(stress_info.vertices.shape[0]):
        stress_info.relief[i_n] = stress_info.mass_modes[i_n] @ stress_info.gram_inverse[None]


@qd.func
def func_balance_cooperative(
    omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo, enabled: bool = True
):
    for i_b in range(qd.select(enabled, stress_state.active.shape[0], 0)):
        stress_state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    qd.loop_config(block_dim=256)
    for i_thread in range(
        qd.select(
            enabled, ((stress_info.vertices.shape[0] + 7) // 8) * ((stress_state.active.shape[0] + 31) // 32) * 256, 0
        )
    ):
        i_t = qd.simt.block.thread_idx()
        i_blk = i_thread // 256
        n_env_tiles = (stress_state.active.shape[0] + 31) // 32
        i_row, i_col = i_t // 32, i_t % 32
        i_n = (i_blk // n_env_tiles) * 8 + i_row
        i_b = (i_blk % n_env_tiles) * 32 + i_col
        sh_wrench = qd.simt.block.SharedArray((6, 8, 32), gs.qd_float)
        wrench = qd.Vector.zero(gs.qd_float, 6)
        if i_n < stress_info.vertices.shape[0] and i_b < stress_state.active.shape[0]:
            w = omega[i_b]
            terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
            load = stress_state.force[i_n, i_b] + stress_info.centrifugal[i_n] @ terms
            stress_state.rhs[i_n, i_b] = load
            wrench = stress_info.modes[i_n].transpose() @ load
        for a in qd.static(range(6)):
            sh_wrench[a, i_row, i_col] = wrench[a]
        qd.simt.block.sync()
        if i_row == 0 and i_b < stress_state.active.shape[0]:
            total = qd.Vector.zero(gs.qd_float, 6)
            for j_row in qd.static(range(8)):
                for a in qd.static(range(6)):
                    total[a] += sh_wrench[a, j_row, i_col]
            for a in qd.static(range(6)):
                qd.atomic_add(stress_state.wrench[i_b][a], total[a])
        # Grid-stride iterations reuse shared storage after the reducing warp finishes its reads.
        qd.simt.block.sync()
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], qd.select(enabled, stress_state.active.shape[0], 0)):
        stress_state.rhs[i_n, i_b] -= stress_info.relief[i_n] @ stress_state.wrench[i_b]


@qd.kernel
def kernel_centrifugal_wrench(stress_info: StressInfo):
    for _ in range(1):
        stress_info.centrifugal_wrench[None] = qd.Matrix.zero(gs.qd_float, 6, 6)
    for i_n in range(stress_info.vertices.shape[0]):
        value = stress_info.modes[i_n].transpose() @ stress_info.centrifugal[i_n]
        for a, b in qd.static(qd.ndrange(6, 6)):
            qd.atomic_add(stress_info.centrifugal_wrench[None][a, b], value[a, b])


@qd.func
def func_balance_contacts(
    omega: qd.Tensor,
    stress_state: StressState,
    stress_info: StressInfo,
    contacts: StressContactState,
    scatter: StressScatterWorkspace,
):
    is_admitted = scatter.count[None] <= scatter.tasks.shape[0]
    for i_b in range(qd.select(is_admitted, stress_state.active.shape[0], 0)):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        stress_state.wrench[i_b] = stress_info.centrifugal_wrench[None] @ terms
    for i_pair in range(qd.select(is_admitted, contacts.active_count[None], 0)):
        pair = contacts.active_pairs[i_pair]
        i_c, i_b = pair[0], pair[1]
        if contacts.status[i_c, i_b] == 0:
            integrated = scatter.wrench[i_pair]
            force = qd.Vector([integrated[a] for a in qd.static(range(3))])
            moment = qd.Vector([integrated[a + 3] for a in qd.static(range(3))])
            com = qd.Vector([stress_info.mass_properties[a + 1] for a in qd.static(range(3))])
            com /= stress_info.mass_properties[0]
            moment += (contacts.position[i_c, i_b] - com).cross(force)
            for a in qd.static(range(3)):
                qd.atomic_add(stress_state.wrench[i_b][a], force[a])
                qd.atomic_add(stress_state.wrench[i_b][a + 3], moment[a])
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], qd.select(is_admitted, stress_state.active.shape[0], 0)):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        stress_state.rhs[i_n, i_b] = (
            stress_state.force[i_n, i_b]
            + stress_info.centrifugal[i_n] @ terms
            - stress_info.relief[i_n] @ stress_state.wrench[i_b]
        )
    # Overflow scatter recomputes complete nodal loads; its partial contact wrench is unusable.
    func_balance_cooperative(omega, stress_state, stress_info, not is_admitted)


@qd.kernel(graph=True)
def kernel_balance_contacts(
    omega: qd.Tensor,
    stress_state: StressState,
    stress_info: StressInfo,
    contacts: StressContactState,
    scatter: StressScatterWorkspace,
):
    func_balance_contacts(omega, stress_state, stress_info, contacts, scatter)
