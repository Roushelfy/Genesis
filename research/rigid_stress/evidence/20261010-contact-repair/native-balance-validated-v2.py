"""Coalesced node reduction for complete rigid inertia relief."""

import quadrants as qd

import genesis as gs

from .data import StressInfo, StressState


@qd.kernel
def kernel_relief_projection(stress_info: StressInfo):
    for i_n in range(stress_info.vertices.shape[0]):
        stress_info.relief[i_n] = stress_info.mass_modes[i_n] @ stress_info.gram_inverse[None]


@qd.func
def func_balance_cooperative(omega: qd.Tensor, stress_state: StressState, stress_info: StressInfo):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    qd.loop_config(block_dim=256)
    for i_thread in range(
        ((stress_info.vertices.shape[0] + 7) // 8) * ((stress_state.active.shape[0] + 31) // 32) * 256
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
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        stress_state.rhs[i_n, i_b] -= stress_info.relief[i_n] @ stress_state.wrench[i_b]
