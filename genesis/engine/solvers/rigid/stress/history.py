"""Optional native four-column load/displacement history with full residual rejection."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V

from .data import StressState


@dataclass(frozen=True)
class StressHistoryState:
    kind: ClassVar[DataKind] = DataKind.WARMSTART

    loads: qd.Tensor
    displacements: qd.Tensor
    basis: qd.Tensor
    dual_basis: qd.Tensor
    coefficients: qd.Tensor
    norm_squared: qd.Tensor
    raw_norm_squared: qd.Tensor
    count: qd.Tensor
    head: qd.Tensor
    dirty: qd.Tensor
    hit: qd.Tensor


class StressHistory:
    def __init__(self, n_nodes: int, n_envs: int):
        shape = (n_nodes, n_envs, 4)
        self.state = StressHistoryState(
            V_VEC(3, dtype=gs.qd_float, shape=shape),
            V_VEC(3, dtype=gs.qd_float, shape=shape),
            V_VEC(3, dtype=gs.qd_float, shape=shape),
            V_VEC(3, dtype=gs.qd_float, shape=shape),
            V_VEC(4, dtype=gs.qd_float, shape=(n_envs,)),
            V(dtype=gs.qd_float, shape=(n_envs,)),
            V(dtype=gs.qd_float, shape=(n_envs,)),
            V(dtype=gs.qd_int, shape=(n_envs,)),
            V(dtype=gs.qd_int, shape=(n_envs,)),
            V(dtype=gs.qd_bool, shape=(n_envs,)),
            V(dtype=gs.qd_bool, shape=(n_envs,)),
        )
        self.state.count.fill(0)
        self.state.head.fill(0)
        self.state.dirty.fill(True)
        self.state.hit.fill(False)

    def predict(self, state: StressState):
        kernel_predict(self.state, state)

    def mark_hits(self, state: StressState):
        kernel_hits(self.state, state)

    def append(self, state: StressState):
        kernel_append(self.state, state)

    def reset(self, envs_idx):
        kernel_reset(envs_idx, self.state)


@qd.kernel(graph=True)
def kernel_predict(history: StressHistoryState, state: StressState):
    # Two-pass modified Gram-Schmidt avoids normal-equation conditioning for nearly repeated loads.
    for i_h in qd.static(range(4)):
        for i_b in range(history.count.shape[0]):
            if history.dirty[i_b]:
                history.raw_norm_squared[i_b] = 0.0
        for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
            if history.dirty[i_b]:
                column = (history.head[i_b] + i_h) % 4
                load = qd.Vector.zero(gs.qd_float, 3)
                displacement = qd.Vector.zero(gs.qd_float, 3)
                if i_h < history.count[i_b]:
                    load = history.loads[i_n, i_b, column]
                    displacement = history.displacements[i_n, i_b, column]
                history.basis[i_n, i_b, i_h] = load
                history.dual_basis[i_n, i_b, i_h] = displacement
                qd.atomic_add(history.raw_norm_squared[i_b], load.dot(load))
        for _ in qd.static(range(2)):
            for i_b in range(history.count.shape[0]):
                if history.dirty[i_b]:
                    history.coefficients[i_b] = qd.Vector.zero(gs.qd_float, 4)
            for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
                if history.dirty[i_b]:
                    for j in range(i_h):
                        qd.atomic_add(
                            history.coefficients[i_b][j], history.basis[i_n, i_b, i_h].dot(history.basis[i_n, i_b, j])
                        )
            for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
                if history.dirty[i_b]:
                    load, displacement = history.basis[i_n, i_b, i_h], history.dual_basis[i_n, i_b, i_h]
                    for j in range(i_h):
                        load -= history.coefficients[i_b][j] * history.basis[i_n, i_b, j]
                        displacement -= history.coefficients[i_b][j] * history.dual_basis[i_n, i_b, j]
                    history.basis[i_n, i_b, i_h] = load
                    history.dual_basis[i_n, i_b, i_h] = displacement
        for i_b in range(history.count.shape[0]):
            if history.dirty[i_b]:
                history.norm_squared[i_b] = 0.0
        for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
            if history.dirty[i_b]:
                load = history.basis[i_n, i_b, i_h]
                qd.atomic_add(history.norm_squared[i_b], load.dot(load))
        for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
            if history.dirty[i_b]:
                load, displacement = qd.Vector.zero(gs.qd_float, 3), qd.Vector.zero(gs.qd_float, 3)
                if history.norm_squared[i_b] > 1e-24 * history.raw_norm_squared[i_b]:
                    denominator = qd.sqrt(history.norm_squared[i_b])
                    load = history.basis[i_n, i_b, i_h] / denominator
                    displacement = history.dual_basis[i_n, i_b, i_h] / denominator
                history.basis[i_n, i_b, i_h] = load
                history.dual_basis[i_n, i_b, i_h] = displacement
    for i_b in range(history.count.shape[0]):
        history.dirty[i_b] = False
        history.coefficients[i_b] = qd.Vector.zero(gs.qd_float, 4)
    for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
        for i_h in range(4):
            qd.atomic_add(history.coefficients[i_b][i_h], state.rhs[i_n, i_b].dot(history.basis[i_n, i_b, i_h]))
    for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_h in range(4):
            value += history.coefficients[i_b][i_h] * history.dual_basis[i_n, i_b, i_h]
        state.displacement[i_n, i_b] = value


@qd.kernel
def kernel_hits(history: StressHistoryState, state: StressState):
    for i_b in range(history.count.shape[0]):
        history.hit[i_b] = state.valid[i_b]


@qd.kernel(graph=True)
def kernel_append(history: StressHistoryState, state: StressState):
    for i_n, i_b in qd.ndrange(history.loads.shape[0], history.count.shape[0]):
        if state.valid[i_b] and state.step_valid[i_b] and not history.hit[i_b]:
            column = (history.head[i_b] + history.count[i_b]) % 4
            if history.count[i_b] == 4:
                column = history.head[i_b]
            history.loads[i_n, i_b, column] = state.rhs[i_n, i_b]
            history.displacements[i_n, i_b, column] = state.displacement[i_n, i_b]
    for i_b in range(history.count.shape[0]):
        if state.valid[i_b] and state.step_valid[i_b] and not history.hit[i_b]:
            if history.count[i_b] == 4:
                history.head[i_b] = (history.head[i_b] + 1) % 4
            history.count[i_b] = qd.min(4, history.count[i_b] + 1)
            history.dirty[i_b] = True


@qd.kernel
def kernel_reset(envs_idx: qd.types.ndarray(), history: StressHistoryState):
    for i_selected in range(envs_idx.shape[0]):
        i_b = envs_idx[i_selected]
        history.count[i_b] = 0
        history.head[i_b] = 0
        history.dirty[i_b] = True
        history.hit[i_b] = False
