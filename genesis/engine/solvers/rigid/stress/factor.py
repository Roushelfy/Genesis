"""Shared sparse block LDL factorization and batched triangular solves."""

import heapq
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V
from genesis.utils.misc import qd_to_numpy

from .data import StressInfo, StressState


@dataclass(frozen=True)
class StressFactorInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    order: qd.Tensor
    row_start: qd.Tensor
    columns: qd.Tensor
    sources: qd.Tensor
    column_start: qd.Tensor
    rows: qd.Tensor
    transpose_entries: qd.Tensor
    lower: qd.Tensor
    diagonal: qd.Tensor
    diagonal_inverse: qd.Tensor
    valid: qd.Tensor


class StressFactor:
    """Symbolically order mesh connectivity and numerically factor it in Quadrants."""

    def __init__(self, stress_info: StressInfo):
        row_start = qd_to_numpy(stress_info.row_start)
        columns = qd_to_numpy(stress_info.columns)
        n_nodes = stress_info.vertices.shape[0]
        neighbors = [set(columns[row_start[i] : row_start[i + 1]]) - {i} for i in range(n_nodes)]
        queue = [(len(adjacent), i) for i, adjacent in enumerate(neighbors)]
        heapq.heapify(queue)
        order, upper = [], []
        live = [True] * n_nodes
        # This is symbolic graph elimination, independent of physical matrix values.
        for _ in range(n_nodes):
            while queue:
                degree, node = heapq.heappop(queue)
                if live[node] and degree == len(neighbors[node]):
                    break
            adjacent = sorted(neighbors[node])
            order.append(node)
            upper.append(adjacent)
            live[node] = False
            for other in adjacent:
                neighbors[other].remove(node)
                neighbors[other].update(adjacent)
                neighbors[other].discard(other)
                heapq.heappush(queue, (len(neighbors[other]), other))
        inverse_order = np.empty(n_nodes, dtype=gs.np_int)
        inverse_order[order] = np.arange(n_nodes)
        lower_rows = [[] for _ in range(n_nodes)]
        for j, adjacent in enumerate(upper):
            for other in adjacent:
                lower_rows[inverse_order[other]].append(j)
        lower_columns = np.array([j for row in lower_rows for j in row], dtype=gs.np_int)
        starts = np.r_[0, np.cumsum([len(row) for row in lower_rows])]
        sources = np.full(len(lower_columns), -1, dtype=gs.np_int)
        for i, row in enumerate(lower_rows):
            node = order[i]
            source_columns = columns[row_start[node] : row_start[node + 1]]
            for local, j in enumerate(row):
                source_node = order[j]
                offset = np.searchsorted(source_columns, source_node)
                if offset < len(source_columns) and source_columns[offset] == source_node:
                    sources[starts[i] + local] = row_start[node] + offset
        row_indices = np.repeat(np.arange(n_nodes), np.diff(starts))
        transpose_order = np.lexsort((row_indices, lower_columns))
        column_start = np.r_[0, np.cumsum(np.bincount(lower_columns, minlength=n_nodes))]
        self.info = StressFactorInfo(
            order=V(dtype=gs.qd_int, shape=(n_nodes,)),
            row_start=V(dtype=gs.qd_int, shape=(n_nodes + 1,)),
            columns=V(dtype=gs.qd_int, shape=(len(lower_columns),)),
            sources=V(dtype=gs.qd_int, shape=(len(lower_columns),)),
            column_start=V(dtype=gs.qd_int, shape=(n_nodes + 1,)),
            rows=V(dtype=gs.qd_int, shape=(len(lower_columns),)),
            transpose_entries=V(dtype=gs.qd_int, shape=(len(lower_columns),)),
            lower=V_MAT(3, 3, dtype=gs.qd_float, shape=(len(lower_columns),)),
            diagonal=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes,)),
            diagonal_inverse=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes,)),
            valid=V(dtype=gs.qd_bool, shape=()),
        )
        self.info.order.from_numpy(np.array(order, dtype=gs.np_int))
        self.info.row_start.from_numpy(starts.astype(gs.np_int, copy=False))
        self.info.columns.from_numpy(lower_columns)
        self.info.sources.from_numpy(sources)
        self.info.column_start.from_numpy(column_start.astype(gs.np_int, copy=False))
        self.info.rows.from_numpy(row_indices[transpose_order].astype(gs.np_int, copy=False))
        self.info.transpose_entries.from_numpy(transpose_order.astype(gs.np_int, copy=False))
        kernel_factor_init(stress_info, self.info)
        kernel_factor(stress_info, self.info)
        if not qd_to_numpy(self.info.valid):
            gs.raise_exception("Native stress factorization has a nonpositive or nonfinite pivot.")

    def solve(
        self,
        young: float,
        state: StressState,
        stress_info: StressInfo,
        cooperative: bool = True,
        only_failed: bool = False,
    ) -> None:
        if cooperative and gs.backend == gs.cuda and self.info.order.shape[0] <= 1024:
            kernel_solve_cooperative(young, state, stress_info, self.info, self.info.order.shape[0], only_failed)
        else:
            kernel_solve(young, state, stress_info, self.info, only_failed)


@qd.kernel
def kernel_factor_init(stress_info: StressInfo, factor_info: StressFactorInfo):
    for i in range(factor_info.order.shape[0]):
        node = factor_info.order[i]
        diagonal = qd.Matrix.zero(gs.qd_float, 3, 3)
        for entry in range(stress_info.row_start[node], stress_info.row_start[node + 1]):
            if stress_info.columns[entry] == node:
                diagonal = stress_info.stiffness[entry]
        for a, c in qd.static(qd.ndrange(3, 3)):
            if not stress_info.is_free[node][a] or not stress_info.is_free[node][c]:
                diagonal[a, c] = 0.0
        for a in qd.static(range(3)):
            if not stress_info.is_free[node][a]:
                diagonal[a, a] = 1.0
        factor_info.diagonal[i] = diagonal
        for entry in range(factor_info.row_start[i], factor_info.row_start[i + 1]):
            source = factor_info.sources[entry]
            j = factor_info.columns[entry]
            other = factor_info.order[j]
            block = qd.Matrix.zero(gs.qd_float, 3, 3)
            if source >= 0:
                block = stress_info.stiffness[source]
            for a, c in qd.static(qd.ndrange(3, 3)):
                if not stress_info.is_free[node][a] or not stress_info.is_free[other][c]:
                    block[a, c] = 0.0
            factor_info.lower[entry] = block
    for _ in range(1):
        factor_info.valid[None] = True


@qd.kernel
def kernel_factor(stress_info: StressInfo, factor_info: StressFactorInfo):
    for _ in range(1):
        for i in range(factor_info.order.shape[0]):
            diagonal = factor_info.diagonal[i]
            for entry in range(factor_info.row_start[i], factor_info.row_start[i + 1]):
                j = factor_info.columns[entry]
                value = factor_info.lower[entry]
                for previous in range(factor_info.row_start[i], entry):
                    k = factor_info.columns[previous]
                    lo, hi = factor_info.row_start[j], factor_info.row_start[j + 1]
                    for _search in range(32):
                        if lo < hi:
                            middle = (lo + hi) // 2
                            if factor_info.columns[middle] < k:
                                lo = middle + 1
                            else:
                                hi = middle
                    if lo < factor_info.row_start[j + 1]:  # noqa: SIM102 - guard the indirect array read
                        if factor_info.columns[lo] == k:
                            value -= (
                                factor_info.lower[previous]
                                @ factor_info.diagonal[k]
                                @ factor_info.lower[lo].transpose()
                            )
                lower = value @ factor_info.diagonal_inverse[j]
                factor_info.lower[entry] = lower
                diagonal -= lower @ factor_info.diagonal[j] @ lower.transpose()
            diagonal = 0.5 * (diagonal + diagonal.transpose())
            factor_info.diagonal[i] = diagonal
            pivot = diagonal[0, 0]
            second = diagonal[1, 1] - diagonal[1, 0] ** 2 / pivot
            valid = pivot > 0.0 and second > 0.0 and diagonal.determinant() > 0.0
            factor_info.valid[None] = factor_info.valid[None] and valid
            factor_info.diagonal_inverse[i] = diagonal.inverse()


@qd.kernel
def kernel_solve(
    young: float,
    stress_state: StressState,
    stress_info: StressInfo,
    factor_info: StressFactorInfo,
    only_failed: qd.template(),
):
    for i_b in range(stress_state.active.shape[0]):
        if qd.static(not only_failed) or not stress_state.valid[i_b]:
            for i in range(factor_info.order.shape[0]):
                node = factor_info.order[i]
                value = stress_state.rhs[node, i_b] * stress_info.is_free[node] / young
                for entry in range(factor_info.row_start[i], factor_info.row_start[i + 1]):
                    j = factor_info.columns[entry]
                    other = factor_info.order[j]
                    value -= factor_info.lower[entry] @ stress_state.displacement[other, i_b]
                stress_state.displacement[node, i_b] = value

            for i in range(factor_info.order.shape[0]):
                node = factor_info.order[i]
                stress_state.displacement[node, i_b] = (
                    factor_info.diagonal_inverse[i] @ stress_state.displacement[node, i_b]
                )
            for reverse in range(factor_info.order.shape[0]):
                i = factor_info.order.shape[0] - reverse - 1
                node = factor_info.order[i]
                value = stress_state.displacement[node, i_b]
                for entry in range(factor_info.column_start[i], factor_info.column_start[i + 1]):
                    j = factor_info.rows[entry]
                    other = factor_info.order[j]
                    lower_entry = factor_info.transpose_entries[entry]
                    value -= factor_info.lower[lower_entry].transpose() @ stress_state.displacement[other, i_b]
                stress_state.displacement[node, i_b] = value
    for i_b in range(stress_state.active.shape[0]):
        if qd.static(only_failed):
            stress_state.active[i_b] = gs.qd_int(not stress_state.valid[i_b])
            if not stress_state.valid[i_b]:
                stress_state.fallbacks[i_b] += 1
            stress_state.valid[i_b] = True


@qd.kernel
def kernel_solve_cooperative(
    young: float,
    stress_state: StressState,
    stress_info: StressInfo,
    factor_info: StressFactorInfo,
    n_nodes: qd.template(),
    only_failed: qd.template(),
):
    qd.loop_config(block_dim=32)
    for i_thread in range(stress_state.active.shape[0] * 32):
        lane = i_thread % 32
        i_b = i_thread // 32
        if qd.static(not only_failed) or not stress_state.valid[i_b]:
            work = qd.simt.block.SharedArray((n_nodes, 3), gs.qd_float)
            for i in range(factor_info.order.shape[0]):
                start, end = factor_info.row_start[i], factor_info.row_start[i + 1]
                product = qd.Vector.zero(gs.qd_float, 3)
                for chunk in range((end - start + 31) // 32):
                    entry = start + 32 * chunk + lane
                    if entry < end:
                        j = factor_info.columns[entry]
                        vector = qd.Vector([work[j, a] for a in qd.static(range(3))])
                        product += factor_info.lower[entry] @ vector
                for a in qd.static(range(3)):
                    product[a] = qd.simt.subgroup.reduce_all_add(product[a])
                if lane == 0:
                    node = factor_info.order[i]
                    value = stress_state.rhs[node, i_b] * stress_info.is_free[node] / young - product
                    for a in qd.static(range(3)):
                        work[i, a] = value[a]
                qd.simt.subgroup.sync()
            for chunk in range((factor_info.order.shape[0] + 31) // 32):
                i = 32 * chunk + lane
                if i < factor_info.order.shape[0]:
                    vector = qd.Vector([work[i, a] for a in qd.static(range(3))])
                    value = factor_info.diagonal_inverse[i] @ vector
                    for a in qd.static(range(3)):
                        work[i, a] = value[a]
            qd.simt.subgroup.sync()
            for reverse in range(factor_info.order.shape[0]):
                i = factor_info.order.shape[0] - reverse - 1
                start, end = factor_info.column_start[i], factor_info.column_start[i + 1]
                product = qd.Vector.zero(gs.qd_float, 3)
                for chunk in range((end - start + 31) // 32):
                    entry = start + 32 * chunk + lane
                    if entry < end:
                        j = factor_info.rows[entry]
                        lower_entry = factor_info.transpose_entries[entry]
                        vector = qd.Vector([work[j, a] for a in qd.static(range(3))])
                        product += factor_info.lower[lower_entry].transpose() @ vector
                for a in qd.static(range(3)):
                    product[a] = qd.simt.subgroup.reduce_all_add(product[a])
                if lane == 0:
                    for a in qd.static(range(3)):
                        work[i, a] -= product[a]
                qd.simt.subgroup.sync()
            for chunk in range((factor_info.order.shape[0] + 31) // 32):
                i = 32 * chunk + lane
                if i < factor_info.order.shape[0]:
                    node = factor_info.order[i]
                    stress_state.displacement[node, i_b] = qd.Vector([work[i, a] for a in qd.static(range(3))])
    for i_b in range(stress_state.active.shape[0]):
        if qd.static(only_failed):
            stress_state.active[i_b] = gs.qd_int(not stress_state.valid[i_b])
            if not stress_state.valid[i_b]:
                stress_state.fallbacks[i_b] += 1
            stress_state.valid[i_b] = True
