"""Solve independent native feature factors in one cooperative launch."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataItem, DataKind, V, iter_data

from .data import StressInfo, StressState
from .factor import StressFactorInfo
from .feature_operator import FeatureOperator


@dataclass(frozen=True)
class PackedFactorInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    node_offsets: qd.Tensor
    lower_offsets: qd.Tensor
    order: qd.Tensor
    row_start: qd.Tensor
    columns: qd.Tensor
    column_start: qd.Tensor
    rows: qd.Tensor
    transpose_entries: qd.Tensor
    lower: qd.Tensor
    diagonal_inverse: qd.Tensor
    scales: qd.Tensor
    is_free: qd.Tensor


@dataclass(frozen=True)
class PackedFactorState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    work: qd.Tensor
    active: qd.Tensor


@dataclass(frozen=True)
class PackedFactorMemory:
    """Count extra retained storage while the original feature factors remain allocated."""

    graph_bytes: int
    lower_bytes: int
    diagonal_inverse_bytes: int
    scales_bytes: int
    mask_bytes: int
    work_bytes: int
    active_bytes: int
    constant_bytes: int
    workspace_bytes: int
    extra_retained_bytes: int
    shared_bytes_per_block: int
    original_factors_retained: bool = True


class PackedStressFactor:
    """Pack native factors and solve the nonzero feature/environment groups concurrently.

    Operators and states must follow the same feature ordering and environment count. Before each solve, callers
    prepare fresh right-hand sides and set each state's validity false for nonzero groups and true for zero groups.
    This method replaces only the primary factor solve; callers retain original-operator residual acceptance.
    Unsupported devices, precision or factor sizes leave ``supported`` false for caller dispatch to the old path.
    """

    def __init__(self, operators: Sequence[FeatureOperator], states: Sequence[StressState]):
        self.operators, self.states = tuple(operators), tuple(states)
        if not self.operators or len(self.operators) != len(self.states):
            gs.raise_exception("Packed factors require equally sized nonempty operator and state sequences.")
        self.n_environments = self.states[0].valid.shape[0]
        self.node_counts = tuple(operator.info.vertices.shape[0] for operator in self.operators)
        self.lower_counts = tuple(operator.factor.info.lower.shape[0] for operator in self.operators)
        for operator, state in zip(self.operators, self.states):
            if (
                state.rhs.shape != (operator.info.vertices.shape[0], self.n_environments)
                or state.displacement.shape != state.rhs.shape
                or state.valid.shape != (self.n_environments,)
            ):
                gs.raise_exception("Packed factor states must match their operators and share an environment count.")
        self.bucket = next((size for size in (128, 256, 512, 1024, 1536, 2048) if max(self.node_counts) <= size), 0)
        self.supported = gs.backend == gs.cuda and gs.qd_float == qd.f64 and self.bucket > 0
        self.unsupported_reason = ""
        self.info = None
        self.state = None
        if not self.supported:
            self.unsupported_reason = "Packed feature factors require CUDA, FP64 and at most 2048 blocks per feature."
            return
        n_features = len(self.operators)
        node_offsets = np.r_[0, np.cumsum(self.node_counts)].astype(gs.np_int)
        lower_offsets = np.r_[0, np.cumsum(self.lower_counts)].astype(gs.np_int)
        n_nodes, n_lower = int(node_offsets[-1]), int(lower_offsets[-1])
        self.info = PackedFactorInfo(
            node_offsets=V(dtype=gs.qd_int, shape=(n_features + 1,)),
            lower_offsets=V(dtype=gs.qd_int, shape=(n_features + 1,)),
            order=V(dtype=gs.qd_int, shape=(n_nodes,)),
            row_start=V(dtype=gs.qd_int, shape=(n_nodes + n_features,)),
            columns=V(dtype=gs.qd_int, shape=(n_lower,)),
            column_start=V(dtype=gs.qd_int, shape=(n_nodes + n_features,)),
            rows=V(dtype=gs.qd_int, shape=(n_lower,)),
            transpose_entries=V(dtype=gs.qd_int, shape=(n_lower,)),
            lower=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_lower,)),
            diagonal_inverse=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes,)),
            scales=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes,)),
            is_free=V_VEC(3, dtype=gs.qd_int, shape=(n_nodes,)),
        )
        self.info.node_offsets.from_numpy(node_offsets)
        self.info.lower_offsets.from_numpy(lower_offsets)
        self.state = PackedFactorState(
            work=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, self.n_environments)),
            active=V(dtype=gs.qd_int, shape=(n_features, self.n_environments)),
        )
        for i_f, operator in enumerate(self.operators):
            kernel_pack_factor(i_f, operator.info, operator.factor.info, self.info)

    def solve(self, young: float) -> None:
        """Overwrite all grouped coordinates and match primary failed-only factor scheduling metadata."""
        if not self.supported:
            gs.raise_exception(self.unsupported_reason)
        for i_f, state in enumerate(self.states):
            kernel_packed_gather(young, i_f, state, self.info, self.state)
        kernel_packed_solve(self.info, self.state, self.bucket)
        for i_f, state in enumerate(self.states):
            kernel_packed_scatter(i_f, state, self.info, self.state)

    def iter_info_data(self, prefix: str) -> Iterator[DataItem]:
        if self.supported:
            yield from iter_data(self.info, prefix)

    def iter_state_data(self, prefix: str) -> Iterator[DataItem]:
        if self.supported:
            yield from iter_data(self.state, prefix)

    def memory_inventory(self) -> PackedFactorMemory:
        """Return exact extra tensor bytes and the declared per-block shared workspace."""
        n_features = len(self.operators)
        n_nodes, n_lower = sum(self.node_counts), sum(self.lower_counts)
        n_int_bytes = np.dtype(gs.np_int).itemsize
        graph_bytes = (2 * (n_features + 1) + n_nodes + 2 * (n_nodes + n_features) + 3 * n_lower) * n_int_bytes
        lower_bytes, diagonal_bytes, scales_bytes = 72 * n_lower, 72 * n_nodes, 24 * n_nodes
        mask_bytes = 3 * n_nodes * n_int_bytes
        work_bytes = 24 * n_nodes * self.n_environments
        active_bytes = n_features * self.n_environments * n_int_bytes
        constant_bytes = graph_bytes + lower_bytes + diagonal_bytes + scales_bytes + mask_bytes
        workspace_bytes = work_bytes + active_bytes
        if not self.supported:
            return PackedFactorMemory(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return PackedFactorMemory(
            graph_bytes,
            lower_bytes,
            diagonal_bytes,
            scales_bytes,
            mask_bytes,
            work_bytes,
            active_bytes,
            constant_bytes,
            workspace_bytes,
            constant_bytes + workspace_bytes,
            24 * self.bucket,
        )


@qd.kernel
def kernel_pack_factor(i_f: int, stress_info: StressInfo, factor_info: StressFactorInfo, packed: PackedFactorInfo):
    for i in range(factor_info.order.shape[0] + 1):
        i_row = packed.node_offsets[i_f] + i_f + i
        packed.row_start[i_row] = packed.lower_offsets[i_f] + factor_info.row_start[i]
        packed.column_start[i_row] = packed.lower_offsets[i_f] + factor_info.column_start[i]
    for i in range(factor_info.order.shape[0]):
        i_node = factor_info.order[i]
        i_packed = packed.node_offsets[i_f] + i
        packed.order[i_packed] = i_node
        packed.diagonal_inverse[i_packed] = factor_info.diagonal_inverse[i]
        packed.is_free[i_packed] = stress_info.is_free[i_node]
        scale = qd.Vector([1.0, 1.0, 1.0])
        if factor_info.scales.shape[0] > 0:
            scale = factor_info.scales[i_node]
        packed.scales[i_packed] = scale
    for i_entry in range(factor_info.lower.shape[0]):
        i_packed = packed.lower_offsets[i_f] + i_entry
        packed.lower[i_packed] = factor_info.lower[i_entry]
        packed.columns[i_packed] = factor_info.columns[i_entry]
        packed.rows[i_packed] = factor_info.rows[i_entry]
        packed.transpose_entries[i_packed] = packed.lower_offsets[i_f] + factor_info.transpose_entries[i_entry]


@qd.kernel(graph=True)
def kernel_packed_gather(
    young: float, i_f: int, source: StressState, packed: PackedFactorInfo, state: PackedFactorState
):
    for i_b in range(source.valid.shape[0]):
        state.active[i_f, i_b] = qd.cast(not source.valid[i_b], gs.qd_int)
    for i, i_b in qd.ndrange(source.rhs.shape[0], source.rhs.shape[1]):
        i_packed = packed.node_offsets[i_f] + i
        value = qd.Vector.zero(gs.qd_float, 3)
        if not source.valid[i_b]:
            i_node = packed.order[i_packed]
            value = source.rhs[i_node, i_b] * packed.is_free[i_packed] / young
            value *= packed.scales[i_packed]
        state.work[i_packed, i_b] = value


@qd.kernel(graph=True)
def kernel_packed_solve(packed: PackedFactorInfo, state: PackedFactorState, bucket: qd.template()):
    qd.loop_config(block_dim=32)
    for i_thread in range(state.active.shape[0] * state.active.shape[1] * 32):
        lane = i_thread % 32
        i_pair = i_thread // 32
        i_f, i_b = i_pair // state.active.shape[1], i_pair % state.active.shape[1]
        if state.active[i_f, i_b]:
            i_base = packed.node_offsets[i_f]
            n_nodes = packed.node_offsets[i_f + 1] - i_base
            i_row_base = i_base + i_f
            sh_work = qd.simt.block.SharedArray((bucket, 3), gs.qd_float)
            for i in range(n_nodes):
                start, end = packed.row_start[i_row_base + i], packed.row_start[i_row_base + i + 1]
                product = qd.Vector.zero(gs.qd_float, 3)
                for chunk in range((end - start + 31) // 32):
                    i_entry = start + 32 * chunk + lane
                    if i_entry < end:
                        j = packed.columns[i_entry]
                        vector = qd.Vector([sh_work[j, a] for a in qd.static(range(3))])
                        product += packed.lower[i_entry] @ vector
                for a in qd.static(range(3)):
                    product[a] = qd.simt.subgroup.reduce_all_add(product[a])
                if lane == 0:
                    value = state.work[i_base + i, i_b] - product
                    for a in qd.static(range(3)):
                        sh_work[i, a] = value[a]
                qd.simt.subgroup.sync()
            for chunk in range((n_nodes + 31) // 32):
                i = 32 * chunk + lane
                if i < n_nodes:
                    vector = qd.Vector([sh_work[i, a] for a in qd.static(range(3))])
                    value = packed.diagonal_inverse[i_base + i] @ vector
                    for a in qd.static(range(3)):
                        sh_work[i, a] = value[a]
            qd.simt.subgroup.sync()
            for reverse in range(n_nodes):
                i = n_nodes - reverse - 1
                start, end = packed.column_start[i_row_base + i], packed.column_start[i_row_base + i + 1]
                product = qd.Vector.zero(gs.qd_float, 3)
                for chunk in range((end - start + 31) // 32):
                    i_entry = start + 32 * chunk + lane
                    if i_entry < end:
                        j = packed.rows[i_entry]
                        i_lower = packed.transpose_entries[i_entry]
                        vector = qd.Vector([sh_work[j, a] for a in qd.static(range(3))])
                        product += packed.lower[i_lower].transpose() @ vector
                for a in qd.static(range(3)):
                    product[a] = qd.simt.subgroup.reduce_all_add(product[a])
                if lane == 0:
                    for a in qd.static(range(3)):
                        sh_work[i, a] -= product[a]
                qd.simt.subgroup.sync()
            for chunk in range((n_nodes + 31) // 32):
                i = 32 * chunk + lane
                if i < n_nodes:
                    value = qd.Vector([sh_work[i, a] for a in qd.static(range(3))])
                    state.work[i_base + i, i_b] = value * packed.scales[i_base + i]


@qd.kernel(graph=True)
def kernel_packed_scatter(i_f: int, target: StressState, packed: PackedFactorInfo, state: PackedFactorState):
    for i, i_b in qd.ndrange(target.displacement.shape[0], target.displacement.shape[1]):
        i_packed = packed.node_offsets[i_f] + i
        i_node = packed.order[i_packed]
        target.displacement[i_node, i_b] = state.work[i_packed, i_b]
    for i_b in range(target.valid.shape[0]):
        target.active[i_b] = state.active[i_f, i_b]
        target.fallbacks[i_b] += state.active[i_f, i_b]
        target.valid[i_b] = True
