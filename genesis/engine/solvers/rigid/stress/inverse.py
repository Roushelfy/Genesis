"""Memory-bounded full pinned inverse, constructed and applied in Quadrants."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, DataKind

from .data import StressInfo, StressState
from .factor import StressFactor


@dataclass(frozen=True)
class StressInverseInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    blocks: qd.Tensor


class StressInverse:
    def __init__(
        self,
        stress_info: StressInfo,
        factor: StressFactor,
        create_state: Callable[[int], StressState],
        max_bytes: int,
        precision: str,
    ):
        n_nodes = stress_info.vertices.shape[0]
        required_bytes = n_nodes * n_nodes * 9 * (8 if precision == "64" else 4)
        if required_bytes > max_bytes:
            gs.raise_exception(
                f"The full stress inverse needs {required_bytes} bytes, exceeding inverse_max_bytes={max_bytes}. "
                "Choose the sparse direct method or explicitly increase the budget."
            )
        dtype = gs.qd_float if precision == "64" else qd.f32
        self.info = StressInverseInfo(blocks=V_MAT(3, 3, dtype=dtype, shape=(n_nodes, n_nodes)))
        temporary = create_state(3 * n_nodes)
        kernel_unit_loads(temporary)
        factor.solve(1.0, temporary, stress_info)
        kernel_store_inverse(temporary, self.info)
        qd.sync()

    def apply(self, young: float, state: StressState, correction: bool = False, only_failed: bool = False) -> None:
        kernel_apply_inverse(young, state, self.info, correction, only_failed)


@qd.kernel
def kernel_unit_loads(stress_state: StressState):
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        value = qd.Vector.zero(gs.qd_float, 3)
        if i_n == i_b // 3:
            value[i_b % 3] = 1.0
        stress_state.rhs[i_n, i_b] = value


@qd.kernel
def kernel_store_inverse(stress_state: StressState, inverse_info: StressInverseInfo):
    for i, j in qd.ndrange(inverse_info.blocks.shape[0], inverse_info.blocks.shape[1]):
        block = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a, b in qd.static(qd.ndrange(3, 3)):
            block[a, b] = stress_state.displacement[i, 3 * j + b][a]
        inverse_info.blocks[i, j] = block


@qd.kernel(graph=True)
def kernel_apply_inverse(
    young: float,
    stress_state: StressState,
    inverse_info: StressInverseInfo,
    correction: qd.template(),
    only_failed: qd.template(),
):
    func_apply_inverse(young, stress_state, inverse_info, correction, only_failed)


@qd.func
def func_apply_inverse(
    young: float,
    stress_state: StressState,
    inverse_info: StressInverseInfo,
    correction: qd.template(),
    only_failed: qd.template(),
):
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        if qd.static(not correction and not only_failed) or not stress_state.valid[i_b]:
            value = qd.Vector.zero(gs.qd_float, 3)
            for j in range(inverse_info.blocks.shape[1]):
                source = stress_state.rhs[j, i_b]
                if qd.static(correction):
                    source = stress_state.residual[j, i_b]
                value += inverse_info.blocks[i_n, j] @ source
            if qd.static(correction):
                stress_state.displacement[i_n, i_b] += value / young
            else:
                stress_state.displacement[i_n, i_b] = value / young
    for i_b in range(stress_state.active.shape[0]):
        stress_state.active[i_b] = 1
        if qd.static(correction) and not stress_state.valid[i_b]:
            stress_state.corrections[i_b] += 1
        if qd.static(correction or only_failed):
            stress_state.active[i_b] = gs.qd_int(not stress_state.valid[i_b])
        stress_state.valid[i_b] = True
