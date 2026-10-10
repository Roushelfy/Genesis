"""Shared exact responses for complete exterior nodal loads and centrifugal fields."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, DataKind, V

from .data import StressInfo, StressState
from .factor import StressFactor


@dataclass(frozen=True)
class StressSurfaceInverseInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    nodes: qd.Tensor
    force_blocks: qd.Tensor
    centrifugal: qd.Tensor


class StressSurfaceInverse:
    def __init__(
        self,
        stress_info: StressInfo,
        factor: StressFactor,
        create_state: Callable[[int], StressState],
        nodes: np.ndarray,
    ):
        n_nodes, n_surface = stress_info.vertices.shape[0], len(nodes)
        self.info = StressSurfaceInverseInfo(
            nodes=V(dtype=gs.qd_int, shape=(n_surface,)),
            force_blocks=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes, n_surface)),
            centrifugal=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
        )
        self.info.nodes.from_numpy(nodes.astype(gs.np_int, copy=False))
        temporary = create_state(3 * n_surface + 6)
        kernel_surface_unit_loads(temporary, stress_info, self.info)
        factor.solve(1.0, temporary, stress_info)
        kernel_surface_store(temporary, self.info)
        qd.sync()

    def apply(self, young: float, omega: qd.Tensor, state: StressState, packed: bool = True) -> None:
        if packed and gs.backend == gs.cuda:
            kernel_surface_pack(state, self.info)
            kernel_surface_apply_packed(young, omega, state, self.info)
        else:
            kernel_surface_apply(young, omega, state, self.info)


@qd.kernel
def kernel_surface_pack(stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo):
    for i_b in range(stress_state.active.shape[0]):
        count = 0
        for j in range(surface_inverse_info.nodes.shape[0]):
            node = surface_inverse_info.nodes[j]
            force = stress_state.force[node, i_b]
            if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
                stress_state.boundary_columns[count, i_b] = j
                count += 1
        stress_state.boundary_count[i_b] = count


@qd.kernel(graph=True)
def kernel_surface_apply_packed(
    young: float, omega: qd.Tensor, stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo
):
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        value = surface_inverse_info.centrifugal[i_n] @ terms
        for slot in range(stress_state.boundary_count[i_b]):
            j = stress_state.boundary_columns[slot, i_b]
            node = surface_inverse_info.nodes[j]
            value += surface_inverse_info.force_blocks[i_n, j] @ stress_state.force[node, i_b]
        stress_state.displacement[i_n, i_b] = value / young
    for i_b in range(stress_state.active.shape[0]):
        stress_state.active[i_b] = 1
        stress_state.valid[i_b] = True


@qd.kernel(graph=True)
def kernel_surface_unit_loads(
    stress_state: StressState, stress_info: StressInfo, surface_inverse_info: StressSurfaceInverseInfo
):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        n_surface = surface_inverse_info.nodes.shape[0]
        load = qd.Vector.zero(gs.qd_float, 3)
        if i_b < 3 * n_surface:
            j_n = surface_inverse_info.nodes[i_b // 3]
            if i_n == j_n:
                load[i_b % 3] = 1.0
        else:
            for a in qd.static(range(3)):
                load[a] = stress_info.centrifugal[i_n][a, i_b - 3 * n_surface]
        stress_state.rhs[i_n, i_b] = load
        wrench = stress_info.modes[i_n].transpose() @ load
        for a in qd.static(range(6)):
            qd.atomic_add(stress_state.wrench[i_b][a], wrench[a])
    for i_n, i_b in qd.ndrange(stress_info.vertices.shape[0], stress_state.active.shape[0]):
        acceleration = stress_info.gram_inverse[None] @ stress_state.wrench[i_b]
        stress_state.rhs[i_n, i_b] -= stress_info.mass_modes[i_n] @ acceleration


@qd.kernel
def kernel_surface_store(stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo):
    for i, j in qd.ndrange(surface_inverse_info.force_blocks.shape[0], surface_inverse_info.nodes.shape[0]):
        block = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a, b in qd.static(qd.ndrange(3, 3)):
            block[a, b] = stress_state.displacement[i, 3 * j + b][a]
        surface_inverse_info.force_blocks[i, j] = block
    for i in range(surface_inverse_info.centrifugal.shape[0]):
        block = qd.Matrix.zero(gs.qd_float, 3, 6)
        for a, b in qd.static(qd.ndrange(3, 6)):
            block[a, b] = stress_state.displacement[i, 3 * surface_inverse_info.nodes.shape[0] + b][a]
        surface_inverse_info.centrifugal[i] = block


@qd.kernel(graph=True)
def kernel_surface_apply(
    young: float, omega: qd.Tensor, stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo
):
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        value = surface_inverse_info.centrifugal[i_n] @ terms
        for j in range(surface_inverse_info.nodes.shape[0]):
            j_n = surface_inverse_info.nodes[j]
            value += surface_inverse_info.force_blocks[i_n, j] @ stress_state.force[j_n, i_b]
        stress_state.displacement[i_n, i_b] = value / young
    for i_b in range(stress_state.active.shape[0]):
        stress_state.active[i_b] = 1
        stress_state.valid[i_b] = True
