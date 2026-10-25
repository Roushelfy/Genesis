"""Common-gauge Galerkin coarse response and exact regional residual responses."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V

from .data import StressInfo, StressState
from .surface_inverse import StressSurfaceInverseInfo

if TYPE_CHECKING:
    from .model import StressModel


@dataclass(frozen=True)
class StressCoarseInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    basis: qd.Tensor
    projection: qd.Tensor


@dataclass(frozen=True)
class StressCoarseState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    coefficients: qd.Tensor


class StressAtlasCoarse:
    def __init__(self, model: "StressModel", n_envs: int):
        response = model.surface_inverse.info
        n_nodes, n_boundary = response.force_blocks.shape
        # Six distributed source nodes give at most eighteen coarse directions.
        # Dependent directions are removed by the native semidefinite factor.
        n_coarse = 18
        self.info = StressCoarseInfo(
            basis=V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_coarse)),
            projection=V(dtype=gs.qd_float, shape=(n_coarse, 3 * n_boundary)),
        )
        self.state = StressCoarseState(V(dtype=gs.qd_float, shape=(n_coarse, n_envs)))
        self.response = StressSurfaceInverseInfo(
            nodes=response.nodes,
            force_blocks=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_nodes, n_boundary)),
            centrifugal=response.centrifugal,
        )
        product = V_VEC(3, dtype=gs.qd_float, shape=(n_nodes, n_coarse))
        factor = V(dtype=gs.qd_float, shape=(n_coarse, n_coarse))
        kernel_basis(response, self.info)
        kernel_product(product, model.info, self.info)
        kernel_factor(product, factor, self.info)
        kernel_projection(product, factor, response, self.info)
        kernel_residual(response, self.response, self.info)

    def apply(self, state: StressState):
        kernel_coefficients(state, self.state, self.response, self.info)


@qd.kernel
def kernel_basis(response: StressSurfaceInverseInfo, coarse: StressCoarseInfo):
    for i_n, i_c in qd.ndrange(coarse.basis.shape[0], coarse.basis.shape[1]):
        j = (i_c // 3) * (response.nodes.shape[0] - 1) // (coarse.basis.shape[1] // 3 - 1)
        value = qd.Vector.zero(gs.qd_float, 3)
        for a in qd.static(range(3)):
            value[a] = response.force_blocks[i_n, j][a, i_c % 3]
        coarse.basis[i_n, i_c] = value


@qd.kernel
def kernel_product(product: qd.Tensor, info: StressInfo, coarse: StressCoarseInfo):
    for i_n, i_c in qd.ndrange(coarse.basis.shape[0], coarse.basis.shape[1]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for slot in range(info.row_start[i_n], info.row_start[i_n + 1]):
            j = info.columns[slot]
            value += info.stiffness[slot] @ coarse.basis[j, i_c]
        product[i_n, i_c] = value


@qd.kernel
def kernel_factor(product: qd.Tensor, factor: qd.Tensor, coarse: StressCoarseInfo):
    for i, j in qd.ndrange(factor.shape[0], factor.shape[1]):
        value = gs.qd_float(0.0)
        for i_n in range(coarse.basis.shape[0]):
            value += coarse.basis[i_n, i].dot(product[i_n, j])
        factor[i, j] = value
    for _ in range(1):
        scale = gs.qd_float(0.0)
        for i in range(factor.shape[0]):
            scale = qd.max(scale, factor[i, i])
        for i in range(factor.shape[0]):
            diagonal = factor[i, i]
            for k in range(i):
                diagonal -= factor[i, k] ** 2
            active = diagonal > 1e-10 * scale
            factor[i, i] = qd.sqrt(qd.max(diagonal, 0.0)) if active else 0.0
            for j in range(i + 1, factor.shape[0]):
                value = 0.5 * (factor[j, i] + factor[i, j])
                for k in range(i):
                    value -= factor[j, k] * factor[i, k]
                factor[j, i] = value / factor[i, i] if active else 0.0


@qd.kernel
def kernel_projection(
    product: qd.Tensor, factor: qd.Tensor, response: StressSurfaceInverseInfo, coarse: StressCoarseInfo
):
    for i_input in range(coarse.projection.shape[1]):
        for i_c in range(coarse.projection.shape[0]):
            value = gs.qd_float(0.0)
            for i_n in range(coarse.basis.shape[0]):
                column = qd.Vector.zero(gs.qd_float, 3)
                for a in qd.static(range(3)):
                    column[a] = response.force_blocks[i_n, i_input // 3][a, i_input % 3]
                value += product[i_n, i_c].dot(column)
            for previous in range(i_c):
                value -= factor[i_c, previous] * coarse.projection[previous, i_input]
            coarse.projection[i_c, i_input] = value / factor[i_c, i_c] if factor[i_c, i_c] > 0.0 else 0.0
        for reverse in range(coarse.projection.shape[0]):
            i_c = coarse.projection.shape[0] - 1 - reverse
            value = coarse.projection[i_c, i_input]
            for following in range(i_c + 1, coarse.projection.shape[0]):
                value -= factor[following, i_c] * coarse.projection[following, i_input]
            coarse.projection[i_c, i_input] = value / factor[i_c, i_c] if factor[i_c, i_c] > 0.0 else 0.0


@qd.kernel
def kernel_residual(source: StressSurfaceInverseInfo, target: StressSurfaceInverseInfo, coarse: StressCoarseInfo):
    for i_n, j in qd.ndrange(source.force_blocks.shape[0], source.force_blocks.shape[1]):
        value = source.force_blocks[i_n, j]
        for i_c in range(coarse.basis.shape[1]):
            for a, b in qd.static(qd.ndrange(3, 3)):
                value[a, b] -= coarse.basis[i_n, i_c][a] * coarse.projection[i_c, 3 * j + b]
        target.force_blocks[i_n, j] = value


@qd.kernel(graph=True)
def kernel_coefficients(
    state: StressState, workspace: StressCoarseState, response: StressSurfaceInverseInfo, coarse: StressCoarseInfo
):
    for i_c, i_b in qd.ndrange(coarse.projection.shape[0], state.active.shape[0]):
        value = gs.qd_float(0.0)
        for j in range(response.nodes.shape[0]):
            i_n = response.nodes[j]
            force = state.force[i_n, i_b]
            if force.norm_sqr() > 0.0:
                for a in qd.static(range(3)):
                    value += coarse.projection[i_c, 3 * j + a] * force[a]
        workspace.coefficients[i_c, i_b] = value
