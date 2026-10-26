"""Native boundary-load responses in a reduced feature displacement space."""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_MAT, DataKind

from .data import StressInfo, StressState, create_stress_state
from .feature_operator import FeatureOperator, FeatureTransformInfo


@dataclass(frozen=True)
class FeatureBoundaryResponseInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    blocks: qd.Tensor


class FeatureBoundaryResponse:
    """Apply ``K_i^-1 T_i.T R E_boundary`` to grouped raw boundary loads.

    Boundary indices name mother three-component blocks, including both shell
    translation and rotation blocks. Inertia relief belongs to the mother;
    ``projected_relief`` is its relief map projected by the masked transform.
    Centrifugal recovery is prepared and applied separately by the atlas.
    """

    def __init__(
        self,
        operator: FeatureOperator,
        mother_info: StressInfo,
        boundary_nodes: qd.Tensor,
        projected_relief: qd.Tensor,
        max_bytes: int,
    ):
        if gs.qd_float != qd.f64:
            gs.raise_exception("Feature boundary responses require Genesis precision='64'.")
        n_reduced = operator.info.vertices.shape[0]
        if len(boundary_nodes.shape) != 1 or boundary_nodes.shape[0] < 1:
            gs.raise_exception("Feature boundary responses require a nonempty vector of mother block indices.")
        if operator.transform.row_start.shape != (mother_info.vertices.shape[0] + 1,):
            gs.raise_exception("Feature boundary responses require the operator's common mother model.")
        if projected_relief.shape != (n_reduced,) or projected_relief.element_shape != (3, 6):
            gs.raise_exception("Feature boundary responses require one projected 3-by-6 relief map per reduced block.")
        n_boundary = boundary_nodes.shape[0]
        self.required_bytes = n_reduced * n_boundary * 9 * 8
        if self.required_bytes > max_bytes:
            gs.raise_exception(
                f"The feature boundary response needs {self.required_bytes} bytes, exceeding "
                f"inverse_max_bytes={max_bytes}. Choose atlas_solver='factor' or increase the budget."
            )
        self.info = FeatureBoundaryResponseInfo(blocks=V_MAT(3, 3, dtype=gs.qd_float, shape=(n_reduced, n_boundary)))
        temporary = create_stress_state(n_reduced, 3 * n_boundary)
        kernel_feature_boundary_unit_loads(boundary_nodes, projected_relief, temporary, mother_info, operator.transform)
        operator.factor.solve(1.0, temporary, operator.info)
        kernel_feature_boundary_store(temporary, self.info)
        qd.sync()

    def apply(
        self,
        young: float,
        i_group: int,
        forces: qd.Tensor,
        state: StressState,
        only_failed: bool = True,
    ) -> None:
        """Overwrite selected reduced solutions using nonzero raw group-load columns."""
        n_reduced, n_boundary = self.info.blocks.shape
        n_envs = state.active.shape[0]
        if (
            len(forces.shape) != 3
            or forces.shape[0] != n_boundary
            or forces.shape[2] != n_envs
            or not 0 <= i_group < forces.shape[1]
        ):
            gs.raise_exception("Feature response loads must have shape (boundary blocks, groups, environments).")
        if state.rhs.shape != (n_reduced, n_envs) or state.displacement.shape != state.rhs.shape:
            gs.raise_exception("Feature response solutions must match the reduced operator and load environments.")
        if state.boundary_columns.shape != (n_boundary, n_envs):
            gs.raise_exception(
                "Feature response states need one boundary-column slot per boundary block and environment."
            )
        kernel_feature_boundary_pack(i_group, forces, state, only_failed)
        kernel_feature_boundary_apply(young, i_group, forces, state, self.info, only_failed)


@qd.kernel
def kernel_feature_boundary_unit_loads(
    boundary_nodes: qd.Tensor,
    projected_relief: qd.Tensor,
    state: StressState,
    mother_info: StressInfo,
    transform_info: FeatureTransformInfo,
):
    for i_r, i_c in qd.ndrange(state.rhs.shape[0], state.active.shape[0]):
        i_n = boundary_nodes[i_c // 3]
        relief = projected_relief[i_r] @ mother_info.modes[i_n].transpose()
        state.rhs[i_r, i_c] = -relief[:, i_c % 3]
    for i_boundary in range(boundary_nodes.shape[0]):
        i_n = boundary_nodes[i_boundary]
        for i_entry in range(transform_info.row_start[i_n], transform_info.row_start[i_n + 1]):
            i_r = transform_info.columns[i_entry]
            # The stored transform already masks mother pins and reduced pins.
            for a, b in qd.static(qd.ndrange(3, 3)):
                state.rhs[i_r, 3 * i_boundary + a][b] += transform_info.values[i_entry][a, b]


@qd.kernel
def kernel_feature_boundary_store(state: StressState, response_info: FeatureBoundaryResponseInfo):
    for i_r, i_boundary in qd.ndrange(response_info.blocks.shape[0], response_info.blocks.shape[1]):
        value = qd.Matrix.zero(gs.qd_float, 3, 3)
        for a, b in qd.static(qd.ndrange(3, 3)):
            value[a, b] = state.displacement[i_r, 3 * i_boundary + b][a]
        response_info.blocks[i_r, i_boundary] = value


@qd.kernel(graph=True)
def kernel_feature_boundary_pack(i_group: int, forces: qd.Tensor, state: StressState, only_failed: qd.template()):
    for i_b in range(state.active.shape[0]):
        solve_env = True
        if qd.static(only_failed):
            solve_env = not state.valid[i_b]
        if solve_env:
            count = 0
            for i_boundary in range(forces.shape[0]):
                force = forces[i_boundary, i_group, i_b]
                if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
                    state.boundary_columns[count, i_b] = i_boundary
                    count += 1
            state.boundary_count[i_b] = count


@qd.kernel(graph=True)
def kernel_feature_boundary_apply(
    young: float,
    i_group: int,
    forces: qd.Tensor,
    state: StressState,
    response_info: FeatureBoundaryResponseInfo,
    only_failed: qd.template(),
):
    for i_r, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        solve_env = True
        if qd.static(only_failed):
            solve_env = not state.valid[i_b]
        if solve_env:
            value = qd.Vector.zero(gs.qd_float, 3)
            for i_slot in range(state.boundary_count[i_b]):
                i_boundary = state.boundary_columns[i_slot, i_b]
                value += response_info.blocks[i_r, i_boundary] @ forces[i_boundary, i_group, i_b]
            state.displacement[i_r, i_b] = value / young
    for i_b in range(state.active.shape[0]):
        state.active[i_b] = 1
        if qd.static(only_failed):
            state.active[i_b] = gs.qd_int(not state.valid[i_b])
        state.valid[i_b] = True
