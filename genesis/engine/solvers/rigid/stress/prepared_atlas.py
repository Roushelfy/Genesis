"""Shared native feature solves with common relief and one tensor recovery."""

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_MAT, V_VEC, DataItem, DataKind, iter_data

from .bilinear_basis import FeatureMap
from .data import StressInfo, StressState, create_stress_state
from .feature_operator import FeatureOperator, FeatureTransformInfo
from .model import StressModel
from .solve import kernel_balance, kernel_direct_init, kernel_full_residual


@dataclass(frozen=True)
class PreparedFeatureInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    projected_relief: qd.Tensor


@dataclass(frozen=True)
class PreparedAtlasTimings:
    operators_and_factors: float
    relief_projection: float
    buffers: float


class PreparedAtlas:
    """Recover an aggregate in independently supplied common-mother spaces.

    The caller supplies unbalanced per-feature forces, with every source
    assigned within its environment. A single mother's inertia relief is
    projected into each space before the grouped solves. Accepted reduced
    coordinates are lifted and summed before the mother's stress scan.
    This class imposes no spatial coverage claim and performs no mother
    correction. Its numerical acceptance concerns the original reduced
    operators, with the same tolerances as exact recovery.
    """

    def __init__(
        self,
        model: StressModel,
        maps: Sequence[FeatureMap],
        n_envs: int,
        options: RigidStressOptions | None = None,
    ):
        if not maps or n_envs < 1:
            gs.raise_exception("Prepared Atlas requires nonempty feature maps and at least one environment.")
        self.model = model
        self.options = model.options.model_copy(deep=True) if options is None else options.model_copy(deep=True)
        started = time.perf_counter()
        self.operators = tuple(FeatureOperator(model.info, *mapping) for mapping in maps)
        qd.sync()
        operators_done = time.perf_counter()
        self.features = tuple(
            PreparedFeatureInfo(
                projected_relief=V_MAT(3, 6, dtype=gs.qd_float, shape=(operator.info.vertices.shape[0],))
            )
            for operator in self.operators
        )
        for operator, feature in zip(self.operators, self.features):
            kernel_project_common_relief(feature, model.info, operator.transform)
        qd.sync()
        relief_done = time.perf_counter()
        self.states = tuple(create_stress_state(operator.info.vertices.shape[0], n_envs) for operator in self.operators)
        self.omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
        self.omega.fill(0)
        self.build_timings = PreparedAtlasTimings(
            operators_done - started, relief_done - operators_done, time.perf_counter() - relief_done
        )

    def iter_info_data(self, prefix: str) -> Iterator[DataItem]:
        for i_f, (operator, feature) in enumerate(zip(self.operators, self.features)):
            yield from iter_data(operator.info, f"{prefix}.operators.{i_f}.info")
            yield from iter_data(operator.transform, f"{prefix}.operators.{i_f}.transform")
            yield from iter_data(operator.factor.info, f"{prefix}.operators.{i_f}.factor")
            yield from iter_data(feature, f"{prefix}.features.{i_f}")

    def iter_state_data(self, prefix: str) -> Iterator[DataItem]:
        for i_f, state in enumerate(self.states):
            yield from iter_data(state, f"{prefix}.states.{i_f}")

    def recover(
        self, grouped_force: qd.Tensor, parent_state: StressState, valid_environments: qd.Tensor | None = None
    ) -> None:
        """Recover fresh zero-spin loads shaped (mother blocks, features, environments).

        Load adapters must propagate their admission and coverage through
        ``valid_environments``. Omitting it declares externally validated raw
        inputs. Original reduced residuals live in ``states``; the parent
        mother residual is not evaluated by this approximate recovery.
        """
        if (
            grouped_force.shape
            != (
                self.model.info.vertices.shape[0],
                len(self.operators),
                parent_state.active.shape[0],
            )
            or parent_state.active.shape != self.omega.shape
        ):
            gs.raise_exception("Grouped forces must match the Prepared Atlas mother, feature and environment counts.")
        if valid_environments is not None and valid_environments.shape != parent_state.valid.shape:
            gs.raise_exception("Atlas load admission must have one validity value per environment.")
        kernel_sum_feature_forces(grouped_force, parent_state)
        kernel_balance(self.omega, parent_state, self.model.info, self.options.cooperative_balance)
        kernel_direct_init(parent_state)
        parent_state.displacement.fill(0)
        # This field describes a mother residual. A reduced solve does not produce it.
        parent_state.residual_norm_squared.fill(float("nan"))
        for i_f, (operator, feature, state) in enumerate(zip(self.operators, self.features, self.states)):
            kernel_group_wrench(i_f, grouped_force, state, self.model.info)
            kernel_project_group(i_f, grouped_force, state, feature, operator.transform)
            kernel_direct_init(state)
            kernel_prepare_group_solve(state)
            operator.factor.solve(self.options.young, state, operator.info, self.options.cooperative_solve, True)
            kernel_full_residual(
                self.options.young,
                self.options.tolerance,
                self.options.absolute_tolerance,
                state,
                operator.info,
                False,
                False,
            )
            operator.lift(state.displacement, parent_state.displacement, add=True)
            kernel_merge_group_acceptance(state, parent_state)
        if valid_environments is not None:
            kernel_admit_environments(valid_environments, parent_state)
        self.model.evaluate_stress(self.options, parent_state)


@qd.kernel
def kernel_project_common_relief(
    feature_info: PreparedFeatureInfo, mother_info: StressInfo, transform_info: FeatureTransformInfo
):
    for i_r in range(feature_info.projected_relief.shape[0]):
        value = qd.Matrix.zero(gs.qd_float, 3, 6)
        for i_column in range(transform_info.column_start[i_r], transform_info.column_start[i_r + 1]):
            i_n = transform_info.rows[i_column]
            i_entry = transform_info.transpose_entries[i_column]
            value += transform_info.values[i_entry].transpose() @ mother_info.relief[i_n]
        feature_info.projected_relief[i_r] = value


@qd.kernel(graph=True)
def kernel_sum_feature_forces(grouped_force: qd.Tensor, parent_state: StressState):
    for i_n, i_b in qd.ndrange(grouped_force.shape[0], grouped_force.shape[2]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_f in range(grouped_force.shape[1]):
            value += grouped_force[i_n, i_f, i_b]
        parent_state.force[i_n, i_b] = value


@qd.kernel(graph=True)
def kernel_group_wrench(i_f: int, grouped_force: qd.Tensor, state: StressState, mother_info: StressInfo):
    for i_b in range(state.active.shape[0]):
        state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_n, i_b in qd.ndrange(grouped_force.shape[0], grouped_force.shape[2]):
        value = mother_info.modes[i_n].transpose() @ grouped_force[i_n, i_f, i_b]
        for a in qd.static(range(6)):
            qd.atomic_add(state.wrench[i_b][a], value[a])


@qd.kernel(graph=True)
def kernel_project_group(
    i_f: int,
    grouped_force: qd.Tensor,
    state: StressState,
    feature_info: PreparedFeatureInfo,
    transform_info: FeatureTransformInfo,
):
    for i_r, i_b in qd.ndrange(state.rhs.shape[0], state.rhs.shape[1]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for i_column in range(transform_info.column_start[i_r], transform_info.column_start[i_r + 1]):
            i_n = transform_info.rows[i_column]
            i_entry = transform_info.transpose_entries[i_column]
            value += transform_info.values[i_entry].transpose() @ grouped_force[i_n, i_f, i_b]
        state.force[i_r, i_b] = value
        state.rhs[i_r, i_b] = value - feature_info.projected_relief[i_r] @ state.wrench[i_b]
        # An inactive/zero RHS must never lift coordinates from a previous call.
        state.displacement[i_r, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(graph=True)
def kernel_prepare_group_solve(state: StressState):
    for i_b in range(state.active.shape[0]):
        state.valid[i_b] = state.rhs_norm_squared[i_b] == 0.0


@qd.kernel(graph=True)
def kernel_merge_group_acceptance(state: StressState, parent_state: StressState):
    for i_b in range(state.active.shape[0]):
        # only_failed skips zero groups; those skips are scheduling, not fallbacks.
        state.fallbacks[i_b] = 0
        parent_state.valid[i_b] = parent_state.valid[i_b] and state.valid[i_b]


@qd.kernel(graph=True)
def kernel_admit_environments(valid_environments: qd.Tensor, parent_state: StressState):
    for i_b in range(parent_state.valid.shape[0]):
        parent_state.valid[i_b] = parent_state.valid[i_b] and valid_environments[i_b] != 0
