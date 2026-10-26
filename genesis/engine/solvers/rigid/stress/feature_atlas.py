"""Recover common-mother stress from locally enriched displacement spaces."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_MAT, DataItem, DataKind, V, iter_data

from .contact import StressContactState
from .data import StressAtlasScatter, StressInfo, StressState, create_stress_state
from .feature_basis import FeatureBasis
from .feature_operator import FeatureOperator, FeatureTransformInfo
from .feature_regions import FeatureRegionInfo, FeatureRegions, FeatureRegionState
from .inverse import StressInverse
from .model import StressModel
from .solve import kernel_balance, kernel_direct_init, kernel_full_residual
from .surface import StressSurfaceInfo


@dataclass(frozen=True)
class FeatureAtlasInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    centrifugal: qd.Tensor
    spin_rhs_norm_squared: qd.Tensor
    spin_residual_norm_squared: qd.Tensor


@dataclass(frozen=True)
class FeatureLoadInfo:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    relief: qd.Tensor
    local_nodes: qd.Tensor


@dataclass(frozen=True)
class FeatureAtlasState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    active: qd.Tensor
    coverage_fallbacks: qd.Tensor
    rhs_norm_squared: qd.Tensor
    residual_norm_squared: qd.Tensor
    numerical_fallbacks: qd.Tensor
    spin_invalid: qd.Tensor


@dataclass(frozen=True)
class FeatureAtlasInventory:
    kind: ClassVar[DataKind] = DataKind.CONSTANT

    blocks: qd.Tensor
    local_nodes: qd.Tensor
    stiffness_entries: qd.Tensor
    transform_entries: qd.Tensor
    factor_entries: qd.Tensor
    inverse_bytes: qd.Tensor


@dataclass(frozen=True)
class FeatureGroupMetadata:
    index: int
    blocks: int
    local_nodes: int
    stiffness_entries: int
    transform_entries: int
    factor_entries: int
    inverse_bytes: int


@dataclass(frozen=True)
class FeatureAtlasMetadata:
    mother_blocks: int
    physical_nodes: int
    coarse_nodes: int
    boundary_blocks: int
    inverse_bytes: int
    groups: tuple[FeatureGroupMetadata, ...]
    parent_residual_norm_space: Literal["source_spaces"] = "source_spaces"


@dataclass(frozen=True)
class FeatureGroup:
    operator: FeatureOperator
    info: FeatureLoadInfo
    state: StressState
    inverse: StressInverse | None


class FeatureAtlas:
    """Solve independently grouped loads in local spaces and sum displacement before evaluating stress.

    Every feature uses the mother's geometry, constraints and inertia relief. The complete mother load remains in
    the parent state for diagnostics. Numerical acceptance checks each source's declared displacement space.
    The parent's residual_norm_squared sums active source-space numerical residuals. Its rhs_norm_squared measures
    the complete mother load. Spatial approximation error requires an independent mother residual or field comparison.
    """

    def __init__(self, model: StressModel, contacts: StressContactState, options: RigidStressOptions):
        self.model = model
        self.options = options.model_copy(deep=True)
        if model.factor is None:
            gs.raise_exception("Feature stress recovery requires a prepared mother sparse factor.")
        parent_nodes, fixed_nodes = None, None
        if options.atlas_basis is not None:
            with np.load(Path(options.atlas_basis), allow_pickle=False) as asset:
                if "parent_nodes" in asset:
                    parent_nodes = asset["parent_nodes"]
                if "fixed_nodes" in asset:
                    fixed_nodes = asset["fixed_nodes"]
        self.basis = FeatureBasis(model, parent_nodes, fixed_nodes)
        self.regions = FeatureRegions(
            model,
            self.basis,
            contacts,
            options.atlas_regions,
            options.atlas_transition,
            options.atlas_core_scale,
            options.atlas_halo,
        )
        n_nodes = model.info.vertices.shape[0]
        n_envs = contacts.valid.shape[1]
        n_groups = options.atlas_regions + 1
        self.info = FeatureAtlasInfo(
            centrifugal=V_MAT(3, 6, dtype=gs.qd_float, shape=(n_nodes,)),
            spin_rhs_norm_squared=V(dtype=gs.qd_float, shape=(6,)),
            spin_residual_norm_squared=V(dtype=gs.qd_float, shape=(6,)),
        )
        self.state = FeatureAtlasState(
            active=V(dtype=gs.qd_int, shape=(n_groups, n_envs)),
            coverage_fallbacks=V(dtype=gs.qd_int, shape=(n_envs,)),
            rhs_norm_squared=V(dtype=gs.qd_float, shape=(n_envs,)),
            residual_norm_squared=V(dtype=gs.qd_float, shape=(n_envs,)),
            numerical_fallbacks=V(dtype=gs.qd_int, shape=(n_envs,)),
            spin_invalid=V(dtype=gs.qd_int, shape=(n_envs,)),
        )
        self.fallback_state = create_stress_state(n_nodes, n_envs)
        temporary = create_stress_state(n_nodes, n_envs=6)
        kernel_spin_loads(temporary, model.info)
        kernel_direct_init(temporary)
        model.factor.solve(1.0, temporary, model.info, options.cooperative_solve)
        kernel_full_residual(
            1.0, options.tolerance, options.absolute_tolerance, temporary, model.info, only_active=False
        )
        if not kernel_spin_valid(temporary):
            gs.raise_exception("Feature centrifugal preparation failed the original mother equilibrium check.")
        kernel_store_spin(temporary, self.info)

        retained_inverse_bytes = 0
        if model.inverse is not None:
            retained_inverse_bytes = n_nodes * n_nodes * 9 * (8 if options.inverse_precision == "64" else 4)
        if model.surface_inverse is not None:
            n_boundary = model.surface_inverse.info.nodes.shape[0]
            retained_inverse_bytes += (n_nodes * n_boundary * 9 + n_nodes * 18) * np.dtype(gs.np_float).itemsize
            retained_inverse_bytes += n_boundary * np.dtype(gs.np_int).itemsize
        self.groups: list[FeatureGroup] = []
        group_metadata = [
            FeatureGroupMetadata(
                index=0,
                blocks=n_nodes,
                local_nodes=0,
                stiffness_entries=model.info.columns.shape[0],
                transform_entries=0,
                factor_entries=model.factor.info.lower.shape[0],
                inverse_bytes=retained_inverse_bytes,
            )
        ]
        for i_g, local_nodes in enumerate(self.regions.local_nodes, start=1):
            local_nodes = local_nodes[self.basis.coarse_index[local_nodes] < 0]
            feature = self.basis.feature(local_nodes)
            operator = FeatureOperator(
                model.info,
                feature.row_start,
                feature.columns,
                feature.values,
                feature.n_reduced_blocks,
                feature.reduced_free,
            )
            reduced_nodes = operator.info.vertices.shape[0]
            load_info = FeatureLoadInfo(
                relief=V_MAT(3, 6, dtype=gs.qd_float, shape=(reduced_nodes,)),
                local_nodes=V(dtype=gs.qd_int, shape=(len(local_nodes),)),
            )
            load_info.local_nodes.from_numpy(local_nodes)
            kernel_project_relief(model.info, operator.transform, load_info)
            self.groups.append(
                FeatureGroup(operator, load_info, create_stress_state(reduced_nodes, n_envs), inverse=None)
            )
            inverse_bytes = reduced_nodes * reduced_nodes * 9 * 8 if options.atlas_solver == "inverse" else 0
            group_metadata.append(
                FeatureGroupMetadata(
                    index=i_g,
                    blocks=reduced_nodes,
                    local_nodes=len(local_nodes),
                    stiffness_entries=operator.info.columns.shape[0],
                    transform_entries=operator.n_transform_entries,
                    factor_entries=operator.factor.info.lower.shape[0],
                    inverse_bytes=inverse_bytes,
                )
            )
        inverse_bytes = sum(group.inverse_bytes for group in group_metadata)
        if options.atlas_solver == "inverse":
            if inverse_bytes > options.inverse_max_bytes:
                gs.raise_exception(
                    f"The feature stress inverses need {inverse_bytes} total bytes, exceeding "
                    f"inverse_max_bytes={options.inverse_max_bytes}. "
                    "Choose atlas_solver='factor' or increase the budget."
                )
            for i_g, group in enumerate(self.groups):
                inverse = StressInverse(
                    group.operator.info,
                    group.operator.factor,
                    FeatureStateAllocator(group.operator.info.vertices.shape[0]),
                    group_metadata[i_g + 1].inverse_bytes,
                    precision="64",
                )
                self.groups[i_g] = FeatureGroup(group.operator, group.info, group.state, inverse)
        self._metadata = FeatureAtlasMetadata(
            mother_blocks=n_nodes,
            physical_nodes=len(self.basis.parent_nodes),
            coarse_nodes=len(self.basis.coarse_nodes),
            boundary_blocks=self.regions.info.boundary_nodes.shape[0],
            inverse_bytes=inverse_bytes,
            groups=tuple(group_metadata),
        )
        self.inventory = FeatureAtlasInventory(
            blocks=V(dtype=gs.qd_int, shape=(n_groups,)),
            local_nodes=V(dtype=gs.qd_int, shape=(n_groups,)),
            stiffness_entries=V(dtype=gs.qd_int, shape=(n_groups,)),
            transform_entries=V(dtype=gs.qd_int, shape=(n_groups,)),
            factor_entries=V(dtype=gs.qd_int, shape=(n_groups,)),
            inverse_bytes=V(dtype=qd.i64, shape=(n_groups,)),
        )
        self.inventory.blocks.from_numpy(np.array([group.blocks for group in group_metadata], dtype=gs.np_int))
        self.inventory.local_nodes.from_numpy(
            np.array([group.local_nodes for group in group_metadata], dtype=gs.np_int)
        )
        self.inventory.stiffness_entries.from_numpy(
            np.array([group.stiffness_entries for group in group_metadata], dtype=gs.np_int)
        )
        self.inventory.transform_entries.from_numpy(
            np.array([group.transform_entries for group in group_metadata], dtype=gs.np_int)
        )
        self.inventory.factor_entries.from_numpy(
            np.array([group.factor_entries for group in group_metadata], dtype=gs.np_int)
        )
        self.inventory.inverse_bytes.from_numpy(
            np.array([group.inverse_bytes for group in group_metadata], dtype=np.int64)
        )

    @property
    def scatter(self) -> StressAtlasScatter:
        return self.regions.scatter

    def prepare(self, contacts: StressContactState, load_info: StressInfo, surface_info: StressSurfaceInfo) -> None:
        self.regions.prepare(contacts, load_info, surface_info, self.options.atlas_transition)

    def recover(self, omega: qd.Tensor, parent_state: StressState, options: RigidStressOptions) -> None:
        kernel_balance(omega, parent_state, self.model.info, options.cooperative_balance)
        kernel_direct_init(parent_state)
        kernel_atlas_begin(options.young, omega, parent_state, self.state, self.info, self.regions.state)
        coalesced = options.coalesced_residual
        if coalesced is None:
            coalesced = parent_state.active.shape[0] >= 8192
        for i_g in range(len(self.groups) + 1):
            if i_g == 0:
                state = self.fallback_state
                info = self.model.info
                factor = self.model.factor
                inverse = self.model.inverse
            else:
                group = self.groups[i_g - 1]
                state = group.state
                info = group.operator.info
                factor = group.operator.factor
                inverse = group.inverse
            kernel_group_wrench(i_g, state, self.state, self.model.info, self.regions.info, self.regions.state)
            if i_g == 0:
                kernel_fallback_rhs(state, self.model.info, self.regions.info, self.regions.state)
            else:
                kernel_group_rhs(
                    i_g,
                    state,
                    self.groups[i_g - 1].info,
                    self.groups[i_g - 1].operator.transform,
                    self.regions.info,
                    self.regions.state,
                )
            kernel_direct_init(state)
            kernel_group_begin(i_g, state, self.state)
            if inverse is None:
                factor.solve(options.young, state, info, options.cooperative_solve, only_failed=True)
                kernel_primary_factor_done(state)
            else:
                inverse.apply(options.young, state, only_failed=True)
            kernel_full_residual(
                options.young,
                options.tolerance,
                options.absolute_tolerance,
                state,
                info,
                only_active=True,
                coalesced=coalesced,
            )
            if inverse is not None:
                for _ in range(options.inverse_corrections):
                    inverse.apply(options.young, state, correction=True)
                    kernel_full_residual(
                        options.young,
                        options.tolerance,
                        options.absolute_tolerance,
                        state,
                        info,
                        only_active=True,
                        coalesced=coalesced,
                    )
                factor.solve(options.young, state, info, options.cooperative_solve, only_failed=True)
                kernel_full_residual(
                    options.young,
                    options.tolerance,
                    options.absolute_tolerance,
                    state,
                    info,
                    only_active=True,
                    coalesced=coalesced,
                )
            if i_g == 0:
                kernel_fallback_add(parent_state, state, self.state)
            else:
                kernel_group_lift(i_g, parent_state, state, self.state, self.groups[i_g - 1].operator.transform)
            kernel_group_diagnostics(i_g, parent_state, state, self.state)
        self.model.evaluate_stress(options, parent_state)

    def metadata(self) -> FeatureAtlasMetadata:
        return self._metadata

    def data(self, prefix: str) -> Iterator[DataItem]:
        yield DataItem(f"{prefix}.basis.values", self.basis.transform.values, DataKind.CONSTANT)
        yield from iter_data(self.info, f"{prefix}.info")
        yield from iter_data(self.inventory, f"{prefix}.inventory")
        yield from iter_data(self.state, f"{prefix}.state")
        yield from iter_data(self.regions.info, f"{prefix}.regions.info")
        yield from iter_data(self.regions.state, f"{prefix}.regions.state")
        yield from iter_data(self.fallback_state, f"{prefix}.fallback.state")
        for i_g, group in enumerate(self.groups, start=1):
            yield from iter_data(group.operator.info, f"{prefix}.groups.{i_g}.operator")
            yield from iter_data(group.operator.factor.info, f"{prefix}.groups.{i_g}.factor")
            yield from iter_data(group.operator.transform, f"{prefix}.groups.{i_g}.transform")
            yield from iter_data(group.info, f"{prefix}.groups.{i_g}.info")
            yield from iter_data(group.state, f"{prefix}.groups.{i_g}.state")
            if group.inverse is not None:
                yield from iter_data(group.inverse.info, f"{prefix}.groups.{i_g}.inverse")


@dataclass(frozen=True)
class FeatureStateAllocator:
    n_nodes: int

    def __call__(self, n_envs: int) -> StressState:
        return create_stress_state(self.n_nodes, n_envs)


@qd.kernel
def kernel_spin_loads(state: StressState, mother: StressInfo):
    for i_n, i_c in qd.ndrange(state.rhs.shape[0], 6):
        value = mother.centrifugal[i_n] - mother.relief[i_n] @ mother.centrifugal_wrench[None]
        state.rhs[i_n, i_c] = value[:, i_c]


@qd.kernel
def kernel_spin_valid(state: StressState) -> int:
    valid = 1
    for i_b in range(state.valid.shape[0]):
        qd.atomic_min(valid, gs.qd_int(state.valid[i_b]))
    return valid


@qd.kernel
def kernel_store_spin(state: StressState, atlas: FeatureAtlasInfo):
    for i_n in range(atlas.centrifugal.shape[0]):
        value = qd.Matrix.zero(gs.qd_float, 3, 6)
        for a, b in qd.static(qd.ndrange(3, 6)):
            value[a, b] = state.displacement[i_n, b][a]
        atlas.centrifugal[i_n] = value
    for i_c in range(6):
        atlas.spin_rhs_norm_squared[i_c] = state.rhs_norm_squared[i_c]
        atlas.spin_residual_norm_squared[i_c] = state.residual_norm_squared[i_c]


@qd.kernel
def kernel_project_relief(mother: StressInfo, transform: FeatureTransformInfo, load: FeatureLoadInfo):
    for i_r in range(load.relief.shape[0]):
        value = qd.Matrix.zero(gs.qd_float, 3, 6)
        for i_column in range(transform.column_start[i_r], transform.column_start[i_r + 1]):
            i_n = transform.rows[i_column]
            i_e = transform.transpose_entries[i_column]
            value += transform.values[i_e].transpose() @ mother.relief[i_n]
        load.relief[i_r] = value


@qd.kernel(graph=True)
def kernel_atlas_begin(
    young: float,
    omega: qd.Tensor,
    parent: StressState,
    state: FeatureAtlasState,
    atlas: FeatureAtlasInfo,
    regions: FeatureRegionState,
):
    for i_g, i_b in qd.ndrange(state.active.shape[0], state.active.shape[1]):
        state.active[i_g, i_b] = 0
    for i_b in range(parent.active.shape[0]):
        state.coverage_fallbacks[i_b] = 0
        state.rhs_norm_squared[i_b] = 0.0
        state.residual_norm_squared[i_b] = 0.0
        state.numerical_fallbacks[i_b] = 0
        w = omega[i_b]
        norm = w.dot(w)
        state.spin_invalid[i_b] = gs.qd_int(qd.math.isnan(norm) or qd.math.isinf(norm))
        parent.residual_norm_squared[i_b] = 0.0
        parent.iterations[i_b] = 0
        parent.active[i_b] = 0
    for i_c, i_b in qd.ndrange(regions.fallback.shape[0], regions.fallback.shape[1]):
        if regions.fallback[i_c, i_b]:
            qd.atomic_add(state.coverage_fallbacks[i_b], 1)
    for i_b in range(parent.active.shape[0]):
        parent.fallbacks[i_b] = state.coverage_fallbacks[i_b]
    for i_n, i_b in qd.ndrange(parent.displacement.shape[0], parent.active.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] * w[0], w[1] * w[1], w[2] * w[2], w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        value = atlas.centrifugal[i_n] @ terms / young
        parent.displacement[i_n, i_b] = value
        for a in qd.static(range(3)):
            if qd.math.isnan(value[a]) or qd.math.isinf(value[a]):
                qd.atomic_max(state.spin_invalid[i_b], 1)
    for i_b in range(parent.active.shape[0]):
        parent.valid[i_b] = state.spin_invalid[i_b] == 0


@qd.kernel(graph=True)
def kernel_group_wrench(
    i_g: int,
    state: StressState,
    atlas: FeatureAtlasState,
    mother: StressInfo,
    regions: FeatureRegionInfo,
    loads: FeatureRegionState,
):
    for i_b in range(state.active.shape[0]):
        state.wrench[i_b] = qd.Vector.zero(gs.qd_float, 6)
    for i_row, i_b in qd.ndrange(loads.force.shape[0], loads.force.shape[2]):
        i_n = regions.boundary_nodes[i_row]
        force = loads.force[i_row, i_g, i_b]
        wrench = mother.modes[i_n].transpose() @ force
        for a in qd.static(range(6)):
            qd.atomic_add(state.wrench[i_b][a], wrench[a])
        # A balanced load can excite deformation even when its resultant wrench is zero
        if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
            qd.atomic_max(atlas.active[i_g, i_b], 1)


@qd.kernel(graph=True)
def kernel_fallback_rhs(state: StressState, mother: StressInfo, regions: FeatureRegionInfo, loads: FeatureRegionState):
    for i_n, i_b in qd.ndrange(state.rhs.shape[0], state.active.shape[0]):
        value = qd.Vector.zero(gs.qd_float, 3)
        i_row = regions.force_rows[i_n]
        if i_row >= 0:
            value = loads.force[i_row, 0, i_b]
        state.force[i_n, i_b] = value
        state.rhs[i_n, i_b] = value - mother.relief[i_n] @ state.wrench[i_b]


@qd.kernel(graph=True)
def kernel_group_rhs(
    i_g: int,
    state: StressState,
    load: FeatureLoadInfo,
    transform: FeatureTransformInfo,
    regions: FeatureRegionInfo,
    loads: FeatureRegionState,
):
    for i_r, i_b in qd.ndrange(state.rhs.shape[0], state.active.shape[0]):
        value = -load.relief[i_r] @ state.wrench[i_b]
        for i_column in range(transform.column_start[i_r], transform.column_start[i_r + 1]):
            i_n = transform.rows[i_column]
            i_row = regions.force_rows[i_n]
            i_e = transform.transpose_entries[i_column]
            if i_row >= 0:
                value += transform.values[i_e].transpose() @ loads.force[i_row, i_g, i_b]
        state.rhs[i_r, i_b] = value


@qd.kernel(graph=True)
def kernel_group_begin(i_g: int, state: StressState, atlas: FeatureAtlasState):
    for i_b in range(state.active.shape[0]):
        state.active[i_b] = atlas.active[i_g, i_b]
        state.valid[i_b] = not atlas.active[i_g, i_b]
        state.iterations[i_b] = atlas.active[i_g, i_b]
        state.residual_norm_squared[i_b] = 0.0


@qd.kernel(graph=True)
def kernel_primary_factor_done(state: StressState):
    for i_b in range(state.active.shape[0]):
        # only_failed selects nonzero source groups for the initial factor solve
        state.fallbacks[i_b] = 0


@qd.kernel(graph=True)
def kernel_fallback_add(parent: StressState, state: StressState, atlas: FeatureAtlasState):
    for i_n, i_b in qd.ndrange(parent.displacement.shape[0], parent.active.shape[0]):
        if atlas.active[0, i_b]:
            parent.displacement[i_n, i_b] += state.displacement[i_n, i_b]


@qd.kernel(graph=True)
def kernel_group_lift(
    i_g: int,
    parent: StressState,
    state: StressState,
    atlas: FeatureAtlasState,
    transform: FeatureTransformInfo,
):
    for i_n, i_b in qd.ndrange(parent.displacement.shape[0], parent.active.shape[0]):
        if atlas.active[i_g, i_b]:
            value = qd.Vector.zero(gs.qd_float, 3)
            for i_e in range(transform.row_start[i_n], transform.row_start[i_n + 1]):
                i_r = transform.columns[i_e]
                value += transform.values[i_e] @ state.displacement[i_r, i_b]
            parent.displacement[i_n, i_b] += value


@qd.kernel(graph=True)
def kernel_group_diagnostics(i_g: int, parent: StressState, state: StressState, atlas: FeatureAtlasState):
    for i_b in range(parent.active.shape[0]):
        if atlas.active[i_g, i_b]:
            parent.valid[i_b] = parent.valid[i_b] and state.valid[i_b]
            parent.active[i_b] = 1
            parent.iterations[i_b] += state.iterations[i_b]
            parent.corrections[i_b] += state.corrections[i_b]
            parent.fallbacks[i_b] += state.fallbacks[i_b]
            parent.residual_norm_squared[i_b] += state.residual_norm_squared[i_b]
            atlas.rhs_norm_squared[i_b] += state.rhs_norm_squared[i_b]
            atlas.residual_norm_squared[i_b] += state.residual_norm_squared[i_b]
            atlas.numerical_fallbacks[i_b] += state.fallbacks[i_b]
