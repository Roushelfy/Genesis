"""Source-local M12 selectors and environment-isolated RHS aggregation in QD.

The parity adapter retains every positive continuous/narrow contribution. A
radius outside the prepared three-millimetre material-cap bank is explicitly
invalid; native DPR coverage and fallback remain separate qualifications.
"""

from dataclasses import dataclass
from typing import ClassVar

import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V_VEC, DataKind, V

from .material_cap import MaterialCapSources, MaterialCapState, func_material_cap_unit


@dataclass(frozen=True)
class AtlasSelectionState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    weights: qd.Tensor
    angles: qd.Tensor
    source_valid: qd.Tensor
    source_contributions: qd.Tensor
    grouped_force: qd.Tensor
    active_groups: qd.Tensor
    environment_valid: qd.Tensor
    group_count: qd.Tensor
    contribution_count: qd.Tensor
    invalid_source_count: qd.Tensor


class AtlasSelection:
    """Reusable selectors and raw RHS banks; common mother relief follows this."""

    def __init__(self, n_sources: int, n_environments: int, n_blocks: int, *, n_features: int = 12):
        if min(n_sources, n_environments, n_blocks, n_features) < 1:
            raise ValueError("Source, environment, block and feature counts must be positive.")
        self.state = AtlasSelectionState(
            weights=V(dtype=gs.qd_float, shape=(n_features, n_sources)),
            angles=V(dtype=gs.qd_float, shape=(n_features, n_sources)),
            source_valid=V(dtype=gs.qd_bool, shape=(n_sources,)),
            source_contributions=V(dtype=gs.qd_int, shape=(n_sources,)),
            grouped_force=V_VEC(3, dtype=gs.qd_float, shape=(n_blocks, n_features, n_environments)),
            active_groups=V(dtype=gs.qd_int, shape=(n_features, n_environments)),
            environment_valid=V(dtype=gs.qd_int, shape=(n_environments,)),
            group_count=V(dtype=gs.qd_int, shape=()),
            contribution_count=V(dtype=gs.qd_int, shape=()),
            invalid_source_count=V(dtype=gs.qd_int, shape=()),
        )

    def run(
        self,
        sources: MaterialCapSources,
        loads: MaterialCapState,
        centers: qd.Tensor,
        *,
        strategy: str = "continuous",
        support_angle: float = 0.7828297677412418,
        narrow_fraction: float = 0.12,
        maximum_radius: float = 0.003,
    ) -> AtlasSelectionState:
        strategies = {"continuous": 0, "nearest": 1, "narrow": 2}
        if strategy not in strategies:
            raise ValueError("Selection strategy must be continuous, nearest or narrow.")
        if support_angle <= 0.0 or narrow_fraction <= 0.0 or maximum_radius < 0.0:
            raise ValueError("Positive support/transition and a nonnegative prepared radius are required.")
        if centers.shape[0] != self.state.weights.shape[0] or sources.radius.shape[0] != self.state.weights.shape[1]:
            raise ValueError("Source/center sizes do not match the selection workspace.")
        if tuple(loads.raw_force.shape) != (self.state.grouped_force.shape[0], sources.radius.shape[0]):
            raise ValueError("Load block/source dimensions do not match the selection workspace.")
        kernel_atlas_selection_reset(self.state)
        kernel_atlas_selection_weights(
            sources, loads, centers, self.state, strategies[strategy], support_angle, narrow_fraction, maximum_radius
        )
        kernel_atlas_selection_group(sources, loads, self.state)
        kernel_atlas_selection_counts(self.state)
        return self.state


@qd.func
def func_atlas_wendland(distance: float):
    remainder = qd.max(1.0 - distance, 0.0)
    return remainder**4 * (1.0 + 4.0 * distance)


@qd.kernel
def kernel_atlas_selection_reset(state: AtlasSelectionState):
    for block, feature, environment in state.grouped_force:
        state.grouped_force[block, feature, environment] = qd.Vector.zero(gs.qd_float, 3)
    for feature, environment in state.active_groups:
        state.active_groups[feature, environment] = False
    for environment in state.environment_valid:
        state.environment_valid[environment] = True


@qd.kernel
def kernel_atlas_selection_weights(
    sources: MaterialCapSources,
    loads: MaterialCapState,
    centers: qd.Tensor,
    state: AtlasSelectionState,
    strategy: int,
    support_angle: float,
    narrow_fraction: float,
    maximum_radius: float,
):
    for source in range(sources.radius.shape[0]):
        direction, direction_valid = func_material_cap_unit(sources.direction[source])
        nearest, minimum = 0, gs.qd_float(qd.math.inf)
        for feature in range(state.weights.shape[0]):
            center = centers[feature]
            angle = qd.acos(qd.max(-1.0, qd.min(1.0, center.dot(direction))))
            state.angles[feature, source] = angle
            if angle < minimum:
                nearest, minimum = feature, angle
        total = gs.qd_float(0.0)
        for feature in range(state.weights.shape[0]):
            angle = state.angles[feature, source]
            weight = func_atlas_wendland(angle / support_angle)
            if strategy == 1:
                weight = qd.cast(feature == nearest, gs.qd_float)
            elif strategy == 2:
                weight *= func_atlas_wendland((angle - minimum) / (narrow_fraction * support_angle))
            state.weights[feature, source] = weight
            total += weight
        environment = sources.environment[source]
        environment_valid = environment >= 0 and environment < state.environment_valid.shape[0]
        valid = loads.valid[source] and direction_valid and environment_valid
        valid = valid and sources.radius[source] <= maximum_radius * (1.0 + 1e-12) and sources.radius[source] >= 0.0
        valid = valid and total > 0.0 and total < qd.math.inf
        count = 0
        for feature in range(state.weights.shape[0]):
            weight = gs.qd_float(0.0)
            if valid:
                weight = state.weights[feature, source] / total
                if weight > 0.0:
                    count += 1
                    # Atomic OR avoids races between sources sharing a group.
                    qd.atomic_or(state.active_groups[feature, environment], 1)
            state.weights[feature, source] = weight
        state.source_valid[source], state.source_contributions[source] = valid, count
        if environment_valid and not valid:
            qd.atomic_and(state.environment_valid[environment], 0)
        if not environment_valid:
            # An unowned source cannot be silently omitted from a supposedly valid batch.
            for i_b in range(state.environment_valid.shape[0]):
                qd.atomic_and(state.environment_valid[i_b], 0)


@qd.kernel
def kernel_atlas_selection_group(sources: MaterialCapSources, loads: MaterialCapState, state: AtlasSelectionState):
    for block, source in loads.raw_force:
        if state.source_valid[source]:
            environment = sources.environment[source]
            force = loads.raw_force[block, source]
            for feature in range(state.weights.shape[0]):
                weight = state.weights[feature, source]
                if weight > 0.0:
                    state.grouped_force[block, feature, environment] += weight * force


@qd.kernel
def kernel_atlas_selection_counts(state: AtlasSelectionState):
    for _ in range(1):
        groups, contributions, invalid = 0, 0, 0
        for feature, environment in qd.ndrange(state.active_groups.shape[0], state.active_groups.shape[1]):
            groups += qd.cast(state.active_groups[feature, environment], gs.qd_int)
        for source in range(state.source_valid.shape[0]):
            contributions += state.source_contributions[source]
            invalid += qd.cast(not state.source_valid[source], gs.qd_int)
        state.group_count[None], state.contribution_count[None], state.invalid_source_count[None] = (
            groups,
            contributions,
            invalid,
        )
