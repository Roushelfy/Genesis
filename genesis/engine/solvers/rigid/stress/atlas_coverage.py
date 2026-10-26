"""Check discrete material-cap load support in every positively weighted space.

This check concerns the frozen mother's loaded physical nodes. It does not
certify complete native DPR footprints or spatial approximation accuracy.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import DataKind, V

from .atlas_selection import AtlasSelectionState
from .material_cap import MaterialCapSources, MaterialCapState


@dataclass(frozen=True)
class AtlasCoverageState:
    kind: ClassVar[DataKind] = DataKind.SCRATCH

    source_covered: qd.Tensor
    source_failure_count: qd.Tensor
    environment_covered: qd.Tensor
    coverage_failures: qd.Tensor


class AtlasCoverage:
    """Check coarse/local membership using the CPU's relative raw-load cutoff.

    ``source_failure_count`` counts failed positive feature contributions;
    ``coverage_failures`` counts unqualified sources, including invalid inputs.
    Invalid sources with known ownership affect only their own environment.
    A source without valid ownership conservatively invalidates all environments.
    """

    def __init__(self, coarse_index: np.ndarray, local_mask: qd.Tensor, nSources: int, nEnvs: int):
        if (
            not isinstance(coarse_index, np.ndarray)
            or coarse_index.ndim != 1
            or not len(coarse_index)
            or not np.issubdtype(coarse_index.dtype, np.integer)
            or (coarse_index < -1).any()
        ):
            gs.raise_exception("Atlas coverage coarse_index must be a nonempty integer physical-node index array.")
        if len(local_mask.shape) != 2 or local_mask.shape[0] < 1 or local_mask.shape[1] != len(coarse_index):
            gs.raise_exception("Atlas coverage local_mask must have shape (features, physical mother nodes).")
        if nSources < 1 or nEnvs < 1:
            gs.raise_exception("Atlas coverage requires positive source and environment workspace sizes.")
        self.local_mask = local_mask
        self.coarse = V(dtype=gs.qd_int, shape=(len(coarse_index),))
        self.coarse.from_numpy((coarse_index >= 0).astype(gs.np_int, copy=False))
        self.source_maximum = V(dtype=gs.qd_float, shape=(nSources,))
        self.feature_failed = V(dtype=gs.qd_int, shape=(local_mask.shape[0], nSources))
        self.state = AtlasCoverageState(
            source_covered=V(dtype=gs.qd_bool, shape=(nSources,)),
            source_failure_count=V(dtype=gs.qd_int, shape=(nSources,)),
            environment_covered=V(dtype=gs.qd_int, shape=(nEnvs,)),
            coverage_failures=V(dtype=gs.qd_int, shape=()),
        )

    def check(
        self, sources: MaterialCapSources, loads: MaterialCapState, selected: AtlasSelectionState
    ) -> AtlasCoverageState:
        n_sources = self.state.source_covered.shape[0]
        if (
            sources.radius.shape != (n_sources,)
            or sources.environment.shape != (n_sources,)
            or loads.valid.shape != (n_sources,)
            or loads.raw_force.shape != (2 * self.coarse.shape[0], n_sources)
            or selected.source_valid.shape != (n_sources,)
            or selected.weights.shape != self.feature_failed.shape
            or selected.environment_valid.shape != self.state.environment_covered.shape
        ):
            gs.raise_exception(
                "Atlas coverage source, mother, feature and environment workspace dimensions must match."
            )
        self.state.environment_covered.fill(1)
        kernel_atlas_coverage_maximum(loads, self.source_maximum)
        kernel_atlas_coverage_features(
            self.coarse, self.local_mask, loads, selected, self.source_maximum, self.feature_failed
        )
        kernel_atlas_coverage_sources(sources, loads, selected, self.feature_failed, self.state)
        kernel_atlas_coverage_count(self.state)
        return self.state


@qd.kernel
def kernel_atlas_coverage_maximum(loads: MaterialCapState, maximum: qd.Tensor):
    for source in range(maximum.shape[0]):
        value = gs.qd_float(0.0)
        for block in range(loads.raw_force.shape[0]):
            for component in qd.static(range(3)):
                value = qd.max(value, qd.abs(loads.raw_force[block, source][component]))
        maximum[source] = value


@qd.kernel
def kernel_atlas_coverage_features(
    coarse: qd.Tensor,
    local_mask: qd.Tensor,
    loads: MaterialCapState,
    selected: AtlasSelectionState,
    maximum: qd.Tensor,
    feature_failed: qd.Tensor,
):
    for feature, source in qd.ndrange(feature_failed.shape[0], feature_failed.shape[1]):
        failed = 0
        if selected.source_valid[source] and selected.weights[feature, source] > 0.0:
            cutoff = maximum[source] * 1e-14
            for node in range(coarse.shape[0]):
                if not coarse[node] and not local_mask[feature, node]:
                    loaded = False
                    for part, component in qd.static(qd.ndrange(2, 3)):
                        loaded = loaded or qd.abs(loads.raw_force[2 * node + part, source][component]) > cutoff
                    if loaded:
                        failed = 1
                        break
        feature_failed[feature, source] = failed


@qd.kernel
def kernel_atlas_coverage_sources(
    sources: MaterialCapSources,
    loads: MaterialCapState,
    selected: AtlasSelectionState,
    feature_failed: qd.Tensor,
    state: AtlasCoverageState,
):
    for source in range(state.source_covered.shape[0]):
        failures, positive = 0, 0
        for feature in range(feature_failed.shape[0]):
            failures += feature_failed[feature, source]
            positive += qd.cast(selected.weights[feature, source] > 0.0, gs.qd_int)
        environment = sources.environment[source]
        known_environment = environment >= 0 and environment < state.environment_covered.shape[0]
        covered = selected.source_valid[source] and loads.valid[source] and positive > 0 and failures == 0
        covered = covered and known_environment
        state.source_covered[source] = covered
        state.source_failure_count[source] = failures
        if not covered:
            if known_environment:
                qd.atomic_and(state.environment_covered[environment], 0)
            else:
                for other_environment in range(state.environment_covered.shape[0]):
                    qd.atomic_and(state.environment_covered[other_environment], 0)


@qd.kernel
def kernel_atlas_coverage_count(state: AtlasCoverageState):
    for _ in range(1):
        failures = 0
        for source in range(state.source_covered.shape[0]):
            failures += qd.cast(not state.source_covered[source], gs.qd_int)
        state.coverage_failures[None] = failures
