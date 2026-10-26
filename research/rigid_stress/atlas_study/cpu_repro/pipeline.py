"""Evaluate current contact loads and complete sampled shell outputs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .loads import Source, assemble_sources
from .model import Mother, Space


@dataclass
class PipelineStages:
    source_seconds: float = 0.0
    assignment_seconds: float = 0.0
    preparation_seconds: float = 0.0
    projection_seconds: float = 0.0
    solve_seconds: float = 0.0
    merge_seconds: float = 0.0
    stress_seconds: float = 0.0
    recovery_seconds: float = 0.0
    output_seconds: float = 0.0


@dataclass
class PipelineResult:
    values: np.ndarray
    local_stress: np.ndarray | None
    displacement: np.ndarray | None
    rhs_groups: int
    contributions: int
    stages: PipelineStages
    evaluated_blocks: np.ndarray | None = None


def sampled_maximum(local_stress: np.ndarray) -> np.ndarray:
    """Reduce von Mises stress over both surfaces and all quadrature points."""
    batch = local_stress.shape[1]
    stress = local_stress.reshape((-1, 3, batch))
    x, y, shear = stress[:, 0], stress[:, 1], stress[:, 2]
    squared = x * x - x * y + y * y + 3.0 * shear * shear
    return np.sqrt(np.maximum(squared, 0.0)).max(axis=0)


def full_tensor_basis(mother: Mother) -> np.ndarray:
    """Prepare the fixed map from local stress components to body-frame tensors."""
    first, second = mother.model.axes[..., 0], mother.model.axes[..., 1]
    return np.stack(
        (
            first[..., :, None] * first[..., None, :],
            second[..., :, None] * second[..., None, :],
            first[..., :, None] * second[..., None, :]
            + second[..., :, None] * first[..., None, :],
        ),
        axis=2,
    )


def tensor_output(mother: Mother, local_stress: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Recover the body-frame symmetric tensor at every authored sample."""
    batch = local_stress.shape[1]
    stress = local_stress.reshape((2, len(mother.model.quads), 4, 3, batch))
    if basis is None:
        basis = full_tensor_basis(mother)
    return np.einsum("egkij,segkb->segijb", basis, stress).reshape((-1, 9, batch))


def checksum(values: np.ndarray) -> tuple[float, float]:
    """Summarize an already evaluated output outside its timed call."""
    return float(np.sum(values)), float(np.linalg.norm(values))


def source_columns(cases: Sequence[Sequence[Source]]) -> tuple[tuple[Source, ...], np.ndarray]:
    """Flatten source ownership while retaining empty environments."""
    sources = tuple(source for case in cases for source in case)
    environments = np.repeat(np.arange(len(cases)), [len(case) for case in cases])
    return sources, environments


def aggregate_environments(raw: np.ndarray, environments: np.ndarray, batch: int) -> np.ndarray:
    force = np.zeros((raw.shape[0], batch))
    for i_env in range(batch):
        columns = np.flatnonzero(environments == i_env)
        if columns.size:
            force[:, i_env] = raw[:, columns].sum(axis=1)
    return force


class Pipeline:
    """Evaluate a fixed mother and optional feature bank on changing sources."""

    def __init__(
        self,
        mother: Mother,
        spaces: Sequence[Space] = (),
        selector: Callable[[Source, str], tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.mother = mother
        self.spaces = tuple(spaces)
        self.selector = selector
        self.tensor_basis = full_tensor_basis(mother)

    def run(
        self,
        cases: Sequence[Sequence[Source]],
        strategy: str = "mother",
        output: str = "full",
        order: int = 8,
        response=None,
        profile: bool = False,
        recovery_mode: str = "lift",
        rhs_mode: str = "balanced",
    ) -> PipelineResult:
        if output not in ("full", "max", "local"):
            raise ValueError("output must be full, max or local")
        stages = PipelineStages()
        if recovery_mode not in ("lift", "independent"):
            raise ValueError("recovery_mode must be lift or independent")
        if rhs_mode not in ("balanced", "projected"):
            raise ValueError("rhs_mode must be balanced or projected")
        begin = perf_counter() if profile else 0.0
        batch = len(cases)
        if batch < 1:
            raise ValueError("At least one environment is required")
        sources, environments = source_columns(cases)
        raw = assemble_sources(self.mother, sources, order=order, audit=False)
        if profile:
            now = perf_counter()
            stages.source_seconds = now - begin
            begin = now

        displacement = None
        contributions = 0
        rhs_groups = 0
        evaluated_blocks = None
        if strategy == "mother":
            force = aggregate_environments(raw, environments, batch)
            active = np.flatnonzero(np.any(force != 0.0, axis=0))
            balanced = self.mother.balance(force[:, active])
            if profile:
                now = perf_counter()
                stages.preparation_seconds = now - begin
                begin = now
            displacement = np.zeros((self.mother.n_dof, batch))
            active_displacement = self.mother.solve(balanced) if active.size else None
            if active.size:
                displacement[:, active] = active_displacement
            if profile:
                stress_begin = perf_counter()
                stages.solve_seconds = stress_begin - begin
            local_stress = np.zeros((self.mother.S.shape[0], batch))
            if active.size:
                local_stress[:, active] = self.mother.S @ active_displacement
            if profile:
                stages.stress_seconds = perf_counter() - stress_begin
            rhs_groups = len(active)
            contributions = len(sources)
        elif strategy in ("response", "screen"):
            if response is None:
                raise ValueError("A response operator is required")
            force = aggregate_environments(raw, environments, batch)
            if profile:
                now = perf_counter()
                stages.preparation_seconds = now - begin
                begin = now
            if strategy == "screen":
                if output != "max":
                    raise ValueError("Conservative screening requires max output")
                maximum = response.apply(force, use_history=False)
                values = maximum.maxima
                evaluated_blocks = maximum.evaluated_blocks
                local_stress = None
            else:
                local_stress = response.apply(force)
            rhs_groups = int(np.count_nonzero(np.any(force != 0.0, axis=0)))
            contributions = len(sources)
        else:
            if self.selector is None or not self.spaces:
                raise ValueError("Feature recovery requires spaces and a selector")
            assignments = []
            for i_source, source in enumerate(sources):
                models, weights = self.selector(source, strategy)
                for i_model, weight in zip(models, weights, strict=True):
                    assignments.append((int(i_model), int(environments[i_source]), i_source, weight))
            contributions = len(assignments)
            if profile:
                now = perf_counter()
                stages.assignment_seconds = now - begin
                begin = now
            balanced = self.mother.balance(raw) if rhs_mode == "balanced" else raw
            grouped = []
            for i_model in sorted({item[0] for item in assignments}):
                selected = [item for item in assignments if item[0] == i_model]
                envs = np.array(sorted({item[1] for item in selected}), dtype=int)
                force = np.zeros((self.mother.n_dof, len(envs)))
                for _, i_env, i_source, weight in selected:
                    slot = np.searchsorted(envs, i_env)
                    force[:, slot] += weight * balanced[:, i_source]
                grouped.append((i_model, envs, force))
                rhs_groups += len(envs)
            if profile:
                now = perf_counter()
                stages.preparation_seconds = now - begin
                begin = now
            if recovery_mode == "lift":
                displacement = np.zeros((self.mother.n_dof, batch))
                for i_model, envs, force in grouped:
                    space = self.spaces[i_model]
                    projection_begin = perf_counter() if profile else 0.0
                    projected = space.project(force) if rhs_mode == "balanced" else space.project_raw(force)
                    solve_begin = perf_counter() if profile else 0.0
                    reduced = space.solve_projected(projected)
                    merge_begin = perf_counter() if profile else 0.0
                    displacement[:, envs] += space.lift(reduced)
                    if profile:
                        stages.projection_seconds += solve_begin - projection_begin
                        stages.solve_seconds += merge_begin - solve_begin
                        stages.merge_seconds += perf_counter() - merge_begin
                stress_begin = perf_counter() if profile else 0.0
                local_stress = self.mother.S @ displacement
                if profile:
                    stages.stress_seconds = perf_counter() - stress_begin
            else:
                local_stress = np.zeros((self.mother.S.shape[0], batch))
                for i_model, envs, force in grouped:
                    space = self.spaces[i_model]
                    projected = space.project(force) if rhs_mode == "balanced" else space.project_raw(force)
                    local_stress[:, envs] += space.stress(space.solve_projected(projected))

        if profile:
            now = perf_counter()
            stages.recovery_seconds = now - begin
            begin = now
        if strategy == "screen":
            pass
        elif output == "full":
            values = tensor_output(self.mother, local_stress, self.tensor_basis)
        elif output == "max":
            values = sampled_maximum(local_stress)
        else:
            values = local_stress
        if profile:
            stages.output_seconds = perf_counter() - begin
        return PipelineResult(values, local_stress, displacement, rhs_groups, contributions, stages, evaluated_blocks)
