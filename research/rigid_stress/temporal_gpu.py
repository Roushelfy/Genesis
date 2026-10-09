"""Device prediction, circular residual histories, sparse correction and FP32 refinement.

The compact strategy synchronizes the failed-column count for cuSPARSE's host matrix shape. Only this control metadata
crosses to the host, and its cost belongs to recovery timing. The padded strategy uses fixed shapes throughout.
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np
from scipy import sparse

from .cudss import SharedCuDSSFactor
from .sparse_gpu import EggRecoveryGPU, GPURecoveryResult, GPUWorkStatistics, SharedSparseFactor


@dataclass(frozen=True)
class FactorExport:
    L: sparse.csc_matrix
    U: sparse.csc_matrix
    perm_r: np.ndarray
    perm_c: np.ndarray


class MixedFactor:
    """Round a shared FP64 factor to FP32, optionally after an equivalent diagonal congruence scaling."""

    def __init__(self, fem, scaling: bool) -> None:
        factor = fem.factor
        scale = 1 / np.sqrt(fem.k.diagonal()[fem.free]) if scaling else np.ones(len(fem.free))
        row_scale, column_scale = scale[np.argsort(factor.perm_r)], scale[np.argsort(factor.perm_c)]
        lower = sparse.diags(row_scale) @ factor.L @ sparse.diags(1 / row_scale)
        upper = sparse.diags(row_scale) @ factor.U @ sparse.diags(column_scale)
        self.factor = SharedSparseFactor(
            FactorExport(lower, upper, factor.perm_r, factor.perm_c),
            dtype=np.float32,
            unit_lower=fem.is_unit_lower,
        )
        self.scale = cp.asarray(scale)

    def prepare(self, columns: int) -> None:
        self.factor.prepare(columns)

    def solve(self, rhs: cp.ndarray) -> cp.ndarray:
        scaled = (self.scale[:, None] * rhs).astype(np.float32)
        return self.scale[:, None] * self.factor.solve(scaled).astype(np.float64)


class MixedCuDSSFactor:
    """Factor the congruence-scaled operator in FP32; all acceptance/refinement uses the unchanged FP64 operator."""

    def __init__(self, fem, environments: int, scaling: bool) -> None:
        matrix = fem.k[fem.free][:, fem.free]
        scale = 1 / np.sqrt(matrix.diagonal()) if scaling else np.ones(len(fem.free))
        scaled_matrix = sparse.diags(scale) @ matrix @ sparse.diags(scale)
        self.factor = SharedCuDSSFactor(scaled_matrix, environments, dtype=np.float32)
        self.scale = cp.asarray(scale)

    def prepare(self, columns: int) -> None:
        self.factor.prepare(columns)

    def solve(self, rhs: cp.ndarray) -> cp.ndarray:
        scaled = (self.scale[:, None] * rhs).astype(np.float32)
        return self.scale[:, None] * self.factor.solve(scaled).astype(np.float64)


class TemporalRecoveryGPU(EggRecoveryGPU):
    def __init__(
        self,
        fem,
        environments: int,
        history: int = 8,
        rtol: float = 1e-6,
        strategy: str = "compact",
        precision: str = "64",
        scaling: bool = False,
        refinements: int = 4,
        chunk: int | None = None,
        cached_gram: bool = True,
        factor_backend: str = "spsm",
        layout: str = "history-major",
        dense_fraction: float = 0.5,
        shared_factor: SharedSparseFactor | SharedCuDSSFactor | None = None,
    ) -> None:
        if history < 0 or strategy not in ("compact", "padded", "adaptive") or precision not in ("32", "64"):
            raise ValueError("Nonnegative history/refinement and explicit compact/padded, FP32/FP64 choices required")
        if refinements < 0 or layout not in ("history-major", "dof-major") or not 0 < dense_fraction <= 1:
            raise ValueError("Nonnegative refinement count, explicit history layout and density threshold required")
        super().__init__(fem, environments, rtol=rtol, factor_backend=factor_backend, shared_factor=shared_factor)
        self.capacity, self.strategy, self.precision = history, strategy, precision
        self.refinements, self.chunk = refinements, environments if chunk is None else min(chunk, environments)
        if self.chunk < 1:
            raise ValueError("Positive correction chunk required")
        self.is_cached_gram = cached_gram
        self.layout, self.dense_fraction = layout, dense_fraction
        if precision == "64":
            self.correction_factor = self.factor
        elif factor_backend == "spsm":
            self.correction_factor = MixedFactor(fem, scaling)
        else:
            self.correction_factor = MixedCuDSSFactor(fem, environments, scaling)
        for count in range(1, self.chunk + 1):
            if count & (count - 1) == 0 or count == self.chunk:
                self.correction_factor.prepare(count)
                self.factor.prepare(count)
        if layout == "history-major":
            self.Q = cp.zeros((environments, history, fem.ndof))
            self.KQ = cp.zeros_like(self.Q)
        else:
            self.Q = cp.zeros((environments, fem.ndof, history)).transpose(0, 2, 1)
            self.KQ = cp.zeros((environments, fem.ndof, history)).transpose(0, 2, 1)
        self.gram = cp.zeros((environments, history, history))
        self.is_valid = cp.zeros((environments, history), dtype=bool)
        self.cursor = cp.zeros(environments, dtype=np.int32)
        self.frames = cp.zeros(environments, dtype=np.int32)
        self.previous_dt = cp.zeros(environments)
        self.last = cp.zeros((fem.ndof, environments), order="F")
        self.before = cp.zeros_like(self.last, order="F")
        self.environment_ids = cp.arange(environments)
        self.statistics: GPUWorkStatistics | None = None

    def coefficients(self, right: cp.ndarray, valid: cp.ndarray) -> cp.ndarray:
        gram = self.gram if self.is_cached_gram else self.KQ @ self.KQ.transpose(0, 2, 1)
        gram = cp.where(valid[:, :, None] & valid[:, None, :], gram, 0)
        eigenvalues, vectors = cp.linalg.eigh((gram + gram.transpose(0, 2, 1)) / 2)
        keep = eigenvalues > cp.maximum(eigenvalues[:, -1:], 1e-30) * 1e-12
        projected = cp.einsum("bhi,bh->bi", vectors, right)
        projected = cp.where(keep, projected / cp.maximum(eigenvalues, 1e-30), 0)
        return cp.einsum("bhi,bi->bh", vectors, projected)

    def correction(self, residual: cp.ndarray, failed: cp.ndarray, factor) -> tuple[cp.ndarray, int]:
        delta = cp.zeros_like(residual, order="F")
        if self.strategy in ("compact", "adaptive"):
            # Dynamic output shape synchronizes one count. Complete RHS/displacement data remain device-resident.
            selected = cp.flatnonzero(failed)
            count = len(selected)
            if self.strategy == "adaptive" and count >= self.dense_fraction * self.environments:
                selected, count = self.environment_ids, self.environments
        else:
            selected, count = self.environment_ids, self.environments
        solved = 0
        for lo in range(0, count, self.chunk):
            ids = selected[lo : lo + self.chunk]
            columns = len(ids)
            padded = min(1 << (columns - 1).bit_length(), self.chunk)
            packed = cp.zeros((len(self.free), padded), order="F")
            packed[:, :columns] = residual[self.free][:, ids] * failed[ids][None]
            delta[self.free[:, None], ids] = factor.solve(packed)[:, :columns]
            solved += padded
        return delta, solved

    def append(self, delta: cp.ndarray, failed: cp.ndarray) -> cp.ndarray:
        if not self.capacity:
            return cp.zeros(self.environments, dtype=bool)
        q, aq = delta.T.copy(), (self.apply_stiffness(delta)).T.copy()
        original = cp.sum(aq * aq, axis=1)
        valid = self.is_valid.copy()
        valid[self.environment_ids, self.cursor] = False
        for _ in range(2):
            coefficient = self.coefficients(cp.einsum("bhn,bn->bh", self.KQ, aq), valid)
            q -= cp.einsum("bhn,bh->bn", self.Q, coefficient)
            aq -= cp.einsum("bhn,bh->bn", self.KQ, coefficient)
        norm2 = cp.sum(aq * aq, axis=1)
        update = failed & cp.isfinite(norm2) & cp.isfinite(original) & (norm2 > cp.maximum(1e-30, original * 1e-12))
        divisor = cp.sqrt(cp.where(update, norm2, 1))[:, None]
        q, aq = q / divisor, aq / divisor
        self.Q[self.environment_ids, self.cursor] = cp.where(
            update[:, None], q, self.Q[self.environment_ids, self.cursor]
        )
        self.KQ[self.environment_ids, self.cursor] = cp.where(
            update[:, None], aq, self.KQ[self.environment_ids, self.cursor]
        )
        column = cp.einsum("bhn,bn->bh", self.KQ, self.KQ[self.environment_ids, self.cursor])
        self.gram[self.environment_ids, :, self.cursor] = cp.where(
            update[:, None], column, self.gram[self.environment_ids, :, self.cursor]
        )
        self.gram[self.environment_ids, self.cursor, :] = cp.where(
            update[:, None], column, self.gram[self.environment_ids, self.cursor, :]
        )
        self.is_valid[self.environment_ids, self.cursor] |= update
        self.cursor = cp.where(update, (self.cursor + 1) % self.capacity, self.cursor)
        return failed & ~update

    def recover(self, rhs_n: cp.ndarray, dt: float | cp.ndarray = 1.0) -> GPURecoveryResult:
        if rhs_n.shape != self.last.shape or rhs_n.dtype != np.float64:
            raise ValueError("Finite-profile recovery requires FP64 complete right-hand sides")
        timestep = cp.broadcast_to(cp.asarray(dt), (self.environments,))
        is_dt_valid = cp.isfinite(timestep) & (timestep > 0)
        ratio = timestep / cp.maximum(self.previous_dt, 1e-30)
        extrapolate = (self.frames >= 2) & is_dt_valid & (ratio >= 0.5) & (ratio <= 2)
        displacement = cp.asfortranarray(self.last + cp.where(extrapolate, ratio, 0)[None] * (self.last - self.before))
        norm = cp.linalg.norm(rhs_n, axis=0)
        active = norm > 1e-12
        displacement[:] = cp.where(active[None], displacement, 0)
        residual = rhs_n - self.apply_stiffness(displacement)
        if self.capacity:
            coefficient = self.coefficients(cp.einsum("bhn,bn->bh", self.KQ, residual.T), self.is_valid)
            displacement += cp.einsum("bhn,bh->bn", self.Q, coefficient).T
            displacement[:] = cp.where(active[None], displacement, 0)
            residual = rhs_n - self.apply_stiffness(displacement)
        allowed = cp.maximum(self.atol_n, self.rtol * norm)
        absolute = cp.linalg.norm(residual, axis=0)
        failed = (absolute > allowed) | ~cp.isfinite(absolute) | ~cp.isfinite(norm)
        delta, solved = self.correction(residual, failed, self.correction_factor)
        displacement += delta
        refinement_count = cp.zeros(self.environments, dtype=np.int32)
        if self.precision == "32":
            for _ in range(self.refinements):
                residual = rhs_n - self.apply_stiffness(displacement)
                remaining = failed & (cp.linalg.norm(residual, axis=0) > allowed)
                refined, columns = self.correction(residual, remaining, self.correction_factor)
                displacement += refined
                delta += refined
                refinement_count += remaining
                solved += columns
        residual = rhs_n - self.apply_stiffness(displacement)
        absolute = cp.linalg.norm(residual, axis=0)
        fallback = (absolute > allowed) | ~cp.isfinite(absolute)
        if self.precision == "32":
            # Recompute the solution of rejected environments, so a nonfinite approximation cannot poison correction.
            direct, columns = self.correction(rhs_n, fallback, self.factor)
            replacement = cp.where(fallback[None], direct, displacement)
            delta += cp.where(fallback[None], direct - displacement, 0)
            displacement = cp.asfortranarray(replacement)
            solved += columns
        rejected_direction = self.append(delta, failed) if solved else cp.zeros_like(failed)
        residual = rhs_n - self.apply_stiffness(displacement)
        absolute = cp.linalg.norm(residual, axis=0)
        peaks = self.peak(displacement)
        accepted = (absolute <= allowed) & cp.isfinite(absolute) & cp.isfinite(norm) & cp.isfinite(peaks)
        self.before[:] = self.last
        self.last[:] = cp.where(accepted[None], displacement, 0)
        self.frames = cp.where(accepted, self.frames + 1, 0)
        self.previous_dt = cp.where(accepted & is_dt_valid, timestep, 0)
        self.is_valid &= accepted[:, None]
        self.statistics = GPUWorkStatistics(
            failed,
            refinement_count,
            fallback if self.precision == "32" else cp.zeros_like(failed),
            rejected_direction,
            solved,
        )
        return GPURecoveryResult(
            peaks, displacement, absolute / cp.maximum(norm, self.atol_n), absolute, accepted, norm
        )

    def reset(self, environments: cp.ndarray | None = None) -> None:
        ids = self.environment_ids if environments is None else cp.asarray(environments)
        self.last[:, ids], self.before[:, ids] = 0, 0
        self.Q[ids], self.KQ[ids], self.gram[ids] = 0, 0, 0
        self.is_valid[ids] = False
        self.cursor[ids], self.frames[ids], self.previous_dt[ids] = 0, 0, 0
