"""Independent circular CPU histories with rank-safe residual projections and exact direct correction."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .mechanics import P2Shell
from .peak_cpu import P2Peak


@dataclass(frozen=True)
class StepStatistics:
    failed: int
    active: int
    solved_columns: int
    rejected_directions: int


class CachedHistory:
    def __init__(self, matrix: sparse.spmatrix, capacity: int, cached_gram: bool = True) -> None:
        self.K, self.capacity, self.cached_gram = matrix, capacity, cached_gram
        self.Q = np.zeros((matrix.shape[0], capacity), order="F")
        self.KQ = np.zeros_like(self.Q, order="F")
        self.gram = np.zeros((capacity, capacity))
        self.order: list[int] = []
        self.rejected_directions = 0
        self.gram_rebuilds = 0
        self.gram_column_updates = 0

    def coefficients(self, right: np.ndarray, ids: np.ndarray) -> np.ndarray:
        if not len(ids):
            return np.empty(0)
        if self.cached_gram:
            gram = self.gram[np.ix_(ids, ids)]
        else:
            gram = self.KQ[:, ids].T @ self.KQ[:, ids]
            self.gram_rebuilds += 1
        # Truncate a genuinely deficient subspace instead of failing or adding an arbitrary diagonal ridge.
        eigenvalues, vectors = np.linalg.eigh((gram + gram.T) / 2)
        keep = eigenvalues > max(float(eigenvalues.max()), 1e-30) * 1e-12
        if not keep.any():
            return np.zeros(len(ids))
        basis = vectors[:, keep]
        return basis @ ((basis.T @ right) / eigenvalues[keep])

    def project(self, displacement: np.ndarray, residual: np.ndarray) -> None:
        ids = np.asarray(self.order, dtype=int)
        if len(ids):
            displacement += self.Q[:, ids] @ self.coefficients(self.KQ[:, ids].T @ residual, ids)

    def append_direction(self, delta: np.ndarray) -> None:
        if not self.capacity:
            return
        q, aq = delta.copy(), self.K @ delta
        original = float(aq @ aq)
        if not np.isfinite(original) or original <= 1e-30:
            self.rejected_directions += 1
            return
        # Do not evict a useful old slot until a new, independent direction has passed both orthogonalizations.
        slot = self.order[0] if len(self.order) == self.capacity else next(
            i for i in range(self.capacity) if i not in self.order
        )
        ids = np.asarray([i for i in self.order if i != slot], dtype=int)
        for _ in range(2):
            coefficient = self.coefficients(self.KQ[:, ids].T @ aq, ids)
            q -= self.Q[:, ids] @ coefficient
            aq -= self.KQ[:, ids] @ coefficient
        norm2 = float(aq @ aq)
        if not np.isfinite(norm2) or norm2 <= 1e-12 * original:
            self.rejected_directions += 1
            return
        self.Q[:, slot], self.KQ[:, slot] = q / np.sqrt(norm2), aq / np.sqrt(norm2)
        column = self.KQ.T @ self.KQ[:, slot]
        self.gram[:, slot], self.gram[slot] = column, column
        if slot in self.order:
            self.order.remove(slot)
        self.order.append(slot)
        self.gram_column_updates += 1


class BatchRecovery:
    def __init__(
        self, fem: P2Shell, peak: P2Peak, environments: int, capacity: int, rtol: float,
        direct: bool = False, cached_gram: bool = True,
    ) -> None:
        self.fem, self.peak, self.environments = fem, peak, environments
        self.K, self.capacity, self.rtol, self.direct = fem.k.tocsr(), capacity, rtol, direct
        self.cached_gram = cached_gram
        self.history = [CachedHistory(self.K, capacity, cached_gram) for _ in range(environments)]
        self.last: np.ndarray | None = None
        self.before: np.ndarray | None = None
        self.frames = np.zeros(environments, dtype=int)
        self.previous_dt = np.zeros(environments)

    def step(self, rhs: np.ndarray, dt: float | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray, StepStatistics]:
        if self.fem.factor is None:
            raise ValueError("CPU recovery requires an explicitly constructed FP64 factor")
        if rhs.shape != (self.fem.ndof, self.environments) or not np.isfinite(rhs).all():
            raise ValueError("Finite complete RHS for every environment required")
        timestep = np.broadcast_to(np.asarray(dt, dtype=float), (self.environments,)).copy()
        valid_dt = np.isfinite(timestep) & (timestep > 0)
        timestep[~valid_dt] = 0
        displacement = np.zeros_like(rhs, order="F")
        if not self.direct and self.last is not None:
            displacement[:] = self.last
            ratio = np.divide(timestep, self.previous_dt, out=np.zeros_like(timestep), where=self.previous_dt > 0)
            extrapolate = (self.frames >= 2) & valid_dt & (ratio >= 0.5) & (ratio <= 2)
            if self.before is not None:
                displacement[:, extrapolate] += ratio[extrapolate] * (
                    self.last[:, extrapolate] - self.before[:, extrapolate]
                )
        norm = np.linalg.norm(rhs, axis=0)
        active = norm > 1e-12
        displacement[:, ~active] = 0
        residual = rhs - self.K @ displacement
        if not self.direct:
            for environment in np.flatnonzero(active):
                self.history[environment].project(displacement[:, environment], residual[:, environment])
            residual = rhs - self.K @ displacement
        allowed = np.maximum(1e-11, self.rtol * norm)
        failed = active & (np.linalg.norm(residual, axis=0) > allowed)
        ids = np.flatnonzero(failed)
        if len(ids):
            delta = np.zeros((self.fem.ndof, len(ids)), order="F")
            delta[self.fem.free] = self.fem.factor.solve(np.asfortranarray(residual[self.fem.free][:, ids]))
            displacement[:, ids] += delta
            if not self.direct:
                for column, environment in enumerate(ids):
                    self.history[environment].append_direction(delta[:, column])
        peaks = np.array([self.peak(displacement[:, i])[0] for i in range(self.environments)])
        self.before, self.last = self.last, displacement.copy(order="F")
        self.frames += 1
        self.previous_dt[:] = timestep
        return displacement, peaks, StepStatistics(
            len(ids), int(active.sum()), len(ids), sum(h.rejected_directions for h in self.history)
        )

    def reset(self, environments: np.ndarray) -> None:
        ids = np.asarray(environments)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= self.environments)):
            raise ValueError("Reset IDs must be an integer vector of valid environment indices")
        for environment in ids:
            self.history[environment] = CachedHistory(self.K, self.capacity, self.cached_gram)
        if self.last is not None:
            self.last[:, ids] = 0
        if self.before is not None:
            self.before[:, ids] = 0
        self.frames[ids], self.previous_dt[ids] = 0, 0
