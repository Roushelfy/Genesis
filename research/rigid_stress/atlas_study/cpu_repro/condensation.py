"""Exact block condensation of one fixed projected shell stiffness."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu


Direction = Literal["coarse_first", "local_first"]


def sparse_bytes(matrix: sparse.spmatrix) -> int:
    """Count compressed sparse matrix array payload."""
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def factor_bytes(factor: object) -> int:
    """Count sparse factor numerical and permutation array payload."""
    return sparse_bytes(factor.L) + sparse_bytes(factor.U) + factor.perm_r.nbytes + factor.perm_c.nbytes


class DirectSystem:
    """Solve a reduced stiffness with symmetric diagonal equilibration."""

    def __init__(self, matrix: sparse.spmatrix):
        start = perf_counter()
        self.matrix = matrix.tocsr()
        diagonal = self.matrix.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("Projected stiffness requires positive diagonal entries.")
        self.scaling = 1.0 / np.sqrt(diagonal)
        scaled = sparse.diags(self.scaling) @ self.matrix @ sparse.diags(self.scaling)
        self.factor = splu(scaled.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                           options={"SymmetricMode": True})
        self.build_seconds = perf_counter() - start

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        scale = self.scaling if rhs.ndim == 1 else self.scaling[:, None]
        return scale * self.factor.solve(np.asfortranarray(scale * rhs))

    @property
    def payload_bytes(self) -> int:
        return self.scaling.nbytes + factor_bytes(self.factor)


class CondensedSystem:
    """Solve a coupled coarse/local system through either exact Schur direction."""

    def __init__(self, matrix: sparse.spmatrix, n_coarse: int, direction: Direction):
        start = perf_counter()
        self.matrix = matrix.tocsr()
        self.n_coarse = n_coarse
        self.direction = direction
        if not 0 < n_coarse < self.matrix.shape[0]:
            raise ValueError("Both coarse and local coordinate blocks must be nonempty.")
        if direction not in ("coarse_first", "local_first"):
            raise ValueError("Unknown condensation direction.")
        diagonal = self.matrix.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("Projected stiffness requires positive diagonal entries.")
        self.scaling = 1.0 / np.sqrt(diagonal)
        scaled = (sparse.diags(self.scaling) @ self.matrix @ sparse.diags(self.scaling)).tocsc()
        coarse = scaled[:n_coarse, :n_coarse].toarray()
        self.coupling = scaled[:n_coarse, n_coarse:].toarray()
        local = scaled[n_coarse:, n_coarse:]
        self.coarse_factor = None
        self.local_factor = None
        if direction == "coarse_first":
            self.coarse_factor = linalg.cho_factor(coarse, lower=True, check_finite=False)
            self.harmonic = linalg.cho_solve(self.coarse_factor, self.coupling, check_finite=False)
            schur = local.toarray() - self.coupling.T @ self.harmonic
        else:
            self.local_factor = splu(local, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                     options={"SymmetricMode": True})
            self.harmonic = self.local_factor.solve(np.asfortranarray(self.coupling.T))
            schur = coarse - self.coupling @ self.harmonic
        schur = 0.5 * (schur + schur.T)
        schur_diagonal = np.diag(schur)
        if np.any(schur_diagonal <= 0.0):
            raise ValueError("Exact Schur complement lost its positive diagonal.")
        self.schur_scaling = 1.0 / np.sqrt(schur_diagonal)
        equilibrated = self.schur_scaling[:, None] * schur * self.schur_scaling[None, :]
        self.schur_factor = linalg.cho_factor(equilibrated, lower=True, check_finite=False)
        self.build_seconds = perf_counter() - start

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        is_vector = rhs.ndim == 1
        loads = rhs[:, None] if is_vector else rhs
        scaled = self.scaling[:, None] * loads
        coarse_rhs, local_rhs = scaled[:self.n_coarse], scaled[self.n_coarse:]
        if self.direction == "coarse_first":
            coarse_particular = linalg.cho_solve(self.coarse_factor, coarse_rhs, check_finite=False)
            local_reduced = local_rhs - self.coupling.T @ coarse_particular
            local_solution = self._schur_solve(local_reduced)
            coarse_solution = coarse_particular - self.harmonic @ local_solution
        else:
            local_particular = self.local_factor.solve(np.asfortranarray(local_rhs))
            coarse_reduced = coarse_rhs - self.coupling @ local_particular
            coarse_solution = self._schur_solve(coarse_reduced)
            local_solution = local_particular - self.harmonic @ coarse_solution
        result = self.scaling[:, None] * np.concatenate((coarse_solution, local_solution), axis=0)
        return result[:, 0] if is_vector else result

    def _schur_solve(self, rhs: np.ndarray) -> np.ndarray:
        scale = self.schur_scaling[:, None]
        return scale * linalg.cho_solve(self.schur_factor, scale * rhs, check_finite=False)

    @property
    def payload_bytes(self) -> int:
        payload = self.scaling.nbytes + self.coupling.nbytes + self.harmonic.nbytes
        payload += self.schur_scaling.nbytes + self.schur_factor[0].nbytes
        if self.coarse_factor is not None:
            payload += self.coarse_factor[0].nbytes
        if self.local_factor is not None:
            payload += factor_bytes(self.local_factor)
        return payload


@dataclass(frozen=True)
class CouplingDropResult:
    solution: np.ndarray
    n_groups: int
    eliminated_cross_norm_relative: float


def solve_without_cross_patch_coupling(
    matrix: sparse.spmatrix, n_coarse: int, rhs: np.ndarray, local_groups: np.ndarray
) -> CouplingDropResult:
    """Evaluate the approximation obtained by dropping Schur cross-patch blocks."""
    diagonal = matrix.diagonal()
    scaling = 1.0 / np.sqrt(diagonal)
    scaled = (sparse.diags(scaling) @ matrix @ sparse.diags(scaling)).tocsc()
    coarse = scaled[:n_coarse, :n_coarse].toarray()
    coupling = scaled[:n_coarse, n_coarse:].toarray()
    local = scaled[n_coarse:, n_coarse:].toarray()
    if local_groups.shape != (local.shape[0],):
        raise ValueError("A group label is required for every local coordinate.")
    coarse_factor = linalg.cho_factor(coarse, lower=True, check_finite=False)
    harmonic = linalg.cho_solve(coarse_factor, coupling, check_finite=False)
    schur = local - coupling.T @ harmonic
    schur = 0.5 * (schur + schur.T)
    loads = rhs[:, None] if rhs.ndim == 1 else rhs
    scaled_loads = scaling[:, None] * loads
    coarse_particular = linalg.cho_solve(coarse_factor, scaled_loads[:n_coarse], check_finite=False)
    local_rhs = scaled_loads[n_coarse:] - coupling.T @ coarse_particular
    local_solution = np.zeros_like(local_rhs)
    eliminated_squared = np.linalg.norm(schur) ** 2
    groups = np.unique(local_groups)
    for group in groups:
        indices = np.flatnonzero(local_groups == group)
        block = schur[np.ix_(indices, indices)]
        eliminated_squared -= np.linalg.norm(block) ** 2
        block_scaling = 1.0 / np.sqrt(np.diag(block))
        factor = linalg.cho_factor(block_scaling[:, None] * block * block_scaling[None, :], lower=True,
                                  check_finite=False)
        local_solution[indices] = block_scaling[:, None] * linalg.cho_solve(
            factor, block_scaling[:, None] * local_rhs[indices], check_finite=False
        )
    coarse_solution = coarse_particular - harmonic @ local_solution
    solution = scaling[:, None] * np.concatenate((coarse_solution, local_solution), axis=0)
    removed = np.sqrt(max(0.0, eliminated_squared)) / max(np.linalg.norm(schur), 1e-30)
    return CouplingDropResult(solution[:, 0] if rhs.ndim == 1 else solution, len(groups), float(removed))
