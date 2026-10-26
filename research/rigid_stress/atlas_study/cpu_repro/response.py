"""Complete fixed-model load responses, conservative sampled maxima and energy POD."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, sparse

if TYPE_CHECKING:
    from .model import Mother


def top_force_map(directors: np.ndarray, thickness: float) -> sparse.csr_matrix:
    """Map nodal top-surface forces to translations and eccentric rotational loads."""
    count = len(directors)
    rows, columns, values = [], [], []
    axes = np.eye(3)
    for node in range(count):
        for component in range(3):
            column = 3 * node + component
            rows.append(6 * node + component)
            columns.append(column)
            values.append(1.0)
            couple = 0.5 * thickness * np.cross(directors[node], axes[component])
            for rotation in range(3):
                if abs(couple[rotation]) > 0.0:
                    rows.append(6 * node + 3 + rotation)
                    columns.append(column)
                    values.append(couple[rotation])
    return sparse.coo_matrix((values, (rows, columns)), shape=(6 * count, 3 * count)).tocsr()


def local_von_mises(stress: np.ndarray) -> np.ndarray:
    """Compute plane-stress von Mises values from common local stress rows."""
    values = stress if stress.ndim == 2 else stress[:, None]
    tensors = values.reshape((-1, 3, values.shape[1]))
    xx, yy, xy = tensors[:, 0], tensors[:, 1], tensors[:, 2]
    square = xx * xx - xx * yy + yy * yy + 3.0 * xy * xy
    return np.sqrt(np.maximum(square, 0.0))


def sampled_maximum(stress: np.ndarray) -> np.ndarray:
    """Reduce all declared stress samples independently per environment."""
    return local_von_mises(stress).max(axis=0)


def relative_tensor_error(reference: np.ndarray, candidate: np.ndarray, point_weights: np.ndarray) -> np.ndarray:
    """Measure an area-weighted symmetric-tensor Frobenius difference."""
    if reference.ndim == 1:
        reference = reference[:, None]
    if candidate.ndim == 1:
        candidate = candidate[:, None]
    difference = (candidate - reference).reshape((-1, 3, reference.shape[1]))
    tensors = reference.reshape((-1, 3, reference.shape[1]))
    weights = point_weights[:, None, None] * np.array([1.0, 1.0, 2.0])[None, :, None]
    numerator = np.sum(weights * difference * difference, axis=(0, 1))
    denominator = np.sum(weights * tensors * tensors, axis=(0, 1))
    return np.sqrt(numerator / np.maximum(denominator, 1e-300))


@dataclass
class ExactTopResponse:
    """Store every nodal top-force response at the common mother output samples."""

    values: np.ndarray
    directors: np.ndarray
    thickness: float
    build_seconds: float

    @classmethod
    def build(cls, mother: Mother, chunk_columns: int = 64, dtype=np.float64) -> ExactTopResponse:
        start = perf_counter()
        shell = mother.model
        load_map = top_force_map(shell.directors, shell.h)
        values = np.empty((mother.S.shape[0], load_map.shape[1]), dtype=np.float64, order="F")
        for first in range(0, load_map.shape[1], chunk_columns):
            last = min(first + chunk_columns, load_map.shape[1])
            balanced = mother.balance(load_map[:, first:last].toarray())
            values[:, first:last] = mother.S @ mother.solve(balanced)
        if np.dtype(dtype) != values.dtype:
            values = values.astype(dtype, order="F")
        return cls(values, shell.directors.copy(), shell.h, perf_counter() - start)

    @property
    def nbytes(self) -> int:
        return self.values.nbytes

    def quantized(self, dtype=np.float32) -> ExactTopResponse:
        return ExactTopResponse(self.values.astype(dtype, order="F"), self.directors, self.thickness, 0.0)

    def pack_raw(self, raw_force: np.ndarray, side: int = 1) -> np.ndarray:
        """Validate and extract the raw top nodal force coordinates before relief."""
        if side != 1:
            raise ValueError("This inventory supports the top physical surface only.")
        force = raw_force if raw_force.ndim == 2 else raw_force[:, None]
        count = len(self.directors)
        if force.shape[0] != 6 * count or not np.isfinite(force).all():
            raise ValueError("Expected finite raw generalized loads with six rows per node.")
        nodes = force.reshape((count, 6, force.shape[1]))
        translations = nodes[:, :3, :]
        expected = 0.5 * self.thickness * np.cross(
            self.directors[:, :, None], translations, axisa=1, axisb=1, axisc=1
        )
        actual = nodes[:, 3:, :]
        difference = np.linalg.norm((actual - expected).reshape((-1, force.shape[1])), axis=0)
        scale = np.maximum(np.linalg.norm(actual.reshape((-1, force.shape[1])), axis=0), 1e-30)
        if np.any(difference > 1e-11 * scale + 1e-20):
            raise ValueError("Raw force contains a couple outside the top nodal-force basis.")
        return translations.reshape((3 * count, force.shape[1]))

    def apply_nodal(self, nodal_force: np.ndarray, sparse_input: bool = True) -> np.ndarray:
        """Apply current nodal inputs with optional exact zero-column omission."""
        force = nodal_force if nodal_force.ndim == 2 else nodal_force[:, None]
        if force.shape[0] != self.values.shape[1] or not np.isfinite(force).all():
            raise ValueError("Nodal input does not match the prepared response basis.")
        force = force.astype(self.values.dtype, copy=False)
        if not sparse_input:
            return self.values @ force
        result = np.zeros((self.values.shape[0], force.shape[1]), dtype=self.values.dtype)
        for environment in range(force.shape[1]):
            columns = np.flatnonzero(np.abs(force[:, environment]) > 0.0)
            if len(columns):
                result[:, environment] = self.values[:, columns] @ force[columns, environment]
        return result

    def apply(self, raw_force: np.ndarray, side: int = 1, sparse_input: bool = True) -> np.ndarray:
        return self.apply_nodal(self.pack_raw(raw_force, side=side), sparse_input=sparse_input)


@dataclass
class BoundHistory:
    nodal_force: np.ndarray
    upper: np.ndarray


@dataclass
class MaximumResult:
    maxima: np.ndarray
    evaluated_blocks: np.ndarray
    total_blocks: int


class MaximumBounds:
    """Screen the complete fixed output set with response-column norm bounds."""

    def __init__(self, response: ExactTopResponse, block_elements: int = 24):
        if response.values.dtype != np.float64:
            raise ValueError("This bounded-maximum experiment requires FP64 response storage.")
        if block_elements < 1 or response.values.shape[0] % 24:
            raise ValueError("Expected two sides, four Gauss points and three stress components.")
        self.response = response
        elements = response.values.shape[0] // 24
        row_grid = np.arange(response.values.shape[0]).reshape((2, elements, 4, 3))
        self.rows = tuple(
            row_grid[:, first : min(first + block_elements, elements)].reshape(-1)
            for first in range(0, elements, block_elements)
        )
        self.limits = np.empty((len(self.rows), response.values.shape[1]))
        for block, rows in enumerate(self.rows):
            columns = response.values[rows].reshape((-1, 3, response.values.shape[1]))
            xx, yy, xy = columns[:, 0], columns[:, 1], columns[:, 2]
            self.limits[block] = np.sqrt(np.maximum(xx * xx - xx * yy + yy * yy + 3.0 * xy * xy, 0.0)).max(axis=0)
        self.history: dict[int, BoundHistory] = {}

    @classmethod
    def build(cls, response: ExactTopResponse, block_elements: int = 24) -> MaximumBounds:
        return cls(response, block_elements=block_elements)

    @property
    def nbytes(self) -> int:
        return self.limits.nbytes + sum(rows.nbytes for rows in self.rows)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            self.history.clear()
        else:
            for environment in env_ids:
                self.history.pop(int(environment), None)

    def apply(
        self,
        raw_force: np.ndarray,
        env_ids: np.ndarray | None = None,
        use_history: bool = False,
        side: int = 1,
    ) -> MaximumResult:
        force = self.response.pack_raw(raw_force, side=side)
        identities = np.arange(force.shape[1]) if env_ids is None else env_ids
        if len(identities) != force.shape[1] or len(np.unique(identities)) != len(identities):
            raise ValueError("Each batch column needs one unique environment identity.")
        maxima = np.zeros(force.shape[1])
        evaluated = np.zeros(force.shape[1], dtype=np.int64)
        for environment, identity in enumerate(identities):
            vector = force[:, environment]
            columns = np.flatnonzero(np.abs(vector) > 0.0)
            if not len(columns):
                if use_history:
                    self.history[int(identity)] = BoundHistory(vector.copy(), np.zeros(len(self.rows)))
                continue
            upper = self.limits[:, columns] @ np.abs(vector[columns])
            previous = self.history.get(int(identity)) if use_history else None
            if previous is not None:
                delta = vector - previous.nodal_force
                changed = np.flatnonzero(np.abs(delta) > 0.0)
                temporal = previous.upper + self.limits[:, changed] @ np.abs(delta[changed])
                upper = np.minimum(upper, temporal)
            # This margin covers tested floating-point evaluation differences.
            # Directed-rounding interval certification is a separate requirement.
            upper = upper * (1.0 + 1e-12) + np.finfo(np.float64).tiny
            for block in np.argsort(-upper, kind="stable"):
                if upper[block] <= maxima[environment]:
                    break
                rows = self.rows[block]
                values = self.response.values[np.ix_(rows, columns)] @ vector[columns]
                exact = sampled_maximum(values)[0]
                maxima[environment] = max(maxima[environment], exact)
                upper[block] = exact + 1e-12 * upper[block] + np.finfo(np.float64).tiny
                evaluated[environment] += 1
            if use_history:
                self.history[int(identity)] = BoundHistory(vector.copy(), upper.copy())
        return MaximumResult(maxima, evaluated, len(self.rows))


@dataclass
class EnergyPOD:
    """Project current loads onto a fixed energy-trained displacement basis."""

    basis: np.ndarray
    stress_basis: np.ndarray
    stiffness_basis: np.ndarray
    factor: tuple[np.ndarray, bool]
    eigenvalues: np.ndarray

    @classmethod
    def build(cls, mother: Mother, snapshots: np.ndarray, rank: int) -> EnergyPOD:
        product = mother.K @ snapshots
        gram = snapshots.T @ product
        values, vectors = linalg.eigh(0.5 * (gram + gram.T))
        valid = np.flatnonzero(values > max(values[-1], 1e-300) * 1e-12)[::-1]
        kept = valid[: min(rank, len(valid))]
        basis = snapshots @ (vectors[:, kept] / np.sqrt(values[kept])[None, :])
        stiffness_basis = mother.K @ basis
        reduced = basis.T @ stiffness_basis
        factor = linalg.cho_factor(0.5 * (reduced + reduced.T), lower=True)
        return cls(basis, mother.S @ basis, stiffness_basis, factor, values[kept])

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def nbytes(self) -> int:
        return self.basis.nbytes + self.stress_basis.nbytes + self.stiffness_basis.nbytes + self.factor[0].nbytes

    def displacement(self, balanced_force: np.ndarray) -> np.ndarray:
        coefficient = linalg.cho_solve(self.factor, self.basis.T @ balanced_force)
        return self.basis @ coefficient

    def apply(self, balanced_force: np.ndarray) -> np.ndarray:
        coefficient = linalg.cho_solve(self.factor, self.basis.T @ balanced_force)
        return self.stress_basis @ coefficient
