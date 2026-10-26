"""Batch linear recovery with current-load prediction gates and exact fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from .model import Mother


@dataclass
class RecoverySystem:
    stiffness: sparse.csr_matrix
    stress: sparse.csr_matrix
    scale: np.ndarray
    factor: object
    free: np.ndarray
    stress_weights: np.ndarray

    @classmethod
    def from_mother(cls, mother: Mother) -> RecoverySystem:
        free = mother.free
        stiffness = mother.K[free][:, free].tocsr()
        scale = mother.factor.scale
        factor = mother.factor.lu
        weights = np.tile(
            np.repeat(mother.model.gauss_weights.reshape(-1), 3) * np.tile([1.0, 1.0, 2.0], 4 * len(mother.model.quads)),
            2,
        )
        return cls(stiffness, mother.S[:, free].tocsr(), scale, factor, free, weights)

    def solve(self, forces: np.ndarray) -> np.ndarray:
        if forces.shape[1] == 0:
            return np.empty_like(forces)
        right = self.scale[:, None] * forces
        return self.scale[:, None] * self.factor.solve(np.asfortranarray(right))


@dataclass(frozen=True)
class RecoveryFrame:
    displacement: np.ndarray
    stress: np.ndarray
    accepted_prediction: np.ndarray
    failed_environments: np.ndarray
    residual_ratio: np.ndarray
    depth: np.ndarray
    cache_gate_drift: float
    cache_product_drift: float
    cache_decision_disagreement: int
    cache_direct_rechecks: int
    threshold_gap: float


class BatchHistory:
    """Recover current full fields, compacting every prediction failure together.

    Force columns already use the fixed mother's common inertia relief. Returned
    displacement and stress arrays are views consumed before the next step.
    """

    def __init__(
        self,
        system: RecoverySystem,
        n_envs: int,
        tolerance: float = 0.01,
        rotation: bool = True,
        cached_product: bool = False,
        prediction: bool = True,
    ):
        if n_envs < 1 or tolerance < 0.0:
            raise ValueError("Require positive environment count and nonnegative tolerance.")
        if cached_product and not prediction:
            raise ValueError("Cached products require prediction.")
        self.system = system
        self.n_envs = n_envs
        self.tolerance = tolerance
        self.rotation = rotation
        self.cached_product = cached_product
        self.prediction = prediction
        n_history = 3 if prediction else 1
        self.q_buffers = [np.zeros((system.stiffness.shape[0], n_envs)) for _ in range(n_history)]
        self.s_buffers = [np.zeros((system.stress.shape[0], n_envs)) for _ in range(n_history)]
        self.g_buffers = [np.zeros_like(self.q_buffers[0]) for _ in range(3)] if cached_product else []
        self.depth = np.zeros(n_envs, dtype=np.int8)

    def reset(self, envs: np.ndarray) -> None:
        self.depth[envs] = 0
        for arrays in (self.q_buffers, self.s_buffers, self.g_buffers):
            for array in arrays:
                array[:, envs] = 0.0

    def _ratio(self, residual: np.ndarray, force_norm: np.ndarray) -> np.ndarray:
        weighted = self.system.scale[:, None] * residual
        return np.linalg.norm(weighted, axis=0) / np.maximum(force_norm, 1e-30)

    def step(
        self,
        forces: np.ndarray,
        active: np.ndarray,
        reset: np.ndarray | None = None,
        audit_cache: bool = False,
    ) -> RecoveryFrame:
        if forces.shape != self.q_buffers[0].shape or active.shape != (self.n_envs,):
            raise ValueError("Force and active shapes must match the prepared batch.")
        if reset is not None:
            self.reset(reset)
        inactive = np.flatnonzero(~active)
        self.reset(inactive)
        accepted = np.zeros(self.n_envs, dtype=bool)
        ratio = np.full(self.n_envs, np.nan)
        gate_drift = product_drift = 0.0
        decisions = 0
        direct_rechecks = 0
        threshold_gap = float("inf")
        if self.prediction:
            older, previous, current = self.q_buffers
            stress_older, stress_previous, stress_current = self.s_buffers
            np.multiply(previous, 2.0, out=current)
            current -= older
            np.multiply(stress_previous, 2.0, out=stress_current)
            stress_current -= stress_older
            eligible = active & (self.depth >= 2)
            force_norm = np.linalg.norm(self.system.scale[:, None] * forces, axis=0)
            if self.cached_product:
                product_older, product_previous, product_current = self.g_buffers
                np.multiply(product_previous, 2.0, out=product_current)
                product_current -= product_older
            else:
                product_current = self.system.stiffness @ current
            ratio[:] = self._ratio(product_current - forces, force_norm)
            if self.cached_product:
                near = np.flatnonzero(eligible & (np.abs(ratio - self.tolerance) <= 1e-10))
                if len(near):
                    product_current[:, near] = self.system.stiffness @ current[:, near]
                    ratio[near] = self._ratio(product_current[:, near] - forces[:, near], force_norm[near])
                    direct_rechecks = len(near)
            accepted[:] = eligible & (ratio <= self.tolerance)
            if np.any(eligible):
                threshold_gap = float(np.min(np.abs(ratio[eligible] - self.tolerance)))
            if audit_cache and self.cached_product:
                direct = self.system.stiffness @ current
                direct_ratio = self._ratio(direct - forces, force_norm)
                gate_drift = float(np.max(np.abs(ratio - direct_ratio)))
                product_drift = float(
                    np.max(np.linalg.norm((direct - product_current) * self.system.scale[:, None], axis=0)
                           / np.maximum(force_norm, 1e-30))
                )
                decisions = int(np.count_nonzero(eligible & ((direct_ratio <= self.tolerance) != accepted)))
        else:
            current = self.q_buffers[0]
            stress_current = self.s_buffers[0]
        failed = np.flatnonzero(active & ~accepted)
        if len(failed):
            solved = self.system.solve(forces[:, failed])
            current[:, failed] = solved
            stress_current[:, failed] = self.system.stress @ solved
            if self.cached_product:
                product_current[:, failed] = self.system.stiffness @ solved
        current[:, inactive] = 0.0
        stress_current[:, inactive] = 0.0
        self.depth[active] = np.minimum(self.depth[active] + 1, 2)
        if self.prediction:
            if self.rotation:
                self.q_buffers = [previous, current, older]
                self.s_buffers = [stress_previous, stress_current, stress_older]
                if self.cached_product:
                    self.g_buffers = [product_previous, product_current, product_older]
            else:
                np.copyto(older, previous)
                np.copyto(previous, current)
                np.copyto(stress_older, stress_previous)
                np.copyto(stress_previous, stress_current)
                if self.cached_product:
                    np.copyto(product_older, product_previous)
                    np.copyto(product_previous, product_current)
        return RecoveryFrame(
            current,
            stress_current,
            accepted,
            failed,
            ratio,
            self.depth.copy(),
            gate_drift,
            product_drift,
            decisions,
            direct_rechecks,
            threshold_gap,
        )


def field_error(system: RecoverySystem, actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    numerator = np.sum(system.stress_weights[:, None] * (actual - expected) ** 2, axis=0)
    denominator = np.sum(system.stress_weights[:, None] * expected**2, axis=0)
    return np.sqrt(numerator / np.maximum(denominator, 1e-60))


def sampled_maximum(stress: np.ndarray) -> np.ndarray:
    local = stress.reshape(-1, 3, stress.shape[-1])
    return np.sqrt(np.maximum(local[:, 0] ** 2 + local[:, 1] ** 2 - local[:, 0] * local[:, 1]
                              + 3.0 * local[:, 2] ** 2, 0.0)).max(axis=0)
