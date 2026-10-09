"""Validate per-environment histories, deficient directions, rollover and timestep-safe prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .history_cpu import CachedHistory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        small = CachedHistory(sparse.eye(5, format="csr"), capacity=4)
        small.append_direction(np.ones(5))
        previous = small.Q.copy()
        small.append_direction(np.zeros(5))
        small.append_direction(2 * np.ones(5))
        assert small.order == [0] and small.rejected_directions == 2
        np.testing.assert_array_equal(previous, small.Q)
        # Duplicated cached directions exercise an explicitly rank-deficient Gram projection.
        small.Q[:, 1], small.KQ[:, 1] = small.Q[:, 0], small.KQ[:, 0]
        small.gram[:2, :2], small.order = 1, [0, 1]
        trial = np.zeros(5)
        small.project(trial, np.arange(5.0))
        np.testing.assert_allclose(trial, 2)
        for capacity in (0, 4, 8):
            model = EggRecoveryCPU(8, EggConfig(level=1), history=capacity, rtol=1e-6)
            fem = model.fem
            rng = np.random.default_rng(11551)
            source = fem.balance(rng.normal(size=(fem.ndof, 8, 3)).reshape(fem.ndof, 24)).reshape(fem.ndof, 8, 3)
            maximum_peak_error, maximum_cache_drift, corrections = 0.0, 0.0, 0
            for frame in range(64):
                rhs = source[:, :, 0] * (1 + 0.05 * frame) + source[:, :, 1] * np.sin(0.05 * frame)
                if frame % 9 == 0:
                    rhs += source[:, :, 2] * rng.uniform(-2, 2, 8)
                rhs[:, frame % 8] = 0
                if frame in (17, 33):
                    preserved = model.solver.last[:, [0, 2, 4]].copy()
                    model.reset([1, 3, 7])
                    np.testing.assert_array_equal(model.solver.last[:, [0, 2, 4]], preserved)
                    np.testing.assert_array_equal(model.solver.frames[[1, 3, 7]], 0)
                dt = np.full(8, 0.01 if frame % 3 else 0.005)
                if frame in (11, 27):
                    dt[[0, 5]] = [np.nan, -1]
                result = model.recover(rhs, dt=dt)
                expected = np.zeros_like(rhs)
                expected[fem.free] = fem.factor.solve(rhs[fem.free])
                peaks = np.array([model.peak(expected[:, i])[0] for i in range(8)])
                error = np.max(abs(peaks - result.peak_pa) / np.maximum(peaks, 1))
                maximum_peak_error = max(maximum_peak_error, float(error))
                corrections += result.corrected_environments
                for history in model.solver.history:
                    ids = np.asarray(history.order, dtype=int)
                    if len(ids):
                        drift = np.max(abs((fem.k @ history.Q[:, ids]) - history.KQ[:, ids]))
                        maximum_cache_drift = max(maximum_cache_drift, float(drift))
                        np.testing.assert_allclose(
                            history.gram[np.ix_(ids, ids)],
                            history.KQ[:, ids].T @ history.KQ[:, ids],
                            atol=1e-12,
                            rtol=1e-12,
                        )
            assert maximum_peak_error < 1e-4
            model.reset()
            assert all(not h.order for h in model.solver.history)
            np.testing.assert_array_equal(model.solver.frames, 0)
            rows.append(
                {
                    "capacity": capacity,
                    "frames": 64,
                    "environments": 8,
                    "peak_relative_error_max": maximum_peak_error,
                    "cached_operator_entry_drift_max": maximum_cache_drift,
                    "corrected_environments_total": corrections,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"passed": True, "rank_deficient_projection_passed": True, "cases": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
