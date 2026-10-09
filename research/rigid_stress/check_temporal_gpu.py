"""Validate device temporal and mixed-precision recovery against identical complete FP64 direct RHS."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .oracle import reference
from .temporal_gpu import TemporalRecoveryGPU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-backend", choices=("spsm", "cudss"), default="spsm")
    parser.add_argument("--layout", choices=("history-major", "dof-major"), default="history-major")
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(8, EggConfig(level=args.level, ordering="column-nd"), history=0, direct=True)
        fem = model.fem
        rng = np.random.default_rng(99285)
        source = fem.balance(rng.normal(size=(fem.ndof, 24))).reshape(fem.ndof, 8, 3)
        configurations = (
            (0, "compact", "64", False, 0, 1e-6),
            (4, "compact", "64", False, 0, 1e-6),
            (8, "padded", "64", False, 0, 1e-6),
            (4, "compact", "32", False, 4, 1e-6),
            (4, "padded", "32", True, 4, 1e-6),
            (0, "compact", "32", True, 0, 1e-6),
            (4, "compact", "32", True, 1, 1e-3),
            (4, "adaptive", "64", False, 0, 1e-6),
        )
        for history, strategy, precision, scaling, refinements, rtol in configurations:
            gpu = TemporalRecoveryGPU(
                fem,
                environments=8,
                history=history,
                rtol=rtol,
                strategy=strategy,
                precision=precision,
                scaling=scaling,
                refinements=refinements,
                chunk=3,
                factor_backend=args.factor_backend,
                layout=args.layout,
            )
            errors, absolute_errors, residuals, failures, fallback_counts, refinement_counts = [], [], [], [], [], []
            for frame in range(40):
                rhs = source[:, :, 0] * (1 + frame * 0.03) + source[:, :, 1] * np.sin(frame * 0.03)
                if frame in (0, 9, 15, 31):
                    rhs += source[:, :, 2] * rng.uniform(-1, 1, 8)
                rhs[:, frame % 8] = 0
                if frame == 17:
                    preserved = cp.asnumpy(gpu.last[:, [0, 2, 4]])
                    gpu.reset(cp.asarray([1, 6]))
                    np.testing.assert_array_equal(cp.asnumpy(gpu.last[:, [0, 2, 4]]), preserved)
                    assert cp.all(gpu.frames[[1, 6]] == 0).item()
                dt = np.full(8, 0.01 if frame % 3 else 0.005)
                if frame == 23:
                    dt[[0, 5]] = [np.nan, -1]
                out = gpu.recover(cp.asarray(rhs), dt=cp.asarray(dt))
                assert cp.all(out.is_accepted).item()
                expected = np.zeros_like(rhs)
                expected[fem.free] = fem.factor.solve(rhs[fem.free])
                peaks = np.array([reference.peak(fem, expected[:, i]) for i in range(8)])
                actual = cp.asnumpy(out.peak_pa)
                error = abs(peaks - actual)
                errors.append(float(np.max(error / np.maximum(peaks, 1))))
                absolute_errors.append(float(error.max()))
                residuals.append(float(cp.max(out.relative_residual).item()))
                failures.append(int(cp.count_nonzero(gpu.statistics.failed).item()))
                fallback_counts.append(int(cp.count_nonzero(gpu.statistics.used_fp64_fallback).item()))
                refinement_counts.append(int(cp.sum(gpu.statistics.refinement_count).item()))
            threshold = 1e-4 if rtol <= 1e-6 else 1e-2
            assert max(errors) <= threshold
            invalid = cp.zeros_like(gpu.last)
            invalid[0, 0] = cp.nan
            assert not gpu.recover(invalid).is_accepted[0].item()
            assert cp.all(gpu.recover(cp.zeros_like(invalid)).is_accepted).item()
            gpu.reset()
            assert not cp.any(gpu.is_valid).item()
            rows.append(
                {
                    "history": history,
                    "strategy": strategy,
                    "precision": precision,
                    "scaling": scaling,
                    "maximum_refinement_steps": refinements,
                    "rtol": rtol,
                    "chunk": 3,
                    "peak_relative_error_max": max(errors),
                    "peak_absolute_error_max_pa": max(absolute_errors),
                    "full_relative_residual_max": max(residuals),
                    "failed_environments_per_frame": failures,
                    "fp64_fallback_environments_per_frame": fallback_counts,
                    "refinement_environment_steps_per_frame": refinement_counts,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        deficient = TemporalRecoveryGPU(fem, environments=1, history=4)
        rhs = source[:, 0, 0:1]
        direction = np.zeros_like(rhs)
        direction[fem.free] = fem.factor.solve(rhs[fem.free])
        magnitude = np.linalg.norm(fem.k @ direction)
        deficient.Q[0, :2] = cp.asarray((direction / magnitude).T)
        deficient.KQ[0, :2] = cp.asarray((fem.k @ direction / magnitude).T)
        deficient.gram[0, :2, :2] = 1
        deficient.is_valid[0, :2] = True
        out = deficient.recover(cp.asarray(rhs))
        assert out.is_accepted[0].item()
        np.testing.assert_allclose(out.peak_pa.get(), [reference.peak(fem, direction[:, 0])], rtol=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"passed": True, "GPU_tested": True, "rank_deficient_projection": True, "cases": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
