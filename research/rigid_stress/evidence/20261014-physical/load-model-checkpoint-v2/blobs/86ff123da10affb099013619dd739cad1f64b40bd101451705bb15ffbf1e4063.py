"""Validate optional native GPU factors against independent CPU FP64 and exported SpSM solves."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .cudss import SharedCuDSSFactor
from .sparse_gpu import EggRecoveryGPU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--cpu-backend", choices=("superlu", "cholmod"), default="superlu")
    parser.add_argument("--large", action="store_true", help="One integrated factorization only for large mesh limits")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(
            17,
            EggConfig(
                level=args.level,
                ordering="column-nd",
                factor_backend=args.cpu_backend,
            ),
            history=0,
            direct=True,
        )
        fem = model.fem
        rng = np.random.default_rng(112983)
        value = fem.balance(rng.normal(size=(fem.ndof, 17)))
        expected = np.zeros(value.shape)
        expected[fem.free] = fem.factor.solve(value[fem.free])
        for natural in (() if args.large else (False, True)):
            factor = SharedCuDSSFactor(fem.k[fem.free][:, fem.free], 1, natural_order=natural)
            for columns in (1, 3, 8, 17):
                actual = factor.solve(cp.asarray(value[fem.free, :columns]))
                cp.cuda.get_current_stream().synchronize()
                relative = np.linalg.norm(fem.k[fem.free][:, fem.free] @ actual.get() - value[fem.free, :columns]) / (
                    np.linalg.norm(value[fem.free, :columns])
                )
                np.testing.assert_allclose(actual.get(), expected[fem.free, :columns], atol=1e-10, rtol=1e-6)
                assert relative <= 1e-6
                rows.append(
                    {
                        "columns": columns,
                        "natural_order": natural,
                        "reduced_relative_residual": float(relative),
                        "cuDSS_version": factor.api.version,
                        "factor_nnz": factor.factor_nnz,
                        "factorization_s": factor.factorization_s,
                        "analysis_s": factor.analysis_s,
                        "memory_estimates_bytes": factor.memory_estimates_bytes,
                    }
                )
            factor.close()
        for columns in ((17,) if args.large else (1, 8, 17)):
            recovery = EggRecoveryGPU(fem, columns, factor_backend="cudss")
            out = recovery.recover(cp.asarray(value[:, :columns]))
            peaks = np.array([model.peak(expected[:, i])[0] for i in range(columns)])
            error = float(np.max(abs(out.peak_pa.get() - peaks) / np.maximum(peaks, 1)))
            assert cp.all(out.is_accepted).item()
            assert error <= 1e-4
            rows.append(
                {
                    "columns": columns,
                    "full_relative_residual_max": float(cp.max(out.relative_residual).item()),
                    "peak_relative_error_max": error,
                }
            )
            recovery.factor.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"GPU_tested": True, "passed": True, "cases": rows}, indent=2) + "\n")
        print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
