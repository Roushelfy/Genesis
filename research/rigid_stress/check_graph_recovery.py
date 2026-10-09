"""Validate native full-residual recovery graphs on default/nondefault consumer streams against CPU FP64."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggRecoveryCPU
from .graph_gpu import CapturedDirectRecoveryGPU


def check(model: EggRecoveryCPU, rows: list[dict], layout: str) -> None:
    recovery = CapturedDirectRecoveryGPU(
        model.fem, model.environments, inertia="quadratic", body_products="fused", sparse_layout=layout
    )
    rng = np.random.default_rng(692371)
    for frame in range(24):
        force = rng.normal(size=(model.fem.ndof, model.environments))
        force[:, ::3] = 0 if frame % 2 else force[:, ::3]
        omega = rng.uniform(-5, 5, size=(model.environments, 3))
        omega[::3] = 0 if frame % 2 else omega[::3]
        rhs = model.compatible_rhs(force, omega)
        device_rhs = recovery.compatible_rhs(cp.asarray(force), cp.asarray(omega))
        np.testing.assert_allclose(device_rhs.get(), rhs, rtol=1e-10, atol=1e-12)
        expected = model.recover(rhs)
        actual = recovery.recover(device_rhs)
        assert cp.all(actual.is_accepted).item()
        difference = np.max(abs(actual.peak_pa.get() - expected.peak_pa) / np.maximum(expected.peak_pa, 1))
        assert difference <= 1e-4
        rows.append(
            {
                "frame": frame,
                "layout": layout,
                "consumer_stream": cp.cuda.get_current_stream().ptr,
                "capture_stream": recovery.capture_stream.ptr,
                "peak_relative_error_max": float(difference),
                "full_relative_residual_max": float(cp.max(actual.relative_residual)),
            }
        )
    cp.cuda.get_current_stream().synchronize()
    recovery.factor.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(8, history=0, direct=True)
        for layout in ("F", "C"):
            check(model, rows, layout)
            with cp.cuda.Stream(non_blocking=True):
                check(model, rows, layout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "GPU_tested": True, "cases": rows}, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
