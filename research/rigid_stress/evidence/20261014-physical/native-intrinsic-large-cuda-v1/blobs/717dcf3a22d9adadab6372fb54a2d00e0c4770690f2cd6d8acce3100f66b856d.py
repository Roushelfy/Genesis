"""Measure actual CUDA graph support and validate changing device right-hand sides after capture."""

import argparse
import json
import traceback
from pathlib import Path

import cupy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU
from .peak_tensor import P2PeakTensorGPU
from .sparse_gpu import EggRecoveryGPU
from .temporal_gpu import TemporalRecoveryGPU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(8, EggConfig(level=1, ordering="column-nd"), history=0, direct=True)
        rng = np.random.default_rng(194824)
        rows = []
        for method in ("direct", "padded", "compact"):
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                recovery = (
                    EggRecoveryGPU(model.fem, 8)
                    if method == "direct"
                    else TemporalRecoveryGPU(
                        model.fem,
                        8,
                        history=4,
                        strategy=method,
                    )
                )
                rhs = cp.zeros((model.fem.ndof, 8), order="F")
                rhs[:] = cp.asarray(model.fem.balance(rng.normal(size=rhs.shape)))
                for _ in range(4):
                    out = recovery.recover(rhs)
                stream.synchronize()
                supported, reason, error = False, "", 0.0
                try:
                    stream.begin_capture()
                    out = recovery.recover(rhs)
                    graph = stream.end_capture()
                except (RuntimeError, cp.cuda.runtime.CUDARuntimeError):
                    reason = traceback.format_exc()
                    if stream.is_capturing():
                        try:
                            stream.end_capture()
                        except cp.cuda.runtime.CUDARuntimeError:
                            pass
                else:
                    supported = True
                    for frame in range(10):
                        value = model.fem.balance(rng.normal(size=rhs.shape)) if frame % 3 else np.zeros(rhs.shape)
                        rhs[:] = cp.asarray(value)
                        graph.launch(stream)
                        stream.synchronize()
                        expected = np.zeros(rhs.shape)
                        expected[model.fem.free] = model.fem.factor.solve(value[model.fem.free])
                        peaks = np.array([model.peak(expected[:, i])[0] for i in range(8)])
                        error = max(error, float(np.max(abs(cp.asnumpy(out.peak_pa) - peaks) / np.maximum(peaks, 1))))
                        assert cp.all(out.is_accepted).item()
                    assert error <= 1e-4
            row = {"method": method, "graph_supported": supported, "reason": reason, "peak_relative_error_max": error}
            rows.append(row)
            print(json.dumps(row), flush=True)
        recovery = EggRecoveryGPU(model.fem, 8)
        value = cp.asarray(model.fem.balance(rng.normal(size=(model.fem.ndof, 8))))
        recovered = recovery.recover(value)
        tensor = P2PeakTensorGPU(model.fem.glambda, model.fem.elements, model.fem.young / (2 * (1 + model.fem.poisson)))
        np.testing.assert_allclose(cp.asnumpy(tensor(recovered.displacement_m)), recovered.peak_pa.get(), rtol=1e-12)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"GPU_tested": True, "cases": rows, "tensor_peak_passed": True}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
