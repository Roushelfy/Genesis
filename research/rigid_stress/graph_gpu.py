"""Capture the fixed-shape complete direct recovery and launch it on the caller's live stream."""

import cupy as cp
import numpy as np

from .sparse_gpu import EggRecoveryGPU, GPURecoveryResult


class CapturedDirectRecoveryGPU(EggRecoveryGPU):
    """Capture only recovery of a changing complete RHS; retain every full residual and global peak.

    Construction owns a nondefault capture stream when the caller uses CUDA's legacy default stream. The graph can
    subsequently launch on that default stream, preserving ordering with live contact mapping and observations.
    An isolated retained pool prevents unrelated eager allocations from overwriting graph temporaries.
    Results are owned persistent buffers: the next recover call overwrites them. Copy values retained across calls.
    """

    def __init__(
        self,
        fem,
        environments: int,
        rtol: float = 1e-6,
        inertia: str = "sparse",
        body_products: str = "cublas",
        sparse_layout: str = "F",
    ) -> None:
        stream = cp.cuda.get_current_stream()
        self.capture_stream = cp.cuda.Stream(non_blocking=True) if stream.ptr == 0 else stream
        with self.capture_stream:
            super().__init__(
                fem,
                environments,
                rtol=rtol,
                factor_backend="cudss",
                async_allocations=True,
                inertia=inertia,
                body_products=body_products,
                sparse_layout=sparse_layout,
            )
            self.graph_rhs = cp.zeros((fem.ndof, environments), dtype=np.float64, order="F")
            for _ in range(4):
                super().recover(self.graph_rhs)
            self.capture_stream.synchronize()
            # Warmup temporaries are dead; avoid retaining a second unused batch alongside the capture pool.
            cp.get_default_memory_pool().free_all_blocks()
            self.capture_pool = cp.cuda.MemoryPool()
            with cp.cuda.using_allocator(self.capture_pool.malloc):
                self.capture_stream.begin_capture()
                try:
                    self.graph_result = super().recover(self.graph_rhs)
                    self.graph = self.capture_stream.end_capture()
                except (RuntimeError, cp.cuda.runtime.CUDARuntimeError):
                    if self.capture_stream.is_capturing():
                        self.capture_stream.end_capture()
                    raise
            self.capture_stream.synchronize()

    def recover(self, rhs_n: cp.ndarray, dt: float | cp.ndarray = 1.0) -> GPURecoveryResult:
        if rhs_n.shape != self.graph_rhs.shape or rhs_n.dtype != np.float64:
            raise ValueError("Captured recovery requires FP64 complete RHS with its fixed mesh/environment count")
        self.graph_rhs[:] = rhs_n
        self.graph.launch(cp.cuda.get_current_stream())
        return self.graph_result
