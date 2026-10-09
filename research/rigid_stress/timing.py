"""Scoped benchmark metadata and sampled total device memory, including native library allocations."""

import hashlib
import os
import platform
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import cupy as cp

from .cudss import SharedCuDSSFactor
from .sparse_gpu import EggRecoveryGPU
from .temporal_gpu import TemporalRecoveryGPU


def host_metadata() -> dict:
    path = Path("/proc/cpuinfo")
    lines = path.read_text().splitlines() if path.is_file() else []
    return {
        "cpu_model": next((line.partition(":")[2].strip() for line in lines if line.startswith("model name")), None),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "thread_environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


class DeviceMemorySampler:
    """Sample CUDA's device-wide memory accounting; report the interval and avoid claiming an exact peak.

    CuPy and Torch pool counters exclude opaque factors and simulator allocations. Device-wide samples include those
    and unrelated processes on the assigned GPU. The sampling thread runs during the measured scope, so its overhead
    is included. Short allocation spikes between samples can be missed.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s, self.device = interval_s, cp.cuda.runtime.getDevice()
        self.stop = Event()
        self.samples: list[tuple[float, int]] = []
        self.errors: list[str] = []
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        try:
            with cp.cuda.Device(self.device):
                while not self.stop.is_set():
                    available, total = cp.cuda.runtime.memGetInfo()
                    self.samples.append((perf_counter(), total - available))
                    self.stop.wait(self.interval_s)
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def metadata(self) -> dict:
        if self.errors or not self.samples:
            raise RuntimeError(f"Device memory sampling failed: {self.errors}")
        return {
            "sampled_device_memory_peak_bytes": max(value for _, value in self.samples),
            "device_memory_sample_count": len(self.samples),
            "device_memory_sample_interval_s": self.interval_s,
            "device_memory_accounting": "CUDA device-wide used memory; sampled, includes native/context allocations",
        }


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = sorted(Path(__file__).parent.glob("*.py")) + [
        root / "genesis/engine/scene.py",
        root / "genesis/engine/simulator.py",
        root / "genesis/engine/solvers/rigid/collider/collider.py",
        root / "genesis/engine/solvers/rigid/collider/contact.py",
        root / "genesis/utils/geom.py",
        root / "genesis/engine/solvers/rigid/constraint/solver.py",
        root / "genesis/engine/entities/rigid_entity/rigid_entity.py",
    ]
    return {str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def factor_metadata(recovery: EggRecoveryGPU) -> dict:
    factors = [recovery.factor]
    if isinstance(recovery, TemporalRecoveryGPU) and recovery.precision == "32":
        factors.append(recovery.correction_factor.factor)
    records = []
    for factor in factors:
        if isinstance(factor, SharedCuDSSFactor):
            records.append(
                {
                    "backend": "cuDSS",
                    "allocator": "cudaMallocAsync" if factor.allocator is not None else "default",
                    "precision": str(factor.dtype),
                    "version": factor.api.version,
                    "nnz": factor.factor_nnz,
                    "analysis_seconds": factor.analysis_s,
                    "factorization_seconds": factor.factorization_s,
                    "memory_estimates_bytes": factor.memory_estimates_bytes,
                }
            )
        else:
            records.append(
                {
                    "backend": "cuSPARSE SpSM",
                    "precision": str(factor.lower.dtype),
                    "nnz": factor.lower.nnz + factor.upper.nnz,
                }
            )
    return {"operator_groups": 1, "factor_precision_copies": len(factors), "factors": records}
