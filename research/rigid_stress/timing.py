"""Scoped benchmark metadata and sampled total device memory, including native library allocations."""

import hashlib
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import cupy as cp


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
        root / "genesis/engine/solvers/rigid/constraint/solver.py",
        root / "genesis/engine/entities/rigid_entity/rigid_entity.py",
    ]
    return {str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
