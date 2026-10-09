"""Device-resident replay of complete recorded rigid contacts, with independently phased recovery histories."""

import json
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np

from .device_pressure import PadPressureGPU
from .sparse_gpu import EggRecoveryGPU, GPURecoveryResult


@dataclass(frozen=True)
class ReplayFrame:
    position_m: cp.ndarray
    force_n: cp.ndarray
    normal: cp.ndarray
    radius_m: cp.ndarray
    friction: cp.ndarray
    valid: cp.ndarray
    omega_rad_s: cp.ndarray
    gravity_m_s2: cp.ndarray
    reset_ids: np.ndarray


class ContactReplayGPU:
    def __init__(self, path: Path, environments: int) -> None:
        source = np.load(path)
        self.frames, self.source_environments = source["omega_rad_s"].shape[:2]
        capacity = int(source["ids"][:, 1].max(initial=-1)) + 1
        shape = (self.frames, self.source_environments, capacity)
        arrays = {
            key: np.zeros((*shape, 3) if key in ("position_m", "force_n", "inward_normal") else shape)
            for key in ("position_m", "force_n", "inward_normal", "radius_m", "friction")
        }
        valid = np.zeros(shape, dtype=bool)
        for frame in range(self.frames):
            first, last = source["offsets"][frame : frame + 2]
            ids = source["ids"][first:last]
            valid[frame, ids[:, 0], ids[:, 1]] = True
            for key, array in arrays.items():
                array[frame, ids[:, 0], ids[:, 1]] = source[key][first:last]
        self.position_m = cp.asarray(arrays["position_m"])
        self.force_n = cp.asarray(arrays["force_n"])
        self.normal = cp.asarray(arrays["inward_normal"])
        self.radius_m, self.friction = cp.asarray(arrays["radius_m"]), cp.asarray(arrays["friction"])
        self.valid = cp.asarray(valid)
        self.omega_rad_s, self.gravity_m_s2 = cp.asarray(source["omega_rad_s"]), cp.asarray(source["gravity_m_s2"])
        self.dt_s, self.source_epsilon = float(source["dt_s"]), float(source["source_epsilon"])
        self.environments = environments
        self.source_ids_cpu = np.arange(environments) % self.source_environments
        self.offset_cpu = ((np.arange(environments) // self.source_environments) * 37) % self.frames
        self.source_ids, self.offset = cp.asarray(self.source_ids_cpu), cp.asarray(self.offset_cpu)
        companion = path.with_name(path.name.removesuffix(".contacts.npz") + ".json")
        records = json.loads(companion.read_text())["records"]
        phase = np.repeat(np.array([row["phase"] for row in records]), self.frames // len(records), axis=0)
        if phase.shape != (self.frames, self.source_environments):
            raise ValueError("A replay requires matching phase records for every source environment")
        self.reset_mask = np.zeros(phase.shape, dtype=bool)
        self.reset_mask[0] = True
        self.reset_mask[1:] = np.diff(phase, axis=0) < -1e-9
        self.path = path

    def frame(self, tick: int) -> ReplayFrame:
        frames = (tick + self.offset) % self.frames
        ids = self.source_ids
        cpu_frames = (tick + self.offset_cpu) % self.frames
        reset = np.flatnonzero(self.reset_mask[cpu_frames, self.source_ids_cpu])
        return ReplayFrame(
            self.position_m[frames, ids],
            self.force_n[frames, ids],
            self.normal[frames, ids],
            self.radius_m[frames, ids],
            self.friction[frames, ids],
            self.valid[frames, ids],
            self.omega_rad_s[frames, ids],
            self.gravity_m_s2[frames, ids],
            reset,
        )


class ReplayRecoveryPipeline:
    def __init__(self, source: ContactReplayGPU, mapper: PadPressureGPU, recovery: EggRecoveryGPU) -> None:
        self.source, self.mapper, self.recovery = source, mapper, recovery
        self.last_rhs: cp.ndarray | None = None
        self.mapping_accepted: cp.ndarray | None = None
        self.reset_count = 0

    def step(self, tick: int) -> GPURecoveryResult:
        frame = self.source.frame(tick)
        if len(frame.reset_ids):
            self.recovery.reset(cp.asarray(frame.reset_ids))
            self.reset_count += len(frame.reset_ids)
        mapped = self.mapper.map(
            frame.position_m,
            frame.force_n,
            frame.radius_m,
            frame.normal,
            frame.friction,
            frame.valid,
            self.source.source_epsilon,
        )
        external = mapped.nodal_force_n + self.recovery.mass_modes[:, :3] @ frame.gravity_m_s2.T
        self.last_rhs = self.recovery.compatible_rhs(external, frame.omega_rad_s)
        self.mapping_accepted = mapped.is_accepted
        return self.recovery.recover(self.last_rhs, self.source.dt_s)
