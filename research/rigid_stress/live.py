"""Device stress observation of every rigid contact solve with matching pre-integration body frames."""

from collections.abc import Callable
from dataclasses import dataclass

import cupy as cp
import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene
from genesis.utils.geom import inv_transform_by_quat

from .device_pressure import DeviceMappedLoads, PadPressureGPU
from .sparse_gpu import EggRecoveryGPU, GPURecoveryResult


@dataclass(frozen=True)
class RigidSnapshot:
    position_m: torch.Tensor
    quaternion: torch.Tensor
    angular_velocity_rad_s: torch.Tensor
    linear_velocity_m_s: torch.Tensor


@dataclass(frozen=True)
class LocalContacts:
    position_m: torch.Tensor
    force_n: torch.Tensor
    inward_normal: torch.Tensor
    friction: torch.Tensor
    is_valid: torch.Tensor
    omega_rad_s: torch.Tensor
    gravity_m_s2: torch.Tensor
    is_egg_a: torch.Tensor


class RigidContactAdapter:
    def __init__(self, egg: RigidEntity, environments: int, gravity: tuple[float, float, float]) -> None:
        self.egg = egg
        self.gravity = torch.tensor(gravity, dtype=gs.tc_float, device=gs.device).expand(environments, -1)
        self.snapshot: RigidSnapshot | None = None

    def capture(self) -> None:
        self.snapshot = RigidSnapshot(
            self.egg.get_pos().clone(),
            self.egg.get_quat().clone(),
            self.egg.get_ang().clone(),
            self.egg.get_vel().clone(),
        )

    def contacts(self) -> LocalContacts:
        if self.snapshot is None:
            raise RuntimeError("Capture the body frame before its rigid contact solve")
        snapshot = self.snapshot
        contacts = self.egg.get_contacts(is_padded=True)
        is_egg_a = (contacts["geom_a"] >= self.egg.geom_start) & (contacts["geom_a"] < self.egg.geom_end)
        force = torch.where(is_egg_a[..., None], contacts["force_a"], contacts["force_b"])
        force = torch.where(contacts["valid_mask"][..., None], force, 0)
        normal = torch.where(is_egg_a[..., None], contacts["normal"], -contacts["normal"])
        return LocalContacts(
            inv_transform_by_quat(contacts["position"] - snapshot.position_m[:, None], snapshot.quaternion[:, None]),
            inv_transform_by_quat(force, snapshot.quaternion[:, None]),
            inv_transform_by_quat(normal, snapshot.quaternion[:, None]),
            contacts["friction"],
            contacts["valid_mask"],
            inv_transform_by_quat(snapshot.angular_velocity_rad_s, snapshot.quaternion),
            inv_transform_by_quat(self.gravity, snapshot.quaternion),
            is_egg_a,
        )


class StressSubstepObserver:
    """Recover every substep and retain the per-environment maximum and cumulative acceptance on the device."""

    def __init__(
        self,
        scene: Scene,
        egg: RigidEntity,
        mapper: PadPressureGPU,
        recovery: EggRecoveryGPU,
        radii_m: cp.ndarray,
        substeps: int,
        dt: float,
        diagnostic: Callable[[LocalContacts, DeviceMappedLoads, GPURecoveryResult, int], None] | None = None,
    ) -> None:
        if radii_m.shape != (recovery.environments, 1) or substeps < 1 or dt <= 0:
            raise ValueError("Per-environment radii, positive substeps and a positive scene timestep required")
        if scene.options.rigid.enable_torsional_friction or scene.options.rigid.enable_rolling_friction:
            raise ValueError("This force-only pad observer requires disabled spin/rolling friction")
        self.adapter = RigidContactAdapter(egg, recovery.environments, scene.options.sim.gravity)
        self.mapper, self.recovery, self.radii_m = mapper, recovery, radii_m
        self.substeps, self.substep_dt, self.diagnostic = substeps, dt / substeps, diagnostic
        self.peak_pa = cp.zeros(recovery.environments)
        self.is_accepted = cp.ones(recovery.environments, dtype=bool)
        self.substep_peak_pa = cp.zeros((substeps, recovery.environments))
        self.substep_is_accepted = cp.ones((substeps, recovery.environments), dtype=bool)
        self.full_relative_residual_max = cp.zeros(recovery.environments)
        self.meaningful_relative_residual_max = cp.zeros(recovery.environments)
        self.absolute_residual_max_n = cp.zeros(recovery.environments)
        self.contact_count = cp.zeros((substeps, recovery.environments), dtype=np.int32)
        self.egg_a_count = cp.zeros((substeps, recovery.environments), dtype=np.int32)
        self.source_epsilon = float(torch.finfo(gs.tc_float).eps)
        self.failed_events = cp.zeros(recovery.environments, dtype=np.int64)
        self.refinement_events = cp.zeros(recovery.environments, dtype=np.int64)
        self.fallback_events = cp.zeros(recovery.environments, dtype=np.int64)
        scene.register_pre_substep_callback(self.before)
        scene.register_post_substep_callback(self.after)

    def before(self, i_substep: int) -> None:
        if i_substep == 0:
            self.peak_pa.fill(0)
            self.is_accepted.fill(True)
            self.full_relative_residual_max.fill(0)
            self.meaningful_relative_residual_max.fill(0)
            self.absolute_residual_max_n.fill(0)
            self.failed_events.fill(0)
            self.refinement_events.fill(0)
            self.fallback_events.fill(0)
        self.adapter.capture()

    def after(self, i_substep: int) -> None:
        local = self.adapter.contacts()
        is_valid = cp.from_dlpack(local.is_valid)
        mapped = self.mapper.map(
            cp.from_dlpack(local.position_m),
            cp.from_dlpack(local.force_n),
            cp.broadcast_to(self.radii_m, is_valid.shape),
            cp.from_dlpack(local.inward_normal),
            cp.from_dlpack(local.friction),
            is_valid,
            self.source_epsilon,
        )
        # The translation columns of M R integrate any constant gravity without materializing a nodal vector field.
        external = mapped.nodal_force_n + self.recovery.mass_modes[:, :3] @ cp.from_dlpack(local.gravity_m_s2).T
        rhs = self.recovery.compatible_rhs(external, cp.from_dlpack(local.omega_rad_s))
        recovered = self.recovery.recover(rhs, self.substep_dt)
        accepted = mapped.is_accepted & recovered.is_accepted
        self.substep_peak_pa[i_substep] = recovered.peak_pa
        self.substep_is_accepted[i_substep] = accepted
        cp.maximum(self.peak_pa, recovered.peak_pa, out=self.peak_pa)
        cp.maximum(self.full_relative_residual_max, recovered.relative_residual, out=self.full_relative_residual_max)
        cp.maximum(
            self.meaningful_relative_residual_max,
            cp.where(recovered.rhs_norm_n >= self.recovery.atol_n / self.recovery.rtol, recovered.relative_residual, 0),
            out=self.meaningful_relative_residual_max,
        )
        cp.maximum(self.absolute_residual_max_n, recovered.absolute_residual_n, out=self.absolute_residual_max_n)
        self.is_accepted &= accepted
        statistics = self.recovery.statistics
        self.failed_events += statistics.failed
        self.refinement_events += statistics.refinement_count
        self.fallback_events += statistics.used_fp64_fallback
        self.contact_count[i_substep] = is_valid.sum(axis=1)
        self.egg_a_count[i_substep] = (is_valid & cp.from_dlpack(local.is_egg_a)).sum(axis=1)
        if self.diagnostic is not None:
            self.diagnostic(local, mapped, recovered, i_substep)

    def reset(self, environments: cp.ndarray) -> None:
        self.recovery.reset(environments)
        self.peak_pa[environments] = 0
        self.is_accepted[environments] = True
        self.substep_peak_pa[:, environments] = 0
        self.substep_is_accepted[:, environments] = True
        self.full_relative_residual_max[environments] = 0
        self.meaningful_relative_residual_max[environments] = 0
        self.absolute_residual_max_n[environments] = 0
        self.contact_count[:, environments] = 0
        self.egg_a_count[:, environments] = 0

    def observation(self) -> torch.Tensor:
        return torch.from_dlpack(self.peak_pa)
