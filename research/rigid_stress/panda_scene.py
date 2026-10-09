"""Reusable rigid Panda/egg scene, independent episode clocks and device stress observations."""

import json
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import torch

import genesis as gs

from .assets import write_egg_assets
from .cpu import EggRecoveryCPU
from .device_pressure import PadPressureGPU
from .live import StressSubstepObserver
from .sparse_gpu import EggRecoveryGPU


@dataclass(frozen=True)
class PandaConfig:
    environments: int = 1
    dt_s: float = 0.01
    substeps: int = 1
    episode_steps: int = 600
    seed: int = 0
    varied: bool = True
    asynchronous: bool = True
    egg_first: bool = True
    iterations: int = 100
    tolerance: float = 1e-12
    radius_m: float = 0.006
    radius_modulation: float = 0.15
    sampling: str = "grid"
    video_path: str | None = None


def smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return value**3 * (10 - 15 * value + 6 * value**2)


class PandaEggScene:
    """Scripted full grasps with optional small policy action perturbations.

    Reset scheduling uses deterministic CPU clocks; poses, actions, contacts and stress stay on the device. Every reset
    restores selected rigid states and clears only their recovery histories. These costs belong to a timed transition.
    The footprint radius is an explicit changing model input, with no inference of contact compliance from rigid points.
    """

    def __init__(
        self,
        model: EggRecoveryCPU | None,
        destination: Path,
        config: PandaConfig,
        recovery: EggRecoveryGPU | None = None,
    ) -> None:
        if config.environments < 1 or config.episode_steps < 100 or not 0 <= config.radius_modulation < 0.5:
            raise ValueError("Positive batch, at least 100 episode steps and radius modulation below 0.5 required")
        self.config, self.model, self.tick = config, model, 0
        self.reset_count = 0
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=config.dt_s, substeps=config.substeps),
            rigid_options=gs.options.RigidOptions(
                batch_dofs_info=True,
                friction_cone=gs.friction_cone.elliptic,
                contact_resolution=gs.contact_resolution.convex,
                iterations=config.iterations,
                tolerance=config.tolerance,
                contact_pruning_tolerance=None,
                enable_torsional_friction=False,
                enable_rolling_friction=False,
                use_hibernation=False,
            ),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        if model is None:
            if recovery is not None:
                raise ValueError("Stress observations require the matching elastic model")
            urdf = destination / "egg_shell.urdf"
            asset = json.loads((destination / "egg_shell.metadata.json").read_text())
            mass, com = asset["mass_kg"], np.array(asset["com_m"])
        else:
            urdf = write_egg_assets(model, destination)
            mass, com = model.fem.mass, model.fem.com
        egg_morph = gs.morphs.URDF(
            file=str(urdf.resolve()),
            pos=(0.65, 0, 0.031),
            align=False,
            convexify=True,
            decimate=False,
        )
        if config.egg_first:
            self.egg = self.scene.add_entity(egg_morph, material=gs.materials.Rigid(friction=0.6))
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        if not config.egg_first:
            self.egg = self.scene.add_entity(egg_morph, material=gs.materials.Rigid(friction=0.6))
        self.camera = None
        if config.video_path is not None:
            Path(config.video_path).parent.mkdir(parents=True, exist_ok=True)
            self.camera = self.scene.add_camera(
                res=(640, 480),
                pos=(0.9, -0.35, 0.36),
                lookat=(0.65, 0, 0.15),
                fov=40,
                GUI=False,
            )
        self.scene.build(n_envs=config.environments, env_spacing=(1, 1))
        np.testing.assert_allclose(self.egg.get_mass().cpu().numpy(), mass, rtol=1e-7)
        np.testing.assert_allclose(self.egg.get_links_COM().cpu().numpy(), com[None], rtol=1e-7, atol=1e-12)
        self.arm = [self.robot.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        self.fingers = [self.robot.get_joint(f"finger_joint{i}").dofs_idx_local[0] for i in (1, 2)]
        self.robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000], self.arm)
        self.robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200], self.arm)
        self.robot.set_dofs_kp([100, 100], self.fingers)
        self.robot.set_dofs_kv([10, 10], self.fingers)
        self.initial_joints = torch.tensor(
            [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04],
            dtype=gs.tc_float,
            device=gs.device,
        ).repeat(config.environments, 1)
        self.initial_position = torch.tensor([0.65, 0, 0.031], dtype=gs.tc_float, device=gs.device).repeat(
            config.environments, 1
        )
        quaternions = np.tile([1.0, 0, 0, 0], (config.environments, 1))
        radii = np.full(config.environments, config.radius_m)
        egg_friction, robot_friction = np.ones(config.environments), np.ones(config.environments)
        offsets = np.zeros((config.environments, 3))
        if config.varied:
            for environment in range(config.environments):
                rng = np.random.default_rng(config.seed + environment)
                offsets[environment] = rng.uniform([-0.003, -0.004, 0], [0.003, 0.004, 0.002])
                roll, pitch, yaw = rng.uniform([-0.05, -0.05, -0.4], [0.05, 0.05, 0.4])
                cr, cpitch, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
                sr, spitch, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
                quaternions[environment] = [
                    cr * cpitch * cy + sr * spitch * sy,
                    sr * cpitch * cy - cr * spitch * sy,
                    cr * spitch * cy + sr * cpitch * sy,
                    cr * cpitch * sy - sr * spitch * cy,
                ]
                radii[environment] *= rng.uniform(0.8, 1.2)
                egg_friction[environment], robot_friction[environment] = rng.uniform(0.7, 1.4), rng.uniform(0.7, 1.3)
        self.initial_position += torch.as_tensor(offsets, dtype=gs.tc_float, device=gs.device)
        self.initial_quaternion = torch.as_tensor(quaternions, dtype=gs.tc_float, device=gs.device)
        self.base_radius_m = torch.as_tensor(radii[:, None], dtype=torch.float64, device=gs.device)
        self.radius_m = self.base_radius_m.clone()
        self.egg_friction = torch.as_tensor(egg_friction[:, None], dtype=gs.tc_float, device=gs.device)
        self.robot_friction = torch.as_tensor(robot_friction[:, None], dtype=gs.tc_float, device=gs.device).expand(
            -1, self.robot.n_links
        )
        self.robot.set_qpos(self.initial_joints)
        self.egg.set_pos(self.initial_position)
        self.egg.set_quat(self.initial_quaternion)
        self.egg.set_friction_ratio(self.egg_friction)
        self.robot.set_friction_ratio(self.robot_friction)
        self.hand = self.robot.get_link("hand")
        orientation = torch.tensor([0, 1, 0, 0], dtype=gs.tc_float, device=gs.device).repeat(config.environments, 1)
        pickup = self.initial_position.clone()
        pickup[:, 2] = 0.135
        raised = pickup.clone()
        raised[:, 2] += 0.12
        translated = raised.clone()
        translated[:, 1] += 0.025
        self.pick_joints = self.robot.inverse_kinematics(link=self.hand, pos=pickup, quat=orientation)
        self.lift_joints = self.robot.inverse_kinematics(link=self.hand, pos=raised, quat=orientation)
        self.slide_joints = self.robot.inverse_kinematics(link=self.hand, pos=translated, quat=orientation)
        self.delay = (
            (np.arange(config.environments) * 37) % config.episode_steps
            if config.asynchronous
            else np.zeros(config.environments, dtype=np.int64)
        )
        self.device_delay = torch.as_tensor(self.delay, dtype=gs.tc_float, device=gs.device)
        self.initial_delay = self.delay.copy()
        self.phase = torch.zeros(config.environments, dtype=gs.tc_float, device=gs.device)
        self.observer: StressSubstepObserver | None = None
        if recovery is not None:
            mapper = PadPressureGPU(model.surface, anchor_to_surface=True, sampling=config.sampling)
            self.observer = StressSubstepObserver(
                self.scene,
                self.egg,
                mapper,
                recovery,
                cp.from_dlpack(self.radius_m),
                config.substeps,
                config.dt_s,
            )
        self.com_body = torch.as_tensor(com, dtype=gs.tc_float, device=gs.device)

    def reset(self, environments: np.ndarray | None = None, restart_clock: bool = True) -> None:
        ids = np.arange(self.config.environments) if environments is None else environments
        self.scene.reset(envs_idx=ids)
        self.robot.set_qpos(self.initial_joints[ids], envs_idx=ids)
        self.egg.set_pos(self.initial_position[ids], envs_idx=ids)
        self.egg.set_quat(self.initial_quaternion[ids], envs_idx=ids)
        self.egg.set_friction_ratio(self.egg_friction[ids], envs_idx=ids)
        self.robot.set_friction_ratio(self.robot_friction[ids], envs_idx=ids)
        self.phase[ids] = 0
        self.radius_m[ids] = self.base_radius_m[ids]
        if restart_clock:
            self.delay[ids] = self.tick
            self.device_delay[ids] = self.tick
        if self.observer is not None:
            self.observer.reset(cp.asarray(ids))
        self.reset_count += len(ids)

    def restart(self) -> None:
        self.tick = 0
        self.delay[:] = self.initial_delay
        self.device_delay[:] = torch.as_tensor(self.initial_delay, dtype=gs.tc_float, device=gs.device)
        self.reset(restart_clock=False)
        self.reset_count = 0

    def observation(self) -> torch.Tensor:
        peak = torch.zeros_like(self.phase) if self.observer is None else self.observer.observation().to(gs.tc_float)
        return torch.cat(
            (
                self.robot.get_dofs_position(),
                self.robot.get_dofs_velocity(),
                self.egg.get_pos(),
                self.egg.get_vel(),
                self.phase[:, None],
                (peak / 1e6)[:, None],
            ),
            dim=1,
        )

    def step(self, residual_action: torch.Tensor | None = None) -> None:
        reset_ids = np.flatnonzero(
            (self.tick > self.delay) & ((self.tick - self.delay) % self.config.episode_steps == 0)
        )
        if len(reset_ids):
            self.reset(reset_ids, restart_clock=False)
        phase = torch.remainder(self.tick - self.device_delay, self.config.episode_steps) / (
            self.config.episode_steps - 1
        )
        self.phase = torch.where(self.tick >= self.device_delay, phase, 0)
        approach = smoothstep(self.phase / 0.15)[:, None]
        lift = smoothstep((self.phase - 0.3) / 0.2)[:, None]
        slide = smoothstep((self.phase - 0.6) / 0.15)[:, None]
        target = self.initial_joints * (1 - approach) + self.pick_joints * approach
        target += (self.lift_joints - self.pick_joints) * lift + (self.slide_joints - self.lift_joints) * slide
        if residual_action is not None:
            target[:, self.arm] += residual_action[:, :7] * 1e-4
        self.robot.control_dofs_position(target[:, self.arm], self.arm)
        grip = torch.where((self.phase < 0.15) | (self.phase > 0.9), 0.04, 0.018)
        limit = torch.where((self.phase >= 0.7) & (self.phase < 0.76), 0.055, 4.0)[:, None].expand(-1, 2)
        self.robot.set_dofs_force_range(-limit, limit, self.fingers)
        self.robot.control_dofs_position(grip[:, None].expand(-1, 2), self.fingers)
        self.radius_m[:] = self.base_radius_m * (
            1 + self.config.radius_modulation * torch.sin(2 * torch.pi * self.phase[:, None])
        )
        self.scene.step(update_visualizer=self.camera is not None)
        self.tick += 1
