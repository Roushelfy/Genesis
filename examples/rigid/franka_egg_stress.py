"""Rigid Panda grasp with optional native, device-resident shell stress observation."""

import argparse
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


class FrankaEgg:
    def __init__(
        self,
        n_envs: int,
        level: int = 1,
        stress: bool = True,
        seed: int = 510000,
        varied: bool = False,
        substeps: int = 1,
        viewer: bool = False,
        cooperative_solve: bool = True,
        method: str = "auto",
        inverse_precision: str = "64",
        history_size: int = 0,
        cooperative_pressure: bool = True,
        conditions: Path | None = None,
        surface_inverse: bool = True,
        cooperative_scatter: bool = True,
        cached_peak: bool = True,
        cached_face_bounds: bool = True,
    ):
        self.n_envs = n_envs
        self.tick = 0
        self.reset_count = 0
        self.arm = np.arange(7)
        self.fingers = np.array([7, 8])
        assets = Path(__file__).parent / "assets" / "hollow_egg" / f"level{level}"
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, substeps=substeps),
            rigid_options=gs.options.RigidOptions(
                batch_dofs_info=True,
                friction_cone=gs.friction_cone.elliptic,
                contact_resolution=gs.contact_resolution.convex,
                iterations=100,
                tolerance=1e-12,
                contact_pruning_tolerance=None,
                enable_torsional_friction=False,
                enable_rolling_friction=False,
                use_hibernation=False,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=viewer,
        )
        self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=0.1))
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Rigid(friction=0.1)
        )
        self.egg = self.scene.add_entity(
            gs.morphs.URDF(
                file=assets / "egg_shell.urdf", pos=(0.65, 0, 0.031), align=False, convexify=True, decimate=False
            ),
            material=gs.materials.Rigid(friction=0.6),
        )
        self.link = self.egg.base_link
        if stress:
            self.link.configure_stress_recovery(
                gs.options.RigidStressOptions(
                    mesh=assets / "elastic.npz",
                    cooperative_solve=cooperative_solve,
                    method=method,
                    inverse_precision=inverse_precision,
                    history_size=history_size,
                    cooperative_pressure=cooperative_pressure,
                    surface_inverse=surface_inverse,
                    cooperative_scatter=cooperative_scatter,
                    cached_peak=cached_peak,
                    cached_face_bounds=cached_face_bounds,
                )
            )
        self.scene.build(n_envs=n_envs)
        self.robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000], self.arm)
        self.robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200], self.arm)
        self.robot.set_dofs_kp([100, 100], self.fingers)
        self.robot.set_dofs_kv([10, 10], self.fingers)
        self.initial_joints = torch.tensor(
            [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04],
            dtype=gs.tc_float,
            device=gs.device,
        ).repeat(n_envs, 1)
        self.condition_count = n_envs
        if conditions is not None:
            with np.load(conditions, allow_pickle=False) as bank:
                self.condition_count = len(bank["position"])
                case_ids = np.arange(n_envs) % self.condition_count
                self.initial_joints = torch.as_tensor(bank["initial_joints"][case_ids], device=gs.device)
                positions = bank["position"][case_ids]
                quaternions = bank["quaternion"][case_ids]
                radii = bank["radius"][case_ids]
                friction = bank["friction"][case_ids, 0]
                self.pick_joints = torch.as_tensor(bank["pick_joints"][case_ids], device=gs.device)
                self.lift_joints = torch.as_tensor(bank["lift_joints"][case_ids], device=gs.device)
                self.slide_joints = torch.as_tensor(bank["slide_joints"][case_ids], device=gs.device)
                self.delays = bank["delays"][case_ids]
        else:
            positions = np.tile([0.65, 0.0, 0.031], (n_envs, 1))
            quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (n_envs, 1))
            radii = np.full(n_envs, 0.006)
            friction = np.ones(n_envs)
        if varied and conditions is None:
            for i_b in range(n_envs):
                random = np.random.default_rng(seed + i_b)
                positions[i_b] += random.uniform([-0.003, -0.004, 0], [0.003, 0.004, 0.002])
                yaw = random.uniform(-0.4, 0.4)
                quaternions[i_b] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                radii[i_b] *= random.uniform(0.8, 1.2)
                friction[i_b] = random.uniform(0.7, 1.4)
        self.initial_position = torch.as_tensor(positions, dtype=gs.tc_float, device=gs.device)
        self.initial_quaternion = torch.as_tensor(quaternions, dtype=gs.tc_float, device=gs.device)
        self.base_radius = torch.as_tensor(radii, dtype=gs.tc_float, device=gs.device)
        self.friction = torch.as_tensor(friction[:, None], dtype=gs.tc_float, device=gs.device)
        self.robot.set_qpos(self.initial_joints)
        self.egg.set_pos(self.initial_position)
        self.egg.set_quat(self.initial_quaternion)
        self.egg.set_friction_ratio(self.friction)
        self.hand = self.robot.get_link("hand")
        orientation = torch.tensor([0, 1, 0, 0], dtype=gs.tc_float, device=gs.device).repeat(n_envs, 1)
        pickup = self.initial_position.clone()
        pickup[:, 2] = 0.135
        raised = pickup.clone()
        raised[:, 2] += 0.12
        slide = raised.clone()
        slide[:, 1] += 0.025
        if conditions is None:
            self.pick_joints = self.robot.inverse_kinematics(link=self.hand, pos=pickup, quat=orientation)
            self.lift_joints = self.robot.inverse_kinematics(link=self.hand, pos=raised, quat=orientation)
            self.slide_joints = self.robot.inverse_kinematics(link=self.hand, pos=slide, quat=orientation)
            self.delays = (np.arange(n_envs) * 37) % 600 if varied else np.zeros(n_envs, dtype=np.int64)
        self.device_delays = torch.as_tensor(self.delays, dtype=gs.tc_float, device=gs.device)
        self.phase = torch.zeros(n_envs, dtype=gs.tc_float, device=gs.device)
        self.stress = stress

    def save_conditions(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            initial_joints=tensor_to_array(self.initial_joints),
            position=tensor_to_array(self.initial_position),
            quaternion=tensor_to_array(self.initial_quaternion),
            radius=tensor_to_array(self.base_radius),
            friction=tensor_to_array(self.friction),
            pick_joints=tensor_to_array(self.pick_joints),
            lift_joints=tensor_to_array(self.lift_joints),
            slide_joints=tensor_to_array(self.slide_joints),
            delays=self.delays,
        )

    def reset(self, envs_idx=None) -> None:
        ids = np.arange(self.n_envs) if envs_idx is None else envs_idx
        self.scene.reset(envs_idx=ids)
        self.robot.set_qpos(self.initial_joints[ids], envs_idx=ids)
        self.egg.set_pos(self.initial_position[ids], envs_idx=ids)
        self.egg.set_quat(self.initial_quaternion[ids], envs_idx=ids)
        self.egg.set_friction_ratio(self.friction[ids], envs_idx=ids)
        self.phase[ids] = 0.0
        self.reset_count += len(ids)

    def restart(self) -> None:
        self.tick = 0
        self.reset()
        self.reset_count = 0

    def observation(self) -> torch.Tensor:
        stress = self.link.get_max_stress(copy=False) if self.stress else torch.zeros_like(self.phase)
        return torch.cat(
            (
                self.robot.get_dofs_position(),
                self.robot.get_dofs_velocity(),
                self.egg.get_pos(),
                self.egg.get_vel(),
                self.phase[:, None],
                stress[:, None] / 1e6,
            ),
            dim=1,
        )

    def step(self, residual_action: torch.Tensor | None = None) -> None:
        ids = np.flatnonzero((self.tick > self.delays) & ((self.tick - self.delays) % 600 == 0))
        if len(ids):
            self.reset(ids)
        phase = torch.remainder(self.tick - self.device_delays, 600) / 599
        self.phase = torch.where(self.tick >= self.device_delays, phase, 0.0)
        approach = torch.clamp(self.phase / 0.15, 0.0, 1.0)[:, None]
        approach = approach * approach * (3.0 - 2.0 * approach)
        lift = torch.clamp((self.phase - 0.3) / 0.2, 0.0, 1.0)[:, None]
        lift = lift * lift * (3.0 - 2.0 * lift)
        slide = torch.clamp((self.phase - 0.6) / 0.15, 0.0, 1.0)[:, None]
        slide = slide * slide * (3.0 - 2.0 * slide)
        target = self.initial_joints * (1.0 - approach) + self.pick_joints * approach
        target += (self.lift_joints - self.pick_joints) * lift + (self.slide_joints - self.lift_joints) * slide
        if residual_action is not None:
            target[:, self.arm] += 1e-4 * residual_action
        self.robot.control_dofs_position(target[:, self.arm], self.arm)
        grip = torch.where((self.phase < 0.15) | (self.phase > 0.9), 0.04, 0.018)
        limit = torch.where((self.phase >= 0.7) & (self.phase < 0.76), 0.055, 4.0)[:, None].expand(-1, 2)
        self.robot.set_dofs_force_range(-limit, limit, self.fingers)
        self.robot.control_dofs_position(grip[:, None].expand(-1, 2), self.fingers)
        if self.stress:
            radius = self.base_radius * (1.0 + 0.15 * torch.sin(2.0 * torch.pi * self.phase))
            self.link.set_stress_contact_radius(radius[:, None])
        self.scene.step()
        self.tick += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--no-stress", action="store_true")
    parser.add_argument("--varied", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64")
    workload = FrankaEgg(args.envs, args.level, not args.no_stress, varied=args.varied, viewer=args.viewer)
    for _ in range(args.steps):
        workload.step()
    workload.scene.rigid_solver.check_errno()
    if workload.stress:
        print("Maximum VM stress, Pa:", tensor_to_array(workload.link.get_max_stress()))


if __name__ == "__main__":
    main()
