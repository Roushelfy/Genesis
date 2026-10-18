"""Finite implicit servo enforcement of the frozen shared motion, with measured tracking errors."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import genesis as gs

from .artifacts import file_id


@dataclass
class PrescribedMotion:
    path: Path
    q: torch.Tensor
    positions: torch.Tensor
    quaternions: torch.Tensor
    stride: int
    dt: float
    damping: float

    @classmethod
    def configure(cls, robot, path, dt, damping):
        with np.load(path) as saved:
            sample_dt = (saved["time_s"][1] - saved["time_s"][0]).item()
            stride = round(dt / sample_dt)
            assert stride > 0 and np.isclose(stride * sample_dt, dt)
            assert dt <= 0.01 and np.isclose(round(0.01 / dt) * dt, 0.01)
            names = saved["link_names"].tolist()
            indices = [names.index(link.name) for link in robot.links]
            q = torch.as_tensor(saved["q"], dtype=gs.tc_float, device=gs.device)
            positions = torch.as_tensor(saved["link_origin_world_m"][:, indices], dtype=gs.tc_float, device=gs.device)
            quaternions = torch.as_tensor(
                saved["link_quaternion_wxyz"][:, indices], dtype=gs.tc_float, device=gs.device
            )
        # With implicit damping, kp=kv/dt enforces v_new=(q_next-q_old)/dt
        # as damping*dt dominates inertia. Tracking and gain sensitivity are
        # measured; this is explicitly separate from the capped controller.
        robot.set_dofs_kp([damping / dt] * robot.n_dofs)
        robot.set_dofs_kv([damping] * robot.n_dofs)
        robot.set_dofs_force_range([-float("inf")] * robot.n_dofs, [float("inf")] * robot.n_dofs)
        return cls(path, q, positions, quaternions, stride, dt, damping)

    def advance(self, workload, step, native_stress):
        sample = (step + 1) * self.stride
        assert sample < len(self.q)
        workload.robot.control_dofs_position(self.q[sample][None])
        control_tick = step // round(0.01 / self.dt)
        phase = torch.remainder(control_tick - workload.device_delays, 600) / 599
        phase = torch.where(control_tick >= workload.device_delays, phase, 0.0)
        if native_stress:
            radius = workload.base_radius * (1 + 0.15 * torch.sin(2 * torch.pi * phase))
            workload.link.set_stress_contact_radius(radius[:, None])
        workload.scene.step()
        return phase

    def metrics(self, robot, step):
        sample = (step + 1) * self.stride
        q_error = (robot.get_qpos()[0] - self.q[sample]).abs()
        position_error = torch.linalg.norm(robot.get_links_pos()[0] - self.positions[sample], dim=1)
        actual_q = robot.get_links_quat()[0]
        target_q = self.quaternions[sample]
        sign = torch.where((actual_q * target_q).sum(dim=1) >= 0, 1, -1)
        difference = torch.linalg.norm(actual_q - sign[:, None] * target_q, dim=1)
        angle_error = 4 * torch.asin((difference / 2).clamp(max=1))
        return {
            "maximum_arm_tracking_error_rad": q_error[:7].max().item(),
            "maximum_finger_tracking_error_m": q_error[7:].max().item(),
            "maximum_robot_authored_origin_error_m": position_error.max().item(),
            "maximum_robot_rotation_tracking_error_rad": angle_error.max().item(),
        }

    def identity(self):
        return {
            "motion": file_id(self.path),
            "tracking_damping": self.damping,
            "tracking_stiffness": self.damping / self.dt,
            "force_caps_overridden": True,
            "enforcement": "finite implicit joint servo; report actual tracking and gain sensitivity",
        }
