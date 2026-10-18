"""Map the native capped PD and scalar finger equality onto existing QIPC public inputs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qipc.scene.joint_collection import JointCollection

from .robot_inputs import KP, KV


@dataclass(frozen=True)
class EqualityParameters:
    inverse_weight: float
    time_constant: float
    damping_ratio: float
    minimum_impedance: float
    maximum_impedance: float
    width: float
    midpoint: float
    power: float

    @classmethod
    def from_native(cls, path):
        saved = json.loads(Path(path).read_text())
        weights = np.array(saved["dofs_inverse_weight"]).reshape(-1)
        parameters = np.array(saved["equality_solver_parameters"]).reshape(-1, 7)
        parameters = parameters[parameters[:, 0] > 0]
        assert weights.shape == (9,) and parameters.shape == (1, 7)
        return cls(weights[7:].sum().item(), *parameters[0].tolist())

    @property
    def inertia_coefficient(self):
        return self.minimum_impedance / (self.inverse_weight * (1 - self.minimum_impedance))

    def impedance(self, residual):
        position = residual.abs() / self.width
        lower = position**self.power / self.midpoint ** (self.power - 1)
        upper = 1 - (1 - position) ** self.power / (1 - self.midpoint) ** (self.power - 1)
        blend = torch.where(position < self.midpoint, lower, upper)
        impedance = self.minimum_impedance + blend * (self.maximum_impedance - self.minimum_impedance)
        return torch.where(
            position > 1, self.maximum_impedance, impedance.clamp(self.minimum_impedance, self.maximum_impedance)
        )

    def attach(self, robot, dt):
        # Native equality has R qforce + J qacc = aref. Moving J qacc
        # to the inertia gives C J.T J, C=1/R. A QIPC coupling damping
        # C/dt supplies this same increment operator. Its predictor/bias
        # enters through the paired force below, rather than fitted K/D.
        robot["finger_joint1"].add_linear_coupling(
            robot["finger_joint2"], stiffness=0.0, damping=self.inertia_coefficient / dt
        )

    def predictor_force(self, q, velocity, dt):
        residual = q[7] - q[8]
        relative_velocity = velocity[7] - velocity[8]
        impedance = self.impedance(residual)
        time_constant = max(self.time_constant, 2 * dt)
        damping = 2 / (self.maximum_impedance * time_constant)
        stiffness = 1 / (self.maximum_impedance * time_constant * self.damping_ratio) ** 2
        force = self.inertia_coefficient * ((1 / dt - damping) * relative_velocity - stiffness * impedance * residual)
        actual_inertia = impedance / (self.inverse_weight * (1 - impedance))
        coefficient_difference = actual_inertia / self.inertia_coefficient - 1
        return force, coefficient_difference


@dataclass
class CappedServo:
    robot: JointCollection
    dt: float
    equality: EqualityParameters | None
    kp: torch.Tensor
    kv: torch.Tensor
    q: torch.Tensor
    velocity: torch.Tensor
    frame_q: torch.Tensor
    frame_velocity: torch.Tensor
    clipped_force: torch.Tensor
    equality_coefficient_difference: torch.Tensor

    @classmethod
    def allocate(cls, robot, dt, equality=None):
        q = robot.get_dofs_position()
        robot.set_dofs_kp(0.0)
        robot.set_dofs_kv(KV)
        robot.set_dofs_force_range(-np.inf, np.inf)
        return cls(
            robot,
            dt,
            equality,
            q.new_tensor(KP),
            q.new_tensor(KV),
            q,
            q.new_zeros(q.shape),
            q.clone(),
            q.new_zeros(q.shape),
            q.new_zeros(q.shape),
            q.new_zeros(()),
        )

    def push(self, target, cap):
        self.frame_q = self.q.clone()
        self.frame_velocity = self.velocity.clone()
        self.clipped_force = torch.clamp(self.kp * (target - self.q) - self.kv * self.velocity, -cap, cap)
        # Native approximate_implicitfast clips the start-frame PD force,
        # then adds the *unclipped* kv Jacobian to the increment operator.
        # QIPC already supplies that operator. Undo its start-velocity
        # contribution so the explicit and implicit portions match.
        force = self.clipped_force + self.kv * self.velocity
        if self.equality is not None:
            paired, self.equality_coefficient_difference = self.equality.predictor_force(self.q, self.velocity, self.dt)
            force[7] += paired
            force[8] -= paired
        self.robot.control_dofs_force(force)

    def update(self):
        self.q = self.robot.get_dofs_position()
        self.velocity = (self.q - self.frame_q) / self.dt

    @property
    def effective_actuator_force(self):
        return self.clipped_force - self.kv * (self.velocity - self.frame_velocity)

    @property
    def equality_work(self):
        if self.equality is None:
            return self.q.new_zeros(())
        paired, _ = self.equality.predictor_force(self.frame_q, self.frame_velocity, self.dt)
        paired -= self.equality.inertia_coefficient / self.dt * (self.velocity[7] - self.velocity[8])
        relative_increment = (self.q[7] - self.frame_q[7]) - (self.q[8] - self.frame_q[8])
        return paired * relative_increment
