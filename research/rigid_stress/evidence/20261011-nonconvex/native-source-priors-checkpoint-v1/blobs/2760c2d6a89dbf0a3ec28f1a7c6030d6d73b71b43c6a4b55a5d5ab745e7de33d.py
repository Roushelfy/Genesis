"""Quadrants configuration inputs for the scripted Franka egg workload."""

from collections import OrderedDict

import numpy as np
import quadrants as qd
import torch

import genesis as gs


class ResetIndexCache:
    """Bounded configuration selectors; auxiliary physics state is never cached."""

    def __init__(self, n_envs, enabled):
        self.enabled = enabled
        self.capacity = n_envs
        self.buffers = OrderedDict()
        self.elements = 0
        self.hits = self.misses = self.evictions = self.single_selections = 0

    def select(self, ids):
        if not self.enabled or not isinstance(ids, np.ndarray) or ids.ndim != 1:
            return ids
        if not np.issubdtype(ids.dtype, np.integer) or not 0 < len(ids) <= self.capacity:
            return ids
        key = tuple(int(value) for value in ids)
        if len(key) == 1:
            self.single_selections += 1
            # Generic Torch masking calls item() on a one-element device
            # tensor; an ordinary integer avoids that synchronization.
            return key[0]
        if key in self.buffers:
            self.hits += 1
            self.buffers.move_to_end(key)
            return self.buffers[key]
        self.misses += 1
        while self.elements + len(key) > self.capacity:
            evicted, _ = self.buffers.popitem(last=False)
            self.elements -= len(evicted)
            self.evictions += 1
        selection = torch.as_tensor(ids.copy(), dtype=torch.int64, device=gs.device)
        self.buffers[key] = selection
        self.elements += len(key)
        return selection

    def describe(self):
        return {
            "enabled": self.enabled,
            "capacity_elements": self.capacity,
            "stored_elements": self.elements,
            "bytes": self.elements * 8,
            "selectors": len(self.buffers),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "single_selections": self.single_selections,
        }


@qd.kernel
def kernel_inputs(
    tick: int,
    delays: qd.types.ndarray(),
    initial: qd.types.ndarray(),
    pick: qd.types.ndarray(),
    lift_joints: qd.types.ndarray(),
    slide_joints: qd.types.ndarray(),
    base_radius: qd.types.ndarray(),
    arm: qd.types.ndarray(),
    action: qd.types.ndarray(),
    phase: qd.types.ndarray(),
    target: qd.types.ndarray(),
    grip: qd.types.ndarray(),
    limit: qd.types.ndarray(),
    radius: qd.types.ndarray(),
    action_present: qd.template(),
    action_f32: qd.template(),
    stress: qd.template(),
):
    for i_b in range(delays.shape[0]):
        value = gs.qd_float(0.0)
        if tick >= delays[i_b]:
            value = gs.qd_float((tick - delays[i_b]) % 600) / 599.0
        phase[i_b] = value
        approach = qd.min(1.0, qd.max(0.0, value / 0.15))
        approach = approach * approach * (3.0 - 2.0 * approach)
        lift = qd.min(1.0, qd.max(0.0, (value - 0.3) / 0.2))
        lift = lift * lift * (3.0 - 2.0 * lift)
        slide = qd.min(1.0, qd.max(0.0, (value - 0.6) / 0.15))
        slide = slide * slide * (3.0 - 2.0 * slide)
        for j in range(target.shape[1]):
            joint = initial[i_b, j] * (1.0 - approach) + pick[i_b, j] * approach
            joint += (lift_joints[i_b, j] - pick[i_b, j]) * lift + (slide_joints[i_b, j] - lift_joints[i_b, j]) * slide
            if qd.static(action_present):
                for a in range(arm.shape[0]):
                    if j == arm[a]:
                        residual = gs.qd_float(0.0)
                        if qd.static(action_f32):
                            residual = gs.qd_float(qd.f32(action[i_b, a]) * qd.f32(1e-4))
                        else:
                            residual = gs.qd_float(action[i_b, a]) * 1e-4
                        joint += residual
            target[i_b, j] = joint
        # The existing scalar torch.where outputs use Torch's default dtype.
        # Store to that same dtype before the ordinary Genesis setters convert it.
        grip[i_b] = 0.04 if value < 0.15 or value > 0.9 else 0.018
        limit[i_b] = 0.055 if value >= 0.7 and value < 0.76 else 4.0
        if qd.static(stress):
            radius[i_b] = base_radius[i_b] * (1.0 + 0.15 * qd.sin(6.283185307179586 * value))


def create_workspace(workload):
    workspace = {
        "target": torch.empty_like(workload.initial_joints),
        "grip": torch.empty(workload.n_envs, device=gs.device, dtype=torch.get_default_dtype()),
        "limit": torch.empty(workload.n_envs, device=gs.device, dtype=torch.get_default_dtype()),
        "radius": torch.empty_like(workload.base_radius),
        "zero_action": torch.zeros((workload.n_envs, len(workload.arm)), device=gs.device, dtype=gs.tc_float),
        "arm_indices": torch.as_tensor(workload.arm, device=gs.device, dtype=gs.tc_int),
    }
    return workspace


def native_step(workload, workspace, residual_action=None):
    ids = np.flatnonzero((workload.tick > workload.delays) & ((workload.tick - workload.delays) % 600 == 0))
    if len(ids):
        workload.reset(ids)
    kernel_inputs(
        workload.tick,
        workload.device_delays,
        workload.initial_joints,
        workload.pick_joints,
        workload.lift_joints,
        workload.slide_joints,
        workload.base_radius,
        workspace["arm_indices"],
        workspace["zero_action"] if residual_action is None else residual_action,
        workload.phase,
        workspace["target"],
        workspace["grip"],
        workspace["limit"],
        workspace["radius"],
        residual_action is not None,
        residual_action is not None and residual_action.dtype == torch.float32,
        workload.stress,
    )
    workload.robot.control_dofs_position(workspace["target"][:, :7], range(7))
    limit = workspace["limit"][:, None].expand(-1, 2)
    workload.robot.set_dofs_force_range(-limit, limit, range(7, 9))
    workload.robot.control_dofs_position(workspace["grip"][:, None].expand(-1, 2), range(7, 9))
    if workload.stress:
        workload.link.set_stress_contact_radius(workspace["radius"][:, None])
    workload.scene.step()
    workload.tick += 1
