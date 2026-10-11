"""Compare one Quadrants controller-input pass with the same scripted Torch inputs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark import rigid_stress as benchmark
from research.rigid_stress import native_trajectory_oracle as oracle

workspaces = {}


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
    workspaces[id(workload)] = workspace
    return workspace


def native_step(workload, residual_action=None):
    ids = np.flatnonzero((workload.tick > workload.delays) & ((workload.tick - workload.delays) % 600 == 0))
    if len(ids):
        workload.reset(ids)
    workspace = workspaces[id(workload)]
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
    workload.robot.control_dofs_position(workspace["target"][:, workload.arm], workload.arm)
    limit = workspace["limit"][:, None].expand(-1, 2)
    workload.robot.set_dofs_force_range(-limit, limit, workload.fingers)
    workload.robot.control_dofs_position(workspace["grip"][:, None].expand(-1, 2), workload.fingers)
    if workload.stress:
        workload.link.set_stress_contact_radius(workspace["radius"][:, None])
    workload.scene.step()
    workload.tick += 1


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-controller", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(remaining)
    if args.native_controller:
        original = FrankaEgg.__init__

        def initialize(workload, *call_args, **kwargs):
            original(workload, *call_args, **kwargs)
            create_workspace(workload)

        FrankaEgg.__init__ = initialize
        FrankaEgg.step = native_step
    sys.argv = ["native_controller_override", *remaining]
    (oracle.main if args.oracle else benchmark.main)()
    output.output.with_suffix(".controller-variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "native_controller": args.native_controller,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "additional_controller_storage": [
                    {
                        name: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "bytes": value.numel() * value.element_size(),
                        }
                        for name, value in workspace.items()
                    }
                    for workspace in workspaces.values()
                ],
                "note": "Same timestep, targets, weak-grip/release phases, contact parameter schedule, public setters, resets and policy residual. Quadrants computes controller configuration inputs; stress numerical functions are unmodified. CUDA and CPU FP64 same-mesh oracles remain required.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
