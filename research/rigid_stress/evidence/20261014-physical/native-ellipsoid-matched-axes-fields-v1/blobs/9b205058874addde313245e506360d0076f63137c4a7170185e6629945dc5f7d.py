"""Compare one Quadrants FP64 observation pass with the existing getters."""

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
from genesis.engine.solvers.rigid.abd.accessor import func_link_offset_shift
from genesis.utils import array_class
from genesis.utils.misc import qd_to_torch, tensor_to_array

workspaces = {}


@qd.kernel(fastcache=True)
def kernel_observation(
    dof_start: int,
    egg: int,
    phase: qd.types.ndarray(),
    peak: qd.types.ndarray(),
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    state: array_class.DynState,
    output: qd.types.ndarray(),
    relative: qd.template(),
    stress: qd.template(),
):
    for i_b in range(output.shape[0]):
        for j in qd.static(range(9)):
            output[i_b, j] = state.dofs.pos[dof_start + j, i_b]
            output[i_b, 9 + j] = state.dofs.vel[dof_start + j, i_b]
        shift = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(relative):
            shift = func_link_offset_shift(egg, i_b, offsets_pos, offsets_quat, state)
        position = state.links.pos[egg, i_b] - shift
        delta = state.links.pos[egg, i_b] - state.links.root_COM[egg, i_b] - shift
        velocity = state.links.cd_vel[egg, i_b] + state.links.cd_ang[egg, i_b].cross(delta)
        for j in qd.static(range(3)):
            output[i_b, 18 + j] = position[j]
            output[i_b, 21 + j] = velocity[j]
        output[i_b, 24] = phase[i_b]
        output[i_b, 25] = 0.0
        if qd.static(stress):
            output[i_b, 25] = peak[i_b] / 1e6


def native_observation(workload):
    solver = workload.scene.rigid_solver
    if workload not in workspaces:
        workspaces[workload] = {
            "output": torch.empty((workload.n_envs, 26), dtype=gs.tc_float, device=gs.device),
            "peak": qd_to_torch(solver.stress_recovery.links[0].state.step_peak, copy=False)
            if workload.stress
            else workload.phase,
        }
    if workload.stress:
        entry = solver.stress_recovery.links[0]
        if workload.link.stress_options != entry.options:
            gs.raise_exception("Changing stress recovery options requires rebuilding the scene.")
    workspace = workspaces[workload]
    kernel_observation(
        workload.robot.dof_start,
        workload.link.idx,
        workload.phase,
        workspace["peak"],
        solver._links_offset_pos,
        solver._links_offset_quat,
        solver.dyn_state,
        workspace["output"],
        not solver._links_offset_pos_is_identity[workload.link.idx],
        workload.stress,
    )
    return workspace["output"]


def main():
    command = [sys.executable, *sys.argv]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-observation", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args, remaining = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output_parser.add_argument("--seed", type=int, default=623001)
    output_parser.add_argument("--steps", type=int, default=2400)
    output, _ = output_parser.parse_known_args(remaining)
    original = FrankaEgg.observation
    maxima = np.zeros(26)
    if args.validate:
        gs.init(backend=gs.gpu, precision="64", seed=output.seed, logging_level="warning")
        workload = FrankaEgg(16, varied=True, seed=output.seed, output_mode="full")
        for tick in range(output.steps):
            expected = tensor_to_array(original(workload))
            actual = tensor_to_array(native_observation(workload))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            maxima = np.maximum(maxima, np.max(np.abs(actual - expected), axis=0))
            workload.step()
            workload.scene.rigid_solver.check_errno()
            if tick % 100 == 99:
                print("Observation equivalent tick", tick, flush=True)
    else:
        if args.native_observation:
            FrankaEgg.observation = native_observation
        sys.argv = ["native_observation_override", *remaining]
        benchmark.main()
    output.output.with_suffix(".observation-variant.json").write_text(
        json.dumps(
            {
                "command": command,
                "native_observation": args.native_observation,
                "validation": args.validate,
                "max_abs_error_by_column": maxima.tolist() if args.validate else None,
                "source_revision": os.environ["RIGID_STRESS_SOURCE_REVISION"],
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "additional_observation_bytes": sum(
                    workspace["output"].numel() * workspace["output"].element_size()
                    for workspace in workspaces.values()
                ),
                "note": "FP64 device observation only, same 9 positions/9 velocities/authored egg position and velocity/phase/global stress in MPa. Uses the existing native offset transform, checks immutable stress options, and retains ordinary policy casting/control budget. Stress and rigid numerical functions remain unchanged. Validation covers all 26 columns on changing contacts and partial resets.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
