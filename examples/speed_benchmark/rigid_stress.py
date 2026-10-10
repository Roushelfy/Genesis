"""Reproducible low-resolution native rigid stress timing and kernel profiling.

Set QD_KERNEL_PROFILER=1 for --scope profile. Use separate processes for matched scopes.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.contact import kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.solve import (
    kernel_balance,
    kernel_direct_init,
    kernel_full_residual,
    kernel_peak,
)
from genesis.utils.misc import qd_to_numpy, tensor_to_array


def profile(workload: FrankaEgg, repetitions: int) -> dict:
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]
    solver = workload.scene.rigid_solver
    options = entry.link.stress_options
    stages = {
        "begin_step": lambda: kernel_begin_step(entry.state),
        "association_transform": lambda: kernel_associate(
            entry.link.idx,
            solver._links_offset_pos,
            solver._links_offset_quat,
            entry.omega,
            solver.dyn_state,
            entry.contacts,
            solver.collider.collider_state,
            solver._links_offset_quat.ndim == 3,
            not solver._disable_constraint,
        ),
        "anchor_search": lambda: kernel_anchor(recovery.source_epsilon, entry.contacts, entry.surface.info),
        "candidate_weights_constraints": lambda: kernel_pressure(entry.contacts, entry.surface.info),
        "nodal_scatter": lambda: kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info),
        "inertia_relief_centrifugal": lambda: kernel_balance(entry.omega, entry.state, entry.model.info),
        "rhs_reduction": lambda: kernel_direct_init(entry.state),
        "linear_solve": lambda: entry.model.factor.solve(options.young, entry.state, entry.model.info),
        "complete_residual": lambda: kernel_full_residual(
            options.young, options.tolerance, options.absolute_tolerance, entry.state, entry.model.info
        ),
        "global_peak": lambda: kernel_peak(options.young, options.poisson, entry.state, entry.model.info),
        "accept": lambda: kernel_accept(entry.state, entry.contacts, solver._errno),
    }
    for stage in stages.values():
        stage()
    qd.sync()
    qd.profiler.clear_kernel_profiler_info()
    times = {}
    for name, stage in stages.items():
        start = time.perf_counter()
        for _ in range(repetitions):
            stage()
        qd.sync()
        times[name] = (time.perf_counter() - start) * 1e3 / repetitions
    qd.profiler.print_kernel_profiler_info()
    solver.check_errno()
    contacts = qd_to_numpy(entry.contacts.valid, transpose=True)
    return {
        "stage_wall_ms": times,
        "contacts_per_environment": contacts.sum(axis=1).tolist(),
        "candidate_count": qd_to_numpy(entry.contacts.candidate_count, transpose=True)[contacts].tolist(),
        "pressure_evaluations": qd_to_numpy(entry.contacts.evaluations, transpose=True)[contacts].tolist(),
        "complete_residual_N": np.sqrt(qd_to_numpy(entry.state.residual_norm_squared)).tolist(),
        "rhs_norm_N": np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=("profile", "rigid", "recovery", "live", "policy", "policy-rigid"), required=True
    )
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--minimum-seconds", type=float, default=10.0)
    parser.add_argument("--varied", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    start = time.perf_counter()
    workload = FrankaEgg(args.envs, args.level, args.scope not in ("rigid", "policy-rigid"), varied=args.varied)
    setup_seconds = time.perf_counter() - start
    policy = (
        torch.nn.Sequential(
            torch.nn.Linear(26, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 7),
        )
        .to(device=gs.device, dtype=gs.tc_float)
        .eval()
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            action = policy(workload.observation()) if args.scope.startswith("policy") else None
            workload.step(action)
        qd.sync()
        workload.scene.rigid_solver.check_errno()
        result = {
            "scope": args.scope,
            "envs": args.envs,
            "level": args.level,
            "precision": "64",
            "quadrature": 10,
            "dt": 0.01,
            "substeps": 1,
            "varied": args.varied,
            "seed": 510000,
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_bytes": torch.cuda.get_device_properties(gs.device).total_memory,
            "quadrants": qd.__version__,
            "torch": torch.__version__,
            "setup_seconds": setup_seconds,
            "warmup_steps": args.warmup,
        }
        if args.scope == "profile":
            result.update(profile(workload, args.steps))
        else:
            rows = []
            for _ in range(args.repetitions):
                if args.scope != "recovery":
                    workload.restart()
                qd.sync()
                start = time.perf_counter()
                steps = 0
                while steps < args.steps or time.perf_counter() - start < args.minimum_seconds:
                    if args.scope == "recovery":
                        workload.scene.rigid_solver.stress_recovery.recover(0)
                    else:
                        action = policy(workload.observation()) if args.scope.startswith("policy") else None
                        workload.step(action)
                    steps += 1
                qd.sync()
                elapsed = time.perf_counter() - start
                workload.scene.rigid_solver.check_errno()
                rows.append(
                    {
                        "steps": steps,
                        "seconds": elapsed,
                        "transitions_per_second": args.envs * steps / elapsed,
                        "batch_steps_per_second": steps / elapsed,
                        "resets": workload.reset_count,
                    }
                )
            result["repeats"] = rows
        result["egg_position_m"] = tensor_to_array(workload.egg.get_pos()).tolist()
        if workload.stress:
            entry = workload.scene.rigid_solver.stress_recovery.links[0]
            result["dofs"] = entry.model.info.vertices.shape[0] * 3
            result["tetrahedra"] = entry.model.info.elements.shape[0]
            result["maximum_stress_Pa"] = tensor_to_array(workload.link.get_max_stress()).tolist()
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["device_free_bytes"] = torch.cuda.mem_get_info()[0]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
