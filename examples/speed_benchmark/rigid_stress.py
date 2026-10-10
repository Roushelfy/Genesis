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
from examples.speed_benchmark.stress_stages import kernel_query, kernel_small_solve, kernel_weights_gram
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    kernel_anchor,
    kernel_pressure,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.solve import (
    kernel_balance,
    kernel_direct_init,
    kernel_full_residual,
    kernel_peak,
)
from genesis.utils import array_class, geom
from genesis.utils.array_class import V_MAT, V_VEC, V
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array


@qd.kernel
def kernel_restore_frame(
    i_l: int,
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    batch_offsets: qd.template(),
    omega: qd.Tensor,
    contact_state: StressContactState,
    dyn_state: array_class.DynState,
):
    for i_b in range(omega.shape[0]):
        position = qd.Vector.zero(gs.qd_float, 3)
        quaternion = qd.Vector.zero(gs.qd_float, 4)
        for a in qd.static(range(3)):
            if qd.static(batch_offsets):
                position[a] = offsets_pos[i_b, i_l, a]
            else:
                position[a] = offsets_pos[i_l, a]
        for a in qd.static(range(4)):
            if qd.static(batch_offsets):
                quaternion[a] = offsets_quat[i_b, i_l, a]
            else:
                quaternion[a] = offsets_quat[i_l, a]
        frame = contact_state.frame_quaternion[i_b]
        dyn_state.links.quat[i_l, i_b] = geom.qd_transform_quat_by_quat(quaternion, frame)
        dyn_state.links.pos[i_l, i_b] = contact_state.frame_position[i_b] + geom.qd_transform_by_quat(position, frame)
        dyn_state.links.cd_ang[i_l, i_b] = geom.qd_transform_by_quat(omega[i_b], frame)


def restore_frame(workload: FrankaEgg) -> None:
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    kernel_restore_frame(
        entry.link.idx,
        solver._links_offset_pos,
        solver._links_offset_quat,
        solver._links_offset_quat.ndim == 3,
        entry.omega,
        entry.contacts,
        solver.dyn_state,
    )


def profile(workload: FrankaEgg, repetitions: int) -> dict:
    restore_frame(workload)
    recovery = workload.scene.rigid_solver.stress_recovery
    entry = recovery.links[0]
    solver = workload.scene.rigid_solver
    options = entry.link.stress_options
    contact_shape = entry.contacts.valid.shape
    visited = V(dtype=gs.qd_int, shape=contact_shape)
    inside = V(dtype=gs.qd_int, shape=contact_shape)
    grams = V_MAT(3, 3, dtype=gs.qd_float, shape=contact_shape)
    coefficients = V_VEC(3, dtype=gs.qd_float, shape=contact_shape)
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
            entry.state.step_valid,
            solver._errno,
        ),
        "anchor_search": lambda: kernel_anchor(recovery.source_epsilon, entry.contacts, entry.surface.info),
        "candidate_weights_constraints": lambda: kernel_pressure(entry.contacts, entry.surface.info),
        "nodal_scatter": lambda: kernel_scatter(entry.contacts, entry.state, entry.model.info, entry.surface.info),
        "inertia_relief_centrifugal": lambda: kernel_balance(entry.omega, entry.state, entry.model.info),
        "rhs_reduction": lambda: kernel_direct_init(entry.state),
        "linear_solve": lambda: entry.model.solve(options, entry.state),
        "complete_residual": lambda: kernel_full_residual(
            options.young, options.tolerance, options.absolute_tolerance, entry.state, entry.model.info, False
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
    kernel_anchor(recovery.source_epsilon, entry.contacts, entry.surface.info)
    kernel_pressure(entry.contacts, entry.surface.info)
    solver.check_errno()
    contacts = qd_to_numpy(entry.contacts.valid, transpose=True)
    # Diagnostic subdivision repeats work and is excluded from the additive pipeline stages above.
    subdivisions = {
        "grid_query_distance": lambda: kernel_query(entry.contacts, entry.surface.info, visited, inside),
        "query_weights_gram": lambda: kernel_weights_gram(entry.contacts, entry.surface.info, grams),
        "small_closed_form_solve": lambda: kernel_small_solve(entry.contacts, grams, coefficients),
    }
    micro_times = {}
    for name, stage in subdivisions.items():
        stage()
        qd.sync()
        start = time.perf_counter()
        for _ in range(repetitions):
            stage()
        qd.sync()
        micro_times[name] = (time.perf_counter() - start) * 1e3 / repetitions
    return {
        "stage_wall_ms": times,
        "contacts_per_environment": contacts.sum(axis=1).tolist(),
        "candidate_count": qd_to_numpy(entry.contacts.candidate_count, transpose=True)[contacts].tolist(),
        "pressure_evaluations": qd_to_numpy(entry.contacts.evaluations, transpose=True)[contacts].tolist(),
        "pressure_subdivision_wall_ms": micro_times,
        "grid_samples_visited": qd_to_numpy(visited, transpose=True)[contacts].tolist(),
        "grid_samples_inside": qd_to_numpy(inside, transpose=True)[contacts].tolist(),
        "full_surface_samples": entry.surface.info.positions.shape[0],
        "complete_residual_N": np.sqrt(qd_to_numpy(entry.state.residual_norm_squared)).tolist(),
        "rhs_norm_N": np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared)).tolist(),
        "profiler_note": "Captured Quadrants graphs are omitted by the kernel profiler; stage wall times include them.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=("profile", "rigid", "recovery", "live", "policy", "policy-rigid"), required=True
    )
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--minimum-seconds", type=float, default=10.0)
    parser.add_argument("--varied", action="store_true")
    parser.add_argument("--serial-solve", action="store_true")
    parser.add_argument("--method", choices=("auto", "direct", "inverse"), default="auto")
    parser.add_argument("--inverse-precision", choices=("64", "32"), default="64")
    parser.add_argument("--trace", action="store_true", help="Export a separate intrusive CUDA/CPU trace pass.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    start = time.perf_counter()
    workload = FrankaEgg(
        args.envs,
        args.level,
        args.scope not in ("rigid", "policy-rigid"),
        varied=args.varied,
        cooperative_solve=not args.serial_solve,
        method=args.method,
        inverse_precision=args.inverse_precision,
    )
    setup_seconds = time.perf_counter() - start
    torch.manual_seed(99173)
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
        quality = []
        for tick in range(args.warmup):
            action = policy(workload.observation()) if args.scope.startswith("policy") else None
            workload.step(action)
            if tick % 10 == 0:
                row = {
                    "tick": tick,
                    "phase": tensor_to_array(workload.phase).tolist(),
                    "height_m": tensor_to_array(workload.egg.get_pos())[:, 2].tolist(),
                }
                if workload.stress:
                    entry = workload.scene.rigid_solver.stress_recovery.links[0]
                    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
                    row["contacts"] = valid.sum(axis=1).tolist()
                    row["maximum_stress_Pa"] = qd_to_numpy(entry.state.step_peak).tolist()
                    row["residual_N"] = np.sqrt(qd_to_numpy(entry.state.residual_norm_squared)).tolist()
                    row["rhs_N"] = np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared)).tolist()
                    row["corrections"] = qd_to_numpy(entry.state.corrections).tolist()
                    row["fallbacks"] = qd_to_numpy(entry.state.fallbacks).tolist()
                quality.append(row)
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
            "cooperative_solve": not args.serial_solve,
            "method": args.method,
            "inverse_precision": args.inverse_precision,
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_bytes": torch.cuda.get_device_properties(gs.device).total_memory,
            "quadrants": qd.__version__,
            "torch": torch.__version__,
            "setup_seconds": setup_seconds,
            "warmup_steps": args.warmup,
            "warmup_quality": quality,
            "policy": {"layers": [26, 128, 128, 7], "seed": 99173, "dtype": "float64", "residual_scale_rad": 1e-4},
        }
        if args.scope == "profile":
            result.update(profile(workload, args.steps))
        else:
            if args.scope == "recovery":
                restore_frame(workload)
                result["recovery_input"] = "Frozen actual contact snapshot with its matching pre-integration frame."
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
            result["method_selected"] = entry.model.selected_method
            result["maximum_stress_Pa"] = tensor_to_array(workload.link.get_max_stress()).tolist()
            buffers = []
            for item in workload.scene.rigid_solver.stress_recovery.data:
                view = qd_to_torch(item.value, copy=False)
                buffers.append(
                    {
                        "name": item.name,
                        "shape": list(view.shape),
                        "dtype": str(view.dtype),
                        "bytes": view.numel() * view.element_size(),
                        "kind": int(item.kind),
                    }
                )
            result["native_buffers"] = buffers
            result["native_buffer_bytes"] = sum(item["bytes"] for item in buffers)
            result["native_allocation_count"] = len(buffers)
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["device_free_bytes"] = torch.cuda.mem_get_info()[0]
        if args.trace:
            qd.sync()
            with torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
            ) as profiler:
                for _ in range(50):
                    with torch.profiler.record_function("rigid_stress_trace_step"):
                        action = policy(workload.observation()) if args.scope.startswith("policy") else None
                        workload.step(action)
                qd.sync()
            trace_path = args.output.with_suffix(".trace.json")
            profiler.export_chrome_trace(str(trace_path))
            result["trace"] = trace_path.name
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
