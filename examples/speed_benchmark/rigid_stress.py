"""Reproducible low-resolution native rigid stress timing and kernel profiling.

Set QD_KERNEL_PROFILER=1 for --scope profile. Use separate processes for matched scopes.
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
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
from genesis.utils.array_class import V_MAT, V_VEC, DataKind, V
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
        "candidate_weights_constraints": lambda: kernel_pressure(
            entry.contacts, entry.surface.info, options.cooperative_pressure
        ),
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
    kernel_pressure(entry.contacts, entry.surface.info, options.cooperative_pressure)
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
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--conditions", type=Path, help="Repeat a frozen condition bank, including controller targets.")
    parser.add_argument("--save-conditions", type=Path, help="Save the constructed condition bank before warmup.")
    parser.add_argument("--serial-solve", action="store_true")
    parser.add_argument("--serial-pressure", action="store_true")
    parser.add_argument("--method", choices=("auto", "direct", "inverse"), default="auto")
    parser.add_argument("--inverse-precision", choices=("64", "32"), default="64")
    parser.add_argument("--history", type=int, choices=(0, 4), default=0)
    parser.add_argument("--trace", action="store_true", help="Export a separate intrusive CUDA/CPU trace pass.")
    parser.add_argument(
        "--compact-log", action="store_true", help="Keep full per-environment diagnostics in JSON only."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
    start = time.perf_counter()
    workload = FrankaEgg(
        args.envs,
        args.level,
        args.scope not in ("rigid", "policy-rigid"),
        seed=args.seed,
        varied=args.varied,
        cooperative_solve=not args.serial_solve,
        method=args.method,
        inverse_precision=args.inverse_precision,
        history_size=args.history,
        cooperative_pressure=not args.serial_pressure,
        conditions=args.conditions,
    )
    if args.save_conditions is not None:
        workload.save_conditions(args.save_conditions)
    setup_seconds = time.perf_counter() - start
    finger_links = np.array([workload.robot.get_link(name).idx for name in ("left_finger", "right_finger")])
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
                collider = workload.scene.rigid_solver.collider.collider_state
                counts = qd_to_numpy(collider.n_contacts)
                source_a = qd_to_numpy(collider.contact_data.link_a, transpose=True)
                source_b = qd_to_numpy(collider.contact_data.link_b, transpose=True)
                source_force = qd_to_numpy(collider.contact_data.force, transpose=True)
                source_normal = qd_to_numpy(collider.contact_data.normal, transpose=True)
                source_order = qd_to_numpy(collider.contact_sort_idx, transpose=True)
                i_b, i_col = np.nonzero(np.arange(source_order.shape[1])[None, :] < counts[:, None])
                ids = source_order[i_b, i_col]
                egg_contacts = (source_a[i_b, ids] == workload.link.idx) | (source_b[i_b, ids] == workload.link.idx)
                i_b, ids = i_b[egg_contacts], ids[egg_contacts]
                forces, normals = source_force[i_b, ids], source_normal[i_b, ids]
                nonzero = np.linalg.norm(forces, axis=1) > 1e-12
                nonzero_contacts = np.bincount(i_b[nonzero], minlength=args.envs)
                finger_contacts = np.column_stack(
                    [
                        np.bincount(
                            i_b[((source_a[i_b, ids] == i_link) | (source_b[i_b, ids] == i_link)) & nonzero],
                            minlength=args.envs,
                        )
                        for i_link in finger_links
                    ]
                )
                tangent = forces - np.sum(forces * normals, axis=1)[:, None] * normals
                tangential = np.bincount(i_b, weights=np.linalg.norm(tangent, axis=1), minlength=args.envs)
                row["finger_contacts"] = finger_contacts.tolist()
                row["nonzero_egg_contacts"] = nonzero_contacts.tolist()
                row["tangential_force_N"] = tangential.tolist()
                if workload.stress:
                    entry = workload.scene.rigid_solver.stress_recovery.links[0]
                    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
                    row["contacts"] = valid.sum(axis=1).tolist()
                    row["maximum_stress_Pa"] = qd_to_numpy(entry.state.step_peak).tolist()
                    row["residual_N"] = np.sqrt(qd_to_numpy(entry.state.residual_norm_squared)).tolist()
                    row["rhs_N"] = np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared)).tolist()
                    row["corrections"] = qd_to_numpy(entry.state.corrections).tolist()
                    row["fallbacks"] = qd_to_numpy(entry.state.fallbacks).tolist()
                    row["contact_integration_retries"] = (
                        qd_to_numpy(entry.contacts.is_refined, transpose=True).sum(axis=1).tolist()
                    )
                    evaluations = qd_to_numpy(entry.contacts.evaluations, transpose=True)
                    row["contact_fit_iterations_max"] = evaluations.max(axis=1).tolist()
                    row["contact_wrench_force_error_N"] = (
                        qd_to_numpy(entry.contacts.force_error, transpose=True).max(axis=1).tolist()
                    )
                    row["contact_wrench_moment_error_Nm"] = (
                        qd_to_numpy(entry.contacts.moment_error, transpose=True).max(axis=1).tolist()
                    )
                    if entry.history is not None:
                        row["history_hits"] = qd_to_numpy(entry.history.state.hit).tolist()
                    radii = qd_to_numpy(entry.contacts.radius, transpose=True)[valid]
                    friction = qd_to_numpy(entry.contacts.friction, transpose=True)[valid]
                    row["contact_radius_range_m"] = [radii.min(), radii.max()] if len(radii) else []
                    row["contact_friction_range"] = [friction.min(), friction.max()] if len(friction) else []
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
            "seed": args.seed,
            "load_model": "finite_pad_adaptive_q10",
            "condition_count": workload.condition_count,
            "conditions": str(args.conditions) if args.conditions is not None else None,
            "conditions_sha256": hashlib.sha256(args.conditions.read_bytes()).hexdigest()
            if args.conditions is not None
            else None,
            "cooperative_solve": not args.serial_solve,
            "cooperative_pressure": not args.serial_pressure,
            "method": args.method,
            "inverse_precision": args.inverse_precision,
            "history": args.history,
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
            result["build_timings_including_JIT"] = asdict(entry.model.build_timings)
            result["maximum_stress_Pa"] = tensor_to_array(workload.link.get_max_stress()).tolist()
            buffers = []
            configurations = []
            for item in workload.scene.rigid_solver.stress_recovery.data:
                if item.kind == DataKind.CONFIG:
                    configurations.append(item.value.model_dump(mode="json"))
                else:
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
            result["stress_configurations"] = configurations
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
    if args.compact_log:
        for key in ("warmup_quality", "egg_position_m", "maximum_stress_Pa"):
            result.pop(key, None)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
