"""Reproducible low-resolution native rigid stress timing and kernel profiling.

Set QD_KERNEL_PROFILER=1 for --scope profile. Use separate processes for matched scopes.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress_metrics import RolloutMetrics
from examples.speed_benchmark.stress_stages import kernel_query, kernel_small_solve, kernel_weights_gram
from genesis.engine.solvers.rigid.stress.association import kernel_associate
from genesis.engine.solvers.rigid.stress.balance import kernel_balance_contacts
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    kernel_anchor,
    kernel_pressure,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.provenance import kernel_capture_rigid_sources
from genesis.engine.solvers.rigid.stress.recovery import kernel_accept, kernel_begin_step
from genesis.engine.solvers.rigid.stress.scatter import kernel_fit_scatter_moments, kernel_scatter_faces
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
    """Match frozen solved contacts to their saved pre-integration body frame."""
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


def prepare_recovery_snapshot(workload: FrankaEgg) -> dict:
    """Check the independent recovery route before timing its frozen input."""
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    original_force = qd_to_numpy(entry.state.force, transpose=True, copy=True)
    original_peak = qd_to_numpy(entry.state.peak, copy=True)
    original_point = qd_to_numpy(entry.contacts.position, transpose=True, copy=True)
    original_wrench_force = qd_to_numpy(entry.contacts.force, transpose=True, copy=True)
    restore_frame(workload)
    solver.stress_recovery.recover(0)
    qd.sync()
    solver.check_errno()
    force = qd_to_numpy(entry.state.force, transpose=True)
    peak = qd_to_numpy(entry.state.peak)
    point = qd_to_numpy(entry.contacts.position, transpose=True)
    wrench_force = qd_to_numpy(entry.contacts.force, transpose=True)
    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
    np.testing.assert_allclose(point[valid], original_point[valid], rtol=0, atol=1e-13)
    np.testing.assert_allclose(wrench_force[valid], original_wrench_force[valid], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(peak, original_peak, rtol=1e-4, atol=1e-3)
    conservation = "componentwise"
    if entry.options.intrinsic_load_partition == "body_source_prior" and entry.intrinsic is not None:
        conservation = "body_aggregate"
        active = valid.any(axis=1)
        first = qd_to_numpy(entry.intrinsic.packing.indices, transpose=True)[active, 0]
        scale = np.linalg.norm(np.where(valid[:, :, None], wrench_force, 0).sum(axis=1), axis=1)[active]
        source_scale = qd_to_numpy(entry.intrinsic.tractions.scale, transpose=True)[active, 0]
        scale = np.where(scale > 0, scale, source_scale)
        loaded = scale > 0
        radius = qd_to_numpy(entry.intrinsic.contacts.radius, transpose=True)[active, 0][loaded]
        force_error = qd_to_numpy(entry.contacts.force_error, transpose=True)[active, first][loaded] / scale[loaded]
        moment_error = qd_to_numpy(entry.contacts.moment_error, transpose=True)[active, first][loaded] / (
            radius * scale[loaded]
        )
    else:
        loaded = valid & (np.linalg.norm(wrench_force, axis=2) > 0)
        scale = np.linalg.norm(wrench_force[loaded], axis=1)
        radius = qd_to_numpy(entry.contacts.radius, transpose=True)[loaded]
        force_error = qd_to_numpy(entry.contacts.force_error, transpose=True)[loaded] / scale
        moment_error = qd_to_numpy(entry.contacts.moment_error, transpose=True)[loaded] / (radius * scale)
    assert (force_error <= 1e-8).all() and (moment_error <= 1e-8).all()
    residual = np.sqrt(qd_to_numpy(entry.state.residual_norm_squared))
    rhs_norm = np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared))
    assert (residual <= np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * rhs_norm)).all()
    return {
        "wrench_conservation": conservation,
        "nodal_force_max_difference_N": float(np.max(np.abs(force - original_force))),
        "peak_max_difference_Pa": float(np.max(np.abs(peak - original_peak))),
        "original_point_max_difference_m": float(np.max(np.abs(point[valid] - original_point[valid]), initial=0)),
        "normalized_force_error_max": float(force_error.max(initial=0)),
        "normalized_moment_error_max": float(moment_error.max(initial=0)),
        "complete_residual_max_N": float(residual.max()),
    }


def profile(workload: FrankaEgg, repetitions: int) -> dict:
    snapshot_check = prepare_recovery_snapshot(workload)
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
        "nodal_scatter": lambda: (
            kernel_scatter_faces(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                entry.scatter,
                options.cached_face_bounds,
            )
            if entry.scatter.tasks.shape[0]
            else kernel_scatter(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                options.cooperative_scatter,
                options.cached_face_bounds,
            )
        ),
        "inertia_relief_centrifugal": lambda: (
            kernel_balance_contacts(entry.omega, entry.state, entry.model.info, entry.contacts, entry.scatter)
            if entry.reuse_contact_wrench and entry.scatter.tasks.shape[0] > 0 and options.cooperative_balance
            else kernel_balance(entry.omega, entry.state, entry.model.info, options.cooperative_balance)
        ),
        "rhs_reduction": lambda: kernel_direct_init(entry.state),
        "linear_solve": lambda: entry.model.solve(options, entry.state, entry.omega, surface_load=True),
        "complete_residual": lambda: kernel_full_residual(
            options.young,
            options.tolerance,
            options.absolute_tolerance,
            entry.state,
            entry.model.info,
            False,
            entry.coalesced_residual,
        ),
        "global_peak": lambda: kernel_peak(
            options.young, options.poisson, entry.state, entry.model.info, options.cached_peak
        ),
        "accept": lambda: kernel_accept(entry.state, entry.contacts, solver._errno),
    }
    if entry.reuse_contact_moments:
        # These stages share current-footprint scratch and must refresh together.
        stages = {
            name: stage
            for name, stage in stages.items()
            if name not in ("anchor_search", "candidate_weights_constraints", "nodal_scatter")
        }
        stages = dict(
            list(stages.items())[:2]
            + [
                (
                    "anchor_search_candidate_weights_constraints_nodal_scatter",
                    lambda: (
                        kernel_anchor(recovery.source_epsilon, entry.contacts, entry.surface.info),
                        kernel_fit_scatter_moments(
                            entry.contacts,
                            entry.state,
                            entry.model.info,
                            entry.surface.info,
                            entry.scatter,
                            options.cached_face_bounds,
                        ),
                    ),
                )
            ]
            + list(stages.items())[2:]
        )
    if entry.intrinsic is not None:
        intrinsic_stages = {
            "source_capture": lambda: kernel_capture_rigid_sources(
                entry.link.idx, entry.contacts, entry.intrinsic.raw_sources, solver.collider.collider_state
            ),
            "intrinsic_contact_pipeline": lambda: entry.intrinsic.scatter(
                entry.contacts,
                entry.state,
                entry.model.info,
                entry.surface.info,
                options,
                recovery.source_epsilon,
                solver._errno,
            ),
        }
        stages = dict(
            list(stages.items())[:2]
            + list(intrinsic_stages.items())
            + [
                (name, stage)
                for name, stage in list(stages.items())[2:]
                if name not in ("anchor_search", "candidate_weights_constraints", "nodal_scatter")
            ]
        )
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
    if entry.intrinsic is None:
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
    if entry.intrinsic is not None:
        subdivisions = {}
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
        "recovery_snapshot_check": snapshot_check,
        "stage_wall_ms": times,
        "contacts_per_environment": contacts.sum(axis=1).tolist(),
        "candidate_count": qd_to_numpy(entry.contacts.candidate_count, transpose=True)[contacts].tolist(),
        "pressure_evaluations": qd_to_numpy(entry.contacts.evaluations, transpose=True)[contacts].tolist(),
        "pressure_subdivision_wall_ms": micro_times,
        "grid_samples_visited": qd_to_numpy(visited, transpose=True)[contacts].tolist()
        if entry.intrinsic is None
        else [],
        "grid_samples_inside": qd_to_numpy(inside, transpose=True)[contacts].tolist()
        if entry.intrinsic is None
        else [],
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
    parser.add_argument("--contact-model", choices=("pressure", "intrinsic"), default="pressure")
    parser.add_argument("--intrinsic-support", choices=("contact", "pair_union"), default="contact")
    parser.add_argument("--dense-surface-fan", action="store_true", help="Compare the full-face fan launch.")
    parser.add_argument(
        "--intrinsic-load-partition", choices=("componentwise", "body_source_prior"), default="componentwise"
    )
    parser.add_argument("--contact-block-size", type=int, default=16)
    parser.add_argument("--egg-geometry", choices=("urdf", "mesh64"), default="urdf")
    parser.add_argument("--collision-method", choices=("auto", "mpr", "gjk"), default="auto")
    parser.add_argument("--contact-patch", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--source-revision", default=os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"))
    parser.add_argument("--conditions", type=Path, help="Repeat a frozen condition bank, including controller targets.")
    parser.add_argument("--save-conditions", type=Path, help="Save the constructed condition bank before warmup.")
    parser.add_argument("--serial-solve", action="store_true")
    parser.add_argument("--serial-pressure", action="store_true")
    parser.add_argument("--serial-tractions", action="store_true")
    parser.add_argument("--cold-tractions", action="store_true", help="Start every intrinsic traction fit from zero.")
    parser.add_argument("--serial-scatter", action="store_true")
    parser.add_argument("--serial-balance", action="store_true", help="Compare original rigid inertia relief.")
    parser.add_argument("--contact-wrench-reuse", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--packed-block-size", choices=(0, 256, 512), type=int, default=None)
    parser.add_argument("--coalesced-residual", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--contact-moment-reuse", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(
        "--contact-warp-scatter", action="store_true", help="Compare the original contact-warp scatter."
    )
    parser.add_argument(
        "--scatter-tasks-per-env",
        type=int,
        default=32,
        help="Bound face-task workspace; overflow fully recomputes native scatter.",
    )
    parser.add_argument("--uncached-peak", action="store_true")
    parser.add_argument("--uncached-face-bounds", action="store_true")
    parser.add_argument("--method", choices=("auto", "direct", "inverse"), default="auto")
    parser.add_argument("--inverse-precision", choices=("64", "32"), default="64")
    parser.add_argument("--full-inverse", action="store_true", help="Compare the full nodal-load inverse application.")
    parser.add_argument(
        "--dense-surface-loads", action="store_true", help="Apply all boundary columns without packing."
    )
    parser.add_argument("--history", type=int, choices=(0, 4), default=0)
    parser.add_argument("--unfused-pipeline", action="store_true", help="Compare separate native recovery passes.")
    parser.add_argument("--policy-precision", choices=("64", "32"), default="32")
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--legacy-reset", action="store_true", help="Compare reset followed by separate pose setters.")
    parser.add_argument(
        "--torch-controller", action="store_true", help="Compare the original scripted input scheduling."
    )
    parser.add_argument("--host-reset-indices", action="store_true", help="Compare uncached reset index transfers.")
    parser.add_argument("--rigid-post-fusion", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--trace", action="store_true", help="Export a separate intrusive CUDA/CPU trace pass.")
    parser.add_argument(
        "--trajectory-metrics", action="store_true", help="Time device-side grasp outcomes and per-environment work."
    )
    parser.add_argument(
        "--compact-log", action="store_true", help="Keep full per-environment diagnostics in JSON only."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trajectory_metrics and args.scope in ("profile", "recovery"):
        parser.error("Trajectory metrics require rigid, live, policy or policy-rigid scope.")
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
        surface_inverse=not args.full_inverse,
        packed_surface_loads=not args.dense_surface_loads,
        fused_pipeline=not args.unfused_pipeline,
        output_mode=args.output_mode,
        saved_reset=not args.legacy_reset,
        cooperative_scatter=not args.serial_scatter,
        cooperative_balance=not args.serial_balance,
        contact_wrench_reuse={"auto": None, "on": True, "off": False}[args.contact_wrench_reuse],
        packed_block_size=args.packed_block_size,
        coalesced_residual={"auto": None, "on": True, "off": False}[args.coalesced_residual],
        contact_moment_reuse={"auto": None, "on": True, "off": False}[args.contact_moment_reuse],
        native_controller=not args.torch_controller,
        resident_reset_indices=not args.host_reset_indices,
        fused_rigid_post={"auto": None, "on": True, "off": False}[args.rigid_post_fusion],
        face_parallel_scatter=not args.contact_warp_scatter,
        scatter_tasks_per_env=args.scatter_tasks_per_env,
        cached_peak=not args.uncached_peak,
        cached_face_bounds=not args.uncached_face_bounds,
        contact_model=args.contact_model,
        contact_block_size=args.contact_block_size,
        egg_geometry=args.egg_geometry,
        use_gjk_collision={"auto": None, "mpr": False, "gjk": True}[args.collision_method],
        enable_contact_patch={"auto": None, "on": True, "off": False}[args.contact_patch],
        intrinsic_support=args.intrinsic_support,
        surface_fan_frontier=not args.dense_surface_fan,
        intrinsic_load_partition=args.intrinsic_load_partition,
        cooperative_tractions=not args.serial_tractions,
        warm_start_tractions=not args.cold_tractions,
    )
    solver = workload.scene.rigid_solver
    metrics = RolloutMetrics(workload) if args.trajectory_metrics else None
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
    policy_dtype = torch.float32 if args.policy_precision == "32" else torch.float64
    policy_reference = copy.deepcopy(policy) if args.policy_precision == "32" else None
    policy = policy.to(dtype=policy_dtype)
    policy_error_rad = 0.0
    with torch.inference_mode():
        quality = []
        for tick in range(args.warmup):
            action = None
            if args.scope.startswith("policy"):
                observation = workload.observation()
                action = policy(observation.to(dtype=policy_dtype))
                if policy_reference is not None and tick % 100 == 0:
                    expected = policy_reference(observation)
                    difference = (action.to(dtype=torch.float64) - expected).abs().max().item()
                    policy_error_rad = max(policy_error_rad, 1e-4 * difference)
                    assert policy_error_rad < 1e-9
            workload.step(action)
            if metrics is not None:
                metrics.record()
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
            "load_model": "intrinsic_surface_fan_q10_squared_coordinates_ports8_adjacent_unfolding_verified_single_face_disks_failed_disk_nine_panel_refinement_failed_wrench_fan_partition"
            + ("_body_source_prior" if args.intrinsic_load_partition == "body_source_prior" else "")
            if args.contact_model == "intrinsic"
            else "finite_pad_adaptive_q10",
            "contact_model": args.contact_model,
            "contact_block_size": args.contact_block_size,
            "intrinsic_support": args.intrinsic_support,
            "surface_fan_frontier": not args.dense_surface_fan,
            "intrinsic_load_partition": args.intrinsic_load_partition,
            "cooperative_tractions": not args.serial_tractions,
            "warm_start_tractions": not args.cold_tractions,
            "effective_contact_block_size": solver.stress_recovery.links[0].intrinsic.contacts.valid.shape[0]
            if args.contact_model == "intrinsic" and workload.stress
            else args.contact_block_size,
            "egg_geometry": args.egg_geometry,
            "collision_method": args.collision_method,
            "contact_patch": args.contact_patch,
            "condition_count": workload.condition_count,
            "conditions": str(args.conditions) if args.conditions is not None else None,
            "conditions_sha256": hashlib.sha256(args.conditions.read_bytes()).hexdigest()
            if args.conditions is not None
            else None,
            "cooperative_solve": not args.serial_solve,
            "cooperative_pressure": not args.serial_pressure,
            "cooperative_scatter": not args.serial_scatter,
            "cooperative_balance": not args.serial_balance,
            "contact_wrench_reuse": args.contact_wrench_reuse,
            "packed_block_size": args.packed_block_size,
            "coalesced_residual": args.coalesced_residual,
            "contact_moment_reuse": args.contact_moment_reuse,
            "effective_contact_moment_reuse": [entry.reuse_contact_moments for entry in solver.stress_recovery.links]
            if solver.stress_recovery is not None
            else [],
            "effective_contact_wrench_reuse": [
                entry.fused_pipeline
                and entry.reuse_contact_wrench
                and entry.scatter.tasks.shape[0] > 0
                and entry.options.cooperative_balance
                for entry in solver.stress_recovery.links
            ]
            if solver.stress_recovery is not None
            else [],
            "effective_packed_block_sizes": [entry.packed_block_size for entry in solver.stress_recovery.links]
            if solver.stress_recovery is not None
            else [],
            "face_parallel_scatter": not args.contact_warp_scatter,
            "scatter_tasks_per_env": args.scatter_tasks_per_env,
            "cached_peak": not args.uncached_peak,
            "cached_face_bounds": not args.uncached_face_bounds,
            "method": args.method,
            "inverse_precision": args.inverse_precision,
            "surface_inverse": not args.full_inverse,
            "packed_surface_loads": not args.dense_surface_loads,
            "fused_pipeline": not args.unfused_pipeline,
            "output_mode": args.output_mode,
            "reset_mode": "individual_setters" if args.legacy_reset else "saved_initial_state",
            "native_controller": workload.controller_inputs is not None,
            "rigid_post_fusion": args.rigid_post_fusion,
            "effective_rigid_post_fusion": workload.stress
            and workload.scene.rigid_solver.stress_recovery.post_kernel is not None,
            "reset_index_cache": workload.reset_indices.describe(),
            "controller_buffers": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
                for name, value in (workload.controller_inputs or {}).items()
            },
            "history": args.history,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_revision": args.source_revision,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "command": [sys.executable, *sys.argv],
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(
                    [
                        *Path("genesis/engine/solvers/rigid/stress").glob("*.py"),
                        Path("genesis/engine/solvers/rigid/rigid_solver.py"),
                        Path("genesis/options/rigid_stress.py"),
                        Path("genesis/options/morphs.py"),
                        Path("genesis/engine/entities/rigid_entity/description.py"),
                        Path("examples/rigid/franka_egg_stress.py"),
                        Path("examples/rigid/egg_stress_controller.py"),
                        Path(__file__),
                    ]
                )
            },
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": "GPU-" + str(torch.cuda.get_device_properties(gs.device).uuid),
            "gpu_total_bytes": torch.cuda.get_device_properties(gs.device).total_memory,
            "quadrants": qd.__version__,
            "torch": torch.__version__,
            "setup_seconds": setup_seconds,
            "warmup_steps": args.warmup,
            "warmup_quality": quality,
            "trajectory_metrics": {"warmup": metrics.summary(), "repeats": []} if metrics is not None else None,
            "policy": {
                "layers": [26, 128, 128, 7],
                "seed": 99173,
                "dtype": "float32" if args.policy_precision == "32" else "float64",
                "residual_scale_rad": 1e-4,
                "scope": "inference rollout",
                "control_error_budget_rad": 1e-9,
                "warmup_reference_max_error_rad": policy_error_rad,
            },
        }
        policy_reference = None
        if args.scope == "profile":
            result.update(profile(workload, args.steps))
        else:
            if args.scope == "recovery":
                prepare_start = time.perf_counter()
                # Live warmup may only compile the combined rigid-post graph.
                # Compile/capture this standalone path before timing its repeats.
                result["recovery_snapshot_check"] = prepare_recovery_snapshot(workload)
                result["recovery_preparation_steps"] = 1
                result["recovery_preparation_seconds_including_JIT"] = time.perf_counter() - prepare_start
                result["recovery_input"] = "Frozen actual contact snapshot with its matching pre-integration frame."
            rows = []
            for _ in range(args.repetitions):
                if args.scope != "recovery":
                    workload.restart()
                if metrics is not None:
                    metrics.reset()
                failure_before = None
                scatter_overflow_before = 0
                if workload.stress:
                    scatter_overflow_before = int(
                        qd_to_numpy(workload.scene.rigid_solver.stress_recovery.links[0].scatter.overflow_calls)
                    )
                    failure_before = qd_to_numpy(
                        workload.scene.rigid_solver.stress_recovery.links[0].state.invalid_steps
                    )
                qd.sync()
                start = time.perf_counter()
                steps = 0
                while steps < args.steps or time.perf_counter() - start < args.minimum_seconds:
                    if args.scope == "recovery":
                        workload.scene.rigid_solver.stress_recovery.recover(0)
                    else:
                        action = (
                            policy(workload.observation().to(dtype=policy_dtype))
                            if args.scope.startswith("policy")
                            else None
                        )
                        workload.step(action)
                        if metrics is not None:
                            metrics.record()
                    steps += 1
                qd.sync()
                elapsed = time.perf_counter() - start
                if metrics is not None:
                    result["trajectory_metrics"]["repeats"].append(metrics.summary())
                invalid_environment_steps = 0
                failed_per_environment = None
                if failure_before is not None:
                    failed_per_environment = (
                        qd_to_numpy(workload.scene.rigid_solver.stress_recovery.links[0].state.invalid_steps)
                        - failure_before
                    )
                    invalid_environment_steps = int(failed_per_environment.sum())
                workload.scene.rigid_solver.check_errno()
                rows.append(
                    {
                        "steps": steps,
                        "seconds": elapsed,
                        "transitions_per_second": (args.envs * steps - invalid_environment_steps) / elapsed,
                        "attempted_transitions_per_second": args.envs * steps / elapsed,
                        "invalid_environment_steps": invalid_environment_steps,
                        "scatter_overflow_calls": int(
                            qd_to_numpy(workload.scene.rigid_solver.stress_recovery.links[0].scatter.overflow_calls)
                        )
                        - scatter_overflow_before
                        if workload.stress
                        else 0,
                        "failed_steps_per_environment": failed_per_environment.tolist()
                        if failed_per_environment is not None
                        else None,
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
            result["surface_inverse_selected"] = entry.model.surface_inverse is not None
            result["build_timings_including_JIT"] = asdict(entry.model.build_timings)
            result["maximum_stress_Pa"] = tensor_to_array(workload.link.get_max_stress()).tolist()
            buffers = []
            configurations = []
            allocations = {}
            for item in workload.scene.rigid_solver.stress_recovery.data:
                if item.kind == DataKind.CONFIG:
                    configurations.append(item.value.model_dump(mode="json"))
                else:
                    if math.prod(item.value.shape) == 0:
                        # Disabled native workspaces have no device allocation;
                        # their null pointer cannot be exported through DLPack.
                        element_shape = (
                            item.value.element_shape
                            if isinstance(item.value, (qd.VectorTensor, qd.MatrixTensor))
                            else ()
                        )
                        buffers.append(
                            {
                                "name": item.name,
                                "shape": [*item.value.shape, *element_shape],
                                "dtype": str(item.value.dtype),
                                "bytes": 0,
                                "kind": int(item.kind),
                            }
                        )
                        continue
                    try:
                        view = qd_to_torch(item.value, copy=False)
                    except (ValueError, RuntimeError) as error:
                        raise RuntimeError(f"Cannot inspect native buffer {item.name}: {error}") from error
                    pointer = view.data_ptr()
                    size = view.numel() * view.element_size()
                    alias_of = allocations[pointer][0] if pointer in allocations else None
                    if alias_of is None:
                        allocations[pointer] = (item.name, size)
                    buffers.append(
                        {
                            "name": item.name,
                            "shape": list(view.shape),
                            "dtype": str(view.dtype),
                            "bytes": size,
                            "alias_of": alias_of,
                            "kind": int(item.kind),
                        }
                    )
            result["stress_configurations"] = configurations
            result["native_buffers"] = buffers
            result["native_logical_buffer_bytes"] = sum(item["bytes"] for item in buffers)
            result["native_buffer_bytes"] = sum(size for _, size in allocations.values())
            result["native_allocation_count"] = len(allocations)
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["saved_reset_state_bytes"] = (
            sum(
                value.numel() * value.element_size()
                for state in workload.reset_state.solvers_state
                if state is not None
                for value in vars(state).values()
                if isinstance(value, torch.Tensor)
            )
            if workload.reset_state is not None
            else 0
        )
        result["device_free_bytes"] = torch.cuda.mem_get_info()[0]
        result["reset_index_cache_at_end"] = workload.reset_indices.describe()
        if args.trace:
            qd.sync()
            with torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
            ) as profiler:
                for _ in range(50):
                    with torch.profiler.record_function("rigid_stress_trace_step"):
                        action = (
                            policy(workload.observation().to(dtype=policy_dtype))
                            if args.scope.startswith("policy")
                            else None
                        )
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
