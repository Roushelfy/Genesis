"""Common controller and measurement scope; imports no external shell or native stress solver."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .common import synchronize
from .peak_capture import PeakCapture


def controller_step(workload, shell_model):
    ids = np.flatnonzero((workload.tick > workload.delays) & ((workload.tick - workload.delays) % 600 == 0))
    if len(ids):
        workload.reset(ids)
    phase = torch.remainder(workload.tick - workload.device_delays, 600) / 599
    phase = torch.where(workload.tick >= workload.device_delays, phase, 0.0)
    approach = torch.clamp(phase / 0.15, 0.0, 1.0)[:, None]
    approach = approach * approach * (3.0 - 2.0 * approach)
    lift = torch.clamp((phase - 0.3) / 0.2, 0.0, 1.0)[:, None]
    lift = lift * lift * (3.0 - 2.0 * lift)
    slide = torch.clamp((phase - 0.6) / 0.15, 0.0, 1.0)[:, None]
    slide = slide * slide * (3.0 - 2.0 * slide)
    target = workload.initial_joints * (1.0 - approach) + workload.pick_joints * approach
    target += (workload.lift_joints - workload.pick_joints) * lift + (
        workload.slide_joints - workload.lift_joints
    ) * slide
    workload.robot.control_dofs_position(target[:, workload.arm], workload.arm)
    grip = torch.where((phase < 0.15) | (phase > 0.9), 0.04, 0.018)
    limit = torch.where((phase >= 0.7) & (phase < 0.76), 0.055, 4.0)[:, None].expand(-1, 2)
    workload.robot.set_dofs_force_range(-limit, limit, workload.fingers)
    workload.robot.control_dofs_position(grip[:, None].expand(-1, 2), workload.fingers)
    if not shell_model and workload.stress:
        radius = workload.base_radius * (1.0 + 0.15 * torch.sin(2.0 * torch.pi * phase))
        workload.link.set_stress_contact_radius(radius[:, None])
    workload.scene.step()
    if shell_model:
        workload.observation.update(workload.scene.sim.shell_solver)
    workload.tick += 1
    return phase


def run(args, workload, shell_model, diagnostic_buffers=None):
    start = perf_counter()
    for _ in range(args.warmup):
        controller_step(workload, shell_model)
    synchronize()
    warm_seconds = perf_counter() - start
    warmup_invalid = invalid_counter(workload, shell_model)
    workload.scene.rigid_solver.check_errno()
    repeats = []
    traces = []
    for repeat in range(args.repetitions):
        workload.tick = 0
        workload.reset()
        workload.reset_count = 0
        invalid_before = invalid_counter(workload, shell_model)
        heights, peaks, phases, residuals, counts, caps, fields, contacts = [], [], [], [], [], [], [], []
        peak_capture = PeakCapture.allocate(args, workload, shell_model) if args.peak_snapshots else None
        synchronize()
        start = perf_counter()
        nonfinite_step = None
        for step in range(args.steps):
            if args.instrumented:
                diagnostic_buffers.clear()
            phase = controller_step(workload, shell_model)
            if args.record_every and step % args.record_every == 0:
                phases.append(tensor_to_array(phase).copy())
                if shell_model:
                    vertices = tensor_to_array(workload.egg.get_verts_pos())[:, : workload.egg.n_verts]
                    heights.append(vertices.mean(axis=1))
                    peaks.append(tensor_to_array(workload.observation.peaks).copy())
                else:
                    heights.append(tensor_to_array(workload.egg.get_pos()).copy())
                    peaks.append(
                        tensor_to_array(workload.link.get_stress_field(copy=False)[1].amax(dim=(-2, -1))).copy()
                        if workload.stress
                        else np.zeros(args.envs)
                    )
                if args.instrumented:
                    residuals.append(tensor_to_array(diagnostic_buffers.residuals).copy())
                    counts.append(tensor_to_array(diagnostic_buffers.counts).copy())
                    caps.append(tensor_to_array(diagnostic_buffers.caps).copy())
                if peak_capture is not None:
                    peak_capture.record(workload, step, phases[-1], peaks[-1])
            if shell_model and step in (11, 179, 359, 539, 719, args.steps - 1) and args.record_every:
                fields.append(
                    (
                        step,
                        tensor_to_array(workload.observation.tensors).copy(),
                        tensor_to_array(workload.egg.get_verts_pos()).copy(),
                    )
                )
                if args.instrumented:
                    contacts.append(
                        (
                            step,
                            tensor_to_array(diagnostic_buffers.impulses).copy(),
                            tensor_to_array(diagnostic_buffers.points).copy(),
                        )
                    )
            if (
                not shell_model
                and workload.stress
                and step in (11, 179, 359, 539, 719, args.steps - 1)
                and args.record_every
            ):
                fields.append(
                    (
                        step,
                        tensor_to_array(workload.link.get_stress_field(copy=False)[0]).copy(),
                        tensor_to_array(workload.egg.get_pos()).copy(),
                    )
                )
            if args.record_every and step % args.record_every == 0:
                if not np.isfinite(heights[-1]).all() or not np.isfinite(peaks[-1]).all():
                    nonfinite_step = step
                    break
        synchronize()
        elapsed = perf_counter() - start
        executed_steps = step + 1
        invalid_after = invalid_counter(workload, shell_model)
        failed_per_env = invalid_after - invalid_before if invalid_after is not None else None
        invalid_environment_steps = int(failed_per_env.sum()) if failed_per_env is not None else None
        if shell_model:
            final_fields = tensor_to_array(workload.observation.tensors)
            final_positions = tensor_to_array(workload.egg.get_verts_pos())
        else:
            final_fields = (
                tensor_to_array(workload.link.get_stress_field(copy=False)[0])
                if workload.stress and args.output_mode == "full"
                else tensor_to_array(workload.link.get_max_stress())
                if workload.stress
                else np.empty(0)
            )
            final_positions = tensor_to_array(workload.egg.get_pos())
        final_output_finite = bool(np.isfinite(final_fields).all() and np.isfinite(final_positions).all())
        repeats.append(
            {
                "repeat": repeat,
                "seconds": elapsed,
                "env_steps": executed_steps * args.envs,
                "requested_steps": args.steps,
                "executed_steps": executed_steps,
                "completed_requested_history": executed_steps == args.steps and nonfinite_step is None,
                "nonfinite_step": nonfinite_step,
                "final_output_finite_checked_after_timer": final_output_finite,
                "accuracy_qualified": None,
                "invalid_environment_steps": invalid_environment_steps,
                "invalid_steps_per_environment": failed_per_env.tolist() if failed_per_env is not None else None,
                "valid_aggregate_env_steps_per_second": (executed_steps * args.envs - invalid_environment_steps)
                / elapsed
                if invalid_environment_steps is not None
                else None,
                "aggregate_env_steps_per_second": executed_steps * args.envs / elapsed,
                "batch_step_ms": elapsed / executed_steps * 1000,
                "partial_reset_envs": workload.reset_count,
                "scope": "intrusive-diagnostic"
                if args.record_every or args.instrumented or args.memory
                else "normal-full-field",
            }
        )
        trace = {
            "positions": np.array(heights),
            "peaks_pa": np.array(peaks),
            "phases": np.array(phases),
            "residuals": np.array(residuals),
            "iterations": np.array(counts),
            "caps": np.array(caps),
        }
        for step, stress, vertices in fields:
            trace[f"stress_step{step}"] = stress
            trace[f"vertices_step{step}"] = vertices
        for step, impulses, points in contacts:
            trace[f"impulses_step{step}"] = impulses
            trace[f"contact_points_step{step}"] = points
        if peak_capture is not None:
            trace.update(peak_capture.payload())
        np.savez_compressed(args.output / f"trajectory-repeat{repeat}.npz", **trace)
        traces.append(
            {
                "all_recorded_fields_finite": all(np.isfinite(value).all() for value in trace.values())
                if args.record_every
                else None,
                "minimum_position_z_m": finite_float(np.array(heights)[..., 2].min()) if heights else None,
                "maximum_position_z_m": finite_float(np.array(heights)[..., 2].max()) if heights else None,
                "maximum_peak_pa": finite_float(np.array(peaks).max()) if peaks else None,
                "cap_exhausted_substeps": int(np.array(caps).sum()) if caps else None,
            }
        )
        print("REPEAT", json.dumps(repeats[-1]), flush=True)
        workload.scene.rigid_solver.check_errno()
    return {
        "warmup_seconds": warm_seconds,
        "warmup_invalid_environment_steps": int(warmup_invalid.sum()) if warmup_invalid is not None else None,
        "repeats": repeats,
        "trajectory_summary": traces,
    }


def invalid_counter(workload, shell_model):
    if not shell_model and workload.stress:
        return qd_to_numpy(
            workload.scene.rigid_solver.stress_recovery.links[0].state.invalid_steps, transpose=True, copy=True
        )
    return None


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def profile_run(args, workload, shell_model):
    for _ in range(args.warmup):
        controller_step(workload, shell_model)
    synchronize()
    started_at_tick = workload.tick
    with torch.profiler.profile(
        activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
    ) as profiler:
        for _ in range(args.steps):
            with torch.profiler.record_function("full_stress_scene_step"):
                controller_step(workload, shell_model)
        synchronize()
    profiler.export_chrome_trace(str(args.output / "trace.json"))
    return {
        "scope": "intrusive-moving-profile",
        "start_tick": started_at_tick,
        "steps": args.steps,
        "warmup": args.warmup,
        "note": "No throughput rate derived from this trace; warmup state retained to include moving loaded contacts.",
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("shell", "rigid", "pressure", "pressure-tuned", "intrinsic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native-assets", type=Path)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--output-mode", choices=("full", "max"), default="full")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--patches", type=int, default=6)
    parser.add_argument("--coarse-interval", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--coupling-friction", type=float, default=0.6)
    parser.add_argument("--wall-volume-input", type=Path)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--record-every", type=int, default=0)
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--peak-snapshots", action="store_true")
    return parser
