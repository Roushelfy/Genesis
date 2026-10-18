"""Measure native manual stress sampling over a continuous contact history after checked in-process warmup/reset."""

import argparse
import gzip
import json
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array

from ..shell_comparison.rollout import controller_step
from .artifacts import file_id, runtime_identity, write_json
from .prescribed_motion import PrescribedMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--method", choices=("dpr", "scr"), required=True)
    parser.add_argument("--sampling", choices=("every_step", "manual"), required=True)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--duration", type=float, default=1.05)
    parser.add_argument("--window-start", type=float, default=0.85)
    parser.add_argument("--keyframes", type=float, nargs="+", default=[0.89, 0.95, 1.05])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "SLURM_JOB_ID" in os.environ
    assert 0 < args.window_start < args.duration <= 1.5
    dt = 0.001
    steps, start = round(args.duration / dt), round(args.window_start / dt)
    keys = {round(time / dt) for time in args.keyframes}
    assert all(abs(round(time / dt) * dt - time) < 1e-12 and 0 < time <= args.duration for time in args.keyframes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    begun = perf_counter()
    workload = FrankaEgg(
        1,
        conditions=args.conditions,
        output_mode="full",
        stress_update_mode=args.sampling,
        native_controller=False,
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        contact_model="pressure" if args.method == "dpr" else "intrinsic",
        intrinsic_load_partition="body_source_prior",
        timestep=dt,
    )
    motion = PrescribedMotion.configure(workload.robot, args.motion, dt, 1e10) if args.motion is not None else None
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - begun
    with np.load(workload.link.stress_options.mesh) as saved:
        np.savez_compressed(args.output / "rest.npz", **{name: saved[name] for name in saved.files})
    initial = (
        workload.robot.get_qpos().clone(),
        workload.robot.get_dofs_velocity().clone(),
        workload.egg.get_pos().clone(),
        workload.egg.get_quat().clone(),
        workload.egg.get_vel().clone(),
        workload.egg.get_ang().clone(),
    )
    payloads, checks = [], []
    windows, rollouts, cold_warmup_seconds = [], [], None
    completed = False
    try:
        for repeat in range(2):
            if repeat:
                workload.restart()
                restored = (
                    workload.robot.get_qpos(),
                    workload.robot.get_dofs_velocity(),
                    workload.egg.get_pos(),
                    workload.egg.get_quat(),
                    workload.egg.get_vel(),
                    workload.egg.get_ang(),
                )
                for expected, actual in zip(initial, restored):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            window_begun = None
            profiler = None
            torch.cuda.synchronize()
            rollout_begun = perf_counter()
            with ExitStack() as stack:
                for step in range(1, steps + 1):
                    if step == start + 1:
                        if repeat and args.profile:
                            profiler = stack.enter_context(
                                torch.profiler.profile(
                                    activities=(
                                        torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA,
                                    )
                                )
                            )
                        torch.cuda.synchronize()
                        window_begun = perf_counter()
                    observe = step in keys
                    if motion is not None:
                        motion.advance(workload, step - 1, True, update_stress=observe)
                    else:
                        if (step - 1) % 10 == 0:
                            controller_step(workload, False, advance=False)
                        if observe:
                            workload.link.request_stress_update()
                        workload.scene.step()
                    workload.scene.rigid_solver.check_errno()
                    finite = bool(
                        torch.isfinite(workload.robot.get_qpos()).all()
                        & torch.isfinite(workload.robot.get_dofs_velocity()).all()
                        & torch.isfinite(workload.egg.get_pos()).all()
                        & torch.isfinite(workload.egg.get_vel()).all()
                    )
                    assert finite, "Native mechanical state became nonfinite"
                    if repeat:
                        checks.append({"time_s": step * dt, "mechanical_state_finite": finite})
                    if repeat and observe:
                        tensor, vm = workload.link.get_stress_field()
                        assert bool(torch.isfinite(tensor).all() & torch.isfinite(vm).all())
                        payloads.append(
                            {
                                "time_s": step * dt,
                                "step": step - 1,
                                "stress_voigt_link_pa": tensor_to_array(tensor)[0],
                                "vm_pa": tensor_to_array(vm)[0],
                                "q": tensor_to_array(workload.robot.get_qpos())[0],
                                "link_position_m": tensor_to_array(workload.egg.get_pos())[0],
                                "link_quaternion_wxyz": tensor_to_array(workload.egg.get_quat())[0],
                            }
                        )
                torch.cuda.synchronize()
                ended = perf_counter()
            rollouts.append(ended - rollout_begun)
            windows.append(ended - window_begun)
            if not repeat:
                cold_warmup_seconds = ended - rollout_begun
            elif args.profile:
                trace = Path(os.environ["TMPDIR"]) / "native-sampling-trace.json"
                profiler.export_chrome_trace(str(trace))
                data = json.loads(trace.read_text())
                kernels = defaultdict(lambda: [0, 0.0])
                for event in data["traceEvents"]:
                    if event.get("cat") == "kernel":
                        kernels[event["name"]][0] += 1
                        kernels[event["name"]][1] += event["dur"]
                write_json(
                    args.output / "kernels.json",
                    [
                        {"name": name, "calls": count, "total_duration_us": duration}
                        for name, (count, duration) in sorted(kernels.items())
                    ],
                )
                with trace.open("rb") as source, gzip.open(args.output / "trace.json.gz", "wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            completed = len(payloads) == len(keys)
    finally:
        begun = perf_counter()
        for payload in payloads:
            np.savez_compressed(args.output / "frames" / f"step{payload['step']:06d}.npz", **payload)
        export_seconds = perf_counter() - begun
        write_json(args.output / "steps.json", checks)
        write_json(
            args.output / "result.json",
            {
                "kind": "native-continuous-contact-window-sampling-control",
                "identity": runtime_identity(),
                "genesis_import": gs.__file__,
                "method": args.method,
                "sampling": args.sampling,
                "conditions": file_id(args.conditions),
                "environment_bank_indices": [0],
                "motion": motion.identity() if motion is not None else None,
                "effective_options": workload.link.stress_options.model_dump(mode="json"),
                "setup_seconds": setup_seconds,
                "cold_original_state_warmup_seconds": cold_warmup_seconds,
                "initial_mechanical_reset_exact": len(rollouts) == 2,
                "window_s": [args.window_start, args.duration],
                "window_seconds": windows,
                "rollout_seconds": rollouts,
                "deferred_compression_IO_seconds": export_seconds,
                "full_spatial_keyframes_s": sorted(args.keyframes),
                "payloads": len(payloads),
                "completed_requested_history": completed,
                "profiled": args.profile,
                "reference_accuracy_qualified": False,
                "scope": "First continuous original-state pass warms physics/recovery and is reported separately. Public in-process restart is checked exactly in joint/link positions and velocities before the second complete original-state history. GPU-fenced window includes controller, mechanics, per-step native/finite checks and selected full-field host transfer. Compression/IO is deferred. Optional contact-window CUPTI trace is intrusive, not primary throughput. Manual observes only declared scene steps, with one physical substep per scene step; held snapshots are never exported as new samples.",
            },
        )
    if not completed:
        raise RuntimeError("Native sampling window did not complete")


if __name__ == "__main__":
    main()
