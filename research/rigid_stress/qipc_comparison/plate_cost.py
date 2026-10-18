"""Warmed sparse-observation cost at a measured, continuously simulated plate load."""

from time import perf_counter
import json

import numpy as np
import torch

from genesis.utils.misc import tensor_to_array

from .artifacts import write_json


def measure(args, scene, plate, egg, link, entry):
    assert args.cost_steps > 0 and args.cost_stride > 0 and args.cost_repeats > 0
    assert not args.empty and args.loads[-1] > 0
    assert not args.shear_distance, "This warmed cost contract measures the normal held-load baseline"
    samples = []
    for repeat in range(-1, args.cost_repeats):
        steps = max(args.cost_stride * 2, 100) if repeat < 0 else args.cost_steps
        observed, transferred = 0, 0
        torch.cuda.synchronize()
        begun = perf_counter()
        for step in range(1, steps + 1):
            plate.control_dofs_force(args.loads[-1])
            observe = step % args.cost_stride == 0 or step == steps
            if observe and entry is not None:
                link.request_stress_update()
            scene.step()
            scene.rigid_solver.check_errno()
            if observe:
                for state in (
                    plate.get_dofs_position(),
                    plate.get_dofs_velocity(),
                    egg.get_pos(),
                    egg.get_vel(),
                    egg.get_ang(),
                ):
                    assert torch.isfinite(state).all()
                if entry is not None:
                    field, vm = link.get_stress_field(copy=False)
                    field, vm = tensor_to_array(field), tensor_to_array(vm)
                    assert field.shape[0] == vm.shape[0] == args.envs
                    assert np.isfinite(field).all() and np.isfinite(vm).all()
                    transferred += field.nbytes + vm.nbytes
                    observed += 1
        torch.cuda.synchronize()
        elapsed = perf_counter() - begun
        # All-environment contact and velocity admission is outside the timed window.
        contacts = {key: tensor_to_array(value) for key, value in egg.get_contacts().items()}
        valid = contacts["valid_mask"]
        egg_a = contacts["link_a"] == link.idx
        forces = np.where(egg_a[..., None], contacts["force_a"], contacts["force_b"])
        upper_force = -(np.minimum(forces[..., 2], 0) * valid).sum(axis=1)
        lower_force = (np.maximum(forces[..., 2], 0) * valid).sum(axis=1)
        speed = tensor_to_array(plate.get_dofs_velocity())
        load_error = np.maximum(np.abs(upper_force - args.loads[-1]), np.abs(lower_force - args.loads[-1]))
        passed = bool(np.all(load_error <= 0.01 * args.loads[-1]) and np.max(np.abs(speed)) <= 1e-6)
        samples.append(
            {
                "repeat": repeat,
                "warmup": repeat < 0,
                "steps": steps,
                "elapsed_seconds": elapsed,
                "latency_ms_per_step": elapsed / steps * 1000,
                "aggregate_env_steps_per_second": args.envs * steps / elapsed,
                "full_field_observations": observed,
                "full_field_host_bytes": transferred,
                "all_environment_load_and_velocity_passed": passed,
                "maximum_load_error_n": float(load_error.max()),
                "maximum_plate_speed_m_per_s": float(np.max(np.abs(speed))),
            }
        )
        write_json(
            args.output / "cost.json",
            {
                "envs": args.envs,
                "method": args.method,
                "samples": samples,
                "scope": "Warmed continuous mechanics at the last held load. Sparse complete stress/VM host transfers, finite checks and actual engine admission included; frame files and post-window contact reductions excluded. This cost does not itself establish physical accuracy. Warmup row excluded from primary rate.",
            },
        )
        assert passed, "Do not admit no-contact or mechanically unsettled speedups"
    if args.profile_steps:
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
        ) as profiler:
            for step in range(1, args.profile_steps + 1):
                with torch.profiler.record_function("plate_force_control"):
                    plate.control_dofs_force(args.loads[-1])
                observe = step % args.cost_stride == 0 or step == args.profile_steps
                if observe and entry is not None:
                    link.request_stress_update()
                with torch.profiler.record_function("plate_mechanics_and_requested_reconstruction"):
                    scene.step()
                    scene.rigid_solver.check_errno()
                if observe and entry is not None:
                    with torch.profiler.record_function("complete_field_observation_to_host"):
                        field, vm = link.get_stress_field(copy=False)
                        field, vm = tensor_to_array(field), tensor_to_array(vm)
                        assert np.isfinite(field).all() and np.isfinite(vm).all()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(args.output / "trace.json"))
        events = json.loads((args.output / "trace.json").read_text())["traceEvents"]
        kernels = {}
        for event in events:
            if event.get("cat") == "kernel" and event.get("ph") == "X":
                name = event["name"]
                row = kernels.setdefault(name, {"count": 0, "total_us": 0.0})
                row["count"] += 1
                row["total_us"] += event["dur"]
        write_json(
            args.output / "profile.json",
            {
                "steps": args.profile_steps,
                "envs": args.envs,
                "kernel_totals": kernels,
                "scope": "Separate intrusive warmed torch CPU/CUDA trace. Kernel duration sums are not wall-clock throughput and may overlap across streams; complete unclassified kernel names are retained.",
            },
        )
