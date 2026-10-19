"""Time fixed-horizon native scenes with explicit GPU output observation cadence.

Primary wall time includes the controller, scene, manual request dispatch and
required native acceptance. Complete motion/field audits run separately.
"""

import argparse
from pathlib import Path
from time import perf_counter

import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_torch, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .native_observation_cadence import ObservationCadence


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--method", choices=("rigid", "dpr", "scr"), required=True)
    cli.add_argument("--envs", type=int, required=True)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--output-mode", choices=("max", "full"), default="max")
    cli.add_argument("--stress-stride", type=int, default=1)
    cli.add_argument("--steps", type=int, default=1200)
    cli.add_argument("--warmup", type=int, default=900)
    cli.add_argument("--repetitions", type=int, default=3)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if min(args.envs, args.stress_stride, args.steps, args.repetitions) < 1 or args.warmup < 0:
        cli.error("Counts must be positive, with nonnegative warmup")
    if args.steps % args.stress_stride or args.warmup % args.stress_stride:
        cli.error("Complete windows must end at a requested observation")
    if args.method == "rigid" and (args.stress_stride != 1 or args.output_mode != "max"):
        cli.error("Pure rigid has one baseline, without a stress cadence/output mode")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "args": vars(args) | {"conditions": str(args.conditions), "output": str(args.output)},
            "conditions": file_id(args.conditions),
        },
    )
    gs.init(backend=gs.gpu, precision="64", seed=623016, logging_level="warning")
    started = perf_counter()
    workload = FrankaEgg(
        args.envs,
        level=1,
        stress=args.method != "rigid",
        seed=623016,
        varied=True,
        conditions=args.conditions,
        history_size=0,
        output_mode=args.output_mode,
        stress_update_mode="manual" if args.stress_stride > 1 else "every_step",
        contact_model="intrinsic" if args.method == "scr" else "pressure",
        contact_block_size=16,
        intrinsic_load_partition="body_source_prior",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
    )
    cadence = ObservationCadence(workload, args.stress_stride)
    setup_seconds = perf_counter() - started
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0] if workload.stress else None
    rows = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            workload.step()
        qd.sync()
        solver.check_errno()
        for repeat in range(args.repetitions):
            workload.restart()
            requests_before = cadence.requests
            invalid_before = qd_to_torch(entry.state.invalid_steps, copy=True) if entry is not None else None
            qd.sync()
            started = perf_counter()
            for _ in range(args.steps):
                workload.step()
            qd.sync()
            elapsed = perf_counter() - started
            solver.check_errno()
            requested = args.steps // args.stress_stride if entry is not None else 0
            if entry is not None:
                invalid = qd_to_torch(entry.state.invalid_steps, copy=False) - invalid_before
                assert (invalid == 0).all()
                assert qd_to_torch(entry.state.step_valid, copy=False).all()
                peak = workload.link.get_max_stress(copy=False)
                assert peak.is_cuda and torch.isfinite(peak).all() and (peak >= 0).all()
                if args.output_mode == "full":
                    tensor, vm = workload.link.get_stress_field(copy=False)
                    assert tensor.is_cuda and vm.is_cuda and torch.isfinite(tensor).all() and torch.isfinite(vm).all()
                if args.stress_stride > 1:
                    assert cadence.requests - requests_before == requested
            rows.append(
                {
                    "repeat": repeat,
                    "scene_steps": args.steps,
                    "seconds": elapsed,
                    "env_scene_steps_per_second": args.envs * args.steps / elapsed,
                    "batch_scene_steps_per_second": args.steps / elapsed,
                    "requested_updates_per_env": requested,
                    "aggregate_requested_stress_updates": requested * args.envs,
                    "invalid_environment_updates": 0,
                    "resets": workload.reset_count,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": True,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_uuid": str(torch.cuda.get_device_properties(gs.device).uuid),
            "gpu_total_bytes": torch.cuda.get_device_properties(gs.device).total_memory,
            "method": args.method,
            "envs": args.envs,
            "output_mode": args.output_mode,
            "stress_stride": args.stress_stride,
            "dt_s": 0.01,
            "substeps": 1,
            "warmup": args.warmup,
            "steps": args.steps,
            "setup_seconds_including_JIT": setup_seconds,
            "conditions": file_id(args.conditions),
            "stress_configuration": entry.options.model_dump(mode="json") if entry is not None else None,
            "effective_rigid_post_fusion": entry is not None and solver.stress_recovery.post_kernel is not None,
            "repeats": rows,
            "final_position_m": tensor_to_array(workload.egg.get_pos()).tolist(),
            "physical_reference_qualified": False,
            "policy_transitions_measured": False,
            "scope": "Native evolving scenes, fixed1200-step primary windows with warm in-process repeats. Required acceptance and request dispatch retained. GPU outputs generated only at the declared cadence; no policy, field readback, per-step recording, independent audit or plotting in wall time. Unsampled outputs are not fresh stress observations; every-step reliability and physical error require their separate evidence.",
        },
    )


if __name__ == "__main__":
    main()
