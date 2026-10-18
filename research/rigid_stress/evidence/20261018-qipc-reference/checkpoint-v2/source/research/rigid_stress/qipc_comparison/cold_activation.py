"""Measure fresh-cache activation and first physical steps separately from warmed throughput."""

import os
from time import perf_counter

import torch

import genesis as gs

from ..shell_comparison.common import synchronize
from ..shell_comparison.rollout import controller_step, make_parser
from .artifacts import file_id, runtime_identity, write_json


def main():
    args = make_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert args.output_mode == "full" and not args.memory and not args.instrumented
    started = perf_counter()
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    synchronize()
    initialized = perf_counter()
    shell = args.model == "shell"
    if shell:
        from ..shell_comparison.application import ShellEgg

        workload = ShellEgg(args)
    else:
        from examples.rigid.franka_egg_stress import FrankaEgg

        tuned = args.model == "pressure-tuned"
        workload = FrankaEgg(
            args.envs,
            conditions=args.conditions,
            substeps=args.substeps,
            output_mode="full",
            native_controller=False,
            stress=args.model != "rigid",
            egg_geometry="mesh64",
            use_gjk_collision=False,
            enable_contact_patch=False,
            contact_model="intrinsic" if args.model == "intrinsic" else "pressure",
            intrinsic_load_partition="body_source_prior",
            scatter_tasks_per_env=128 if tuned else 32,
            contact_wrench_reuse=True if tuned else None,
            contact_moment_reuse=True if tuned else None,
            packed_block_size=512 if tuned else None,
            coalesced_residual=True if tuned else None,
        )
    synchronize()
    built = perf_counter()
    step_seconds = []
    for _ in range(2):
        tick = perf_counter()
        controller_step(workload, shell)
        synchronize()
        step_seconds.append(perf_counter() - tick)
    workload.scene.rigid_solver.check_errno()
    if shell:
        finite = torch.isfinite(workload.observation.tensors).all().item()
    elif workload.stress:
        finite = torch.isfinite(workload.link.get_stress_field(copy=False)[0]).all().item()
    else:
        finite = torch.isfinite(workload.egg.get_pos()).all().item()
    assert finite
    write_json(
        args.output / "result.json",
        {
            "kind": "fresh-cache-full-field-activation",
            "identity": runtime_identity(),
            "conditions": file_id(args.conditions),
            "args": {key: str(value) if hasattr(value, "as_posix") else value for key, value in vars(args).items()},
            "genesis_initialization_seconds": initialized - started,
            "construction_and_build_seconds": built - initialized,
            "init_build_and_first_two_steps_seconds": perf_counter() - started,
            "first_step_seconds": step_seconds[0],
            "second_step_seconds": step_seconds[1],
            "full_fields_finite": bool(finite),
            "cache_paths": {
                name: os.environ[name]
                for name in ("GS_CACHE_FILE_PATH", "QD_OFFLINE_CACHE_FILE_PATH", "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR")
            },
            "scope": "A fresh task group/cache per configuration. Cold kernel/model setup and first physical steps; not a primary throughput sample or physical reference qualification. Source imports occur before the initialization timer. No unrelated shared caches are reset.",
        },
    )


if __name__ == "__main__":
    main()
