"""Native half of the matched workload; the external shell solver is never imported."""

from time import perf_counter

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg

from .common import file_identity, identity, synchronize, vars_json, write_json
from .memory import ProcessMemory
from .rollout import make_parser, profile_run, run


def main():
    args = make_parser().parse_args()
    assert args.model != "shell" and not args.instrumented
    assert args.output_mode == "full" or (not args.record_every and not args.peak_snapshots)
    assert args.sim_dt == 0.01
    args.output.mkdir(parents=True, exist_ok=False)
    memory = ProcessMemory(args.memory)
    memory.start()
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    started = perf_counter()
    workload = FrankaEgg(
        args.envs,
        level=args.level,
        stress=args.model != "rigid",
        conditions=args.conditions,
        substeps=args.substeps,
        output_mode=args.output_mode,
        native_controller=False,
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        contact_model="intrinsic" if args.model == "intrinsic" else "pressure",
        intrinsic_load_partition="body_source_prior",
        scatter_tasks_per_env=128 if args.model == "pressure-tuned" else 32,
        contact_wrench_reuse=True if args.model == "pressure-tuned" else None,
        contact_moment_reuse=True if args.model == "pressure-tuned" else None,
        packed_block_size=512 if args.model == "pressure-tuned" else None,
        coalesced_residual=True if args.model == "pressure-tuned" else None,
    )
    synchronize()
    build_seconds = perf_counter() - started
    result = profile_run(args, workload, False) if args.trace else run(args, workload, False)
    numerical = None
    if workload.stress:
        entry = workload.scene.rigid_solver.stress_recovery.links[0]
        numerical = {
            "requested_options": workload.link.stress_options.model_dump(mode="json"),
            "dofs": entry.model.info.vertices.shape[0] * 3,
            "tetrahedra": entry.model.info.elements.shape[0],
            "selected_method": entry.model.selected_method,
            "surface_inverse": entry.model.surface_inverse is not None,
            "fused_pipeline": entry.fused_pipeline,
            "reuse_moments": entry.reuse_contact_moments,
            "effective_packed_block_size": entry.packed_block_size,
            "fused_rigid_post": workload.scene.rigid_solver.stress_recovery.post_kernel is not None,
        }
    result.update(
        kind="matched-franka-egg",
        args=vars_json(args),
        identity=identity(),
        condition_bank=file_identity(args.conditions),
        build_seconds=build_seconds,
        source_scope="native-unchanged-engine",
        process_memory=memory.finish(),
        native_numerics=numerical,
        rigid_dofs=workload.scene.rigid_solver.n_dofs,
    )
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
