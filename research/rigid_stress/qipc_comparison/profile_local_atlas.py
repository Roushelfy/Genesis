"""Trace the actual native local-feature recovery path on frozen or evolving Franka inputs.

This is an intrusive diagnostic. Use separate processes for normal graphs and
QD_GRAPH=0; uninstrumented module/scene benchmarks supply throughput claims.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.rigid_stress import GeometryFrame, restore_frame
from examples.speed_benchmark.stress_stages import kernel_restore_geometry
from genesis.utils.array_class import DataKind
from genesis.utils.misc import qd_to_numpy, qd_to_torch

from ..shell_comparison.memory import ProcessMemory
from .artifacts import file_id, runtime_identity, write_json
from .check_local_feature_spaces import source_parents


KERNEL_CATEGORIES = (
    ("fused_exact_pipeline", ("kernel_pipeline",)),
    ("selection", ("kernel_select", "kernel_coarse_feature_select")),
    ("grouping", ("kernel_group_wrench",)),
    ("projection", ("kernel_group_rhs",)),
    ("fallback_load_and_add", ("kernel_fallback_rhs", "kernel_fallback_add")),
    ("solve_and_response", ("kernel_solve", "kernel_apply_inverse", "kernel_feature_boundary_apply")),
    ("response_load_packing", ("kernel_feature_boundary_pack",)),
    ("lift", ("kernel_group_lift",)),
    ("spin_and_balance", ("kernel_atlas_begin", "kernel_balance_impl")),
    ("stress_output", ("kernel_peak_impl", "kernel_peak_")),
    (
        "acceptance_and_solve_state",
        (
            "kernel_full_residual_impl",
            "kernel_group_diagnostics",
            "kernel_accept",
            "kernel_begin_step",
            "kernel_direct_init",
            "kernel_group_begin",
            "kernel_primary_factor_done",
        ),
    ),
    (
        "contact_pressure_and_scatter",
        (
            "kernel_associate",
            "kernel_anchor",
            "kernel_pressure",
            "kernel_fit_scatter_moments",
            "kernel_scatter",
            "kernel_balance_contacts",
        ),
    ),
)


def category(name):
    for label, needles in KERNEL_CATEGORIES:
        if any(needle in name for needle in needles):
            return label
    return "other"


def runtime_checks(workload):
    """Read original acceptance and source-space gates after the traced calls."""
    solver = workload.scene.rigid_solver
    solver.check_errno()
    entry = solver.stress_recovery.links[0]
    valid = qd_to_numpy(entry.contacts.valid, transpose=True)
    force = qd_to_numpy(entry.contacts.force, transpose=True)
    magnitude = np.linalg.norm(force, axis=-1)
    loaded = valid & (magnitude > 0)
    radius = qd_to_numpy(entry.contacts.radius, transpose=True)
    assert (radius[loaded] > 0).all()
    normalized_force = qd_to_numpy(entry.contacts.force_error, transpose=True)[loaded] / magnitude[loaded]
    normalized_moment = qd_to_numpy(entry.contacts.moment_error, transpose=True)[loaded] / (
        radius[loaded] * magnitude[loaded]
    )
    assert (normalized_force <= 1e-8).all() and (normalized_moment <= 1e-8).all()
    assert qd_to_numpy(entry.state.valid).all() and qd_to_numpy(entry.state.step_valid).all()
    assert np.isfinite(qd_to_numpy(entry.state.displacement)).all()
    assert np.isfinite(qd_to_numpy(entry.state.peak)).all()
    atlas = entry.feature_atlas
    states = [entry.state] if atlas is None else [atlas.fallback_state, *(group.state for group in atlas.groups)]
    numerical = []
    for i_group, state in enumerate(states):
        residual = np.sqrt(qd_to_numpy(state.residual_norm_squared))
        rhs = np.sqrt(qd_to_numpy(state.rhs_norm_squared))
        gate = np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * rhs)
        assert qd_to_numpy(state.valid).all() and (residual <= gate).all()
        numerical.append(
            dict(
                group=i_group,
                residual_norm=residual.tolist(),
                rhs_norm=rhs.tolist(),
                gate=gate.tolist(),
                fallbacks=qd_to_numpy(state.fallbacks).tolist(),
            )
        )
    result = dict(
        loaded_contacts=int(loaded.sum()),
        normalized_force_error_max=float(normalized_force.max(initial=0)),
        normalized_moment_error_max=float(normalized_moment.max(initial=0)),
        source_spaces=numerical,
        invalid_steps=qd_to_numpy(entry.state.invalid_steps).tolist(),
    )
    if atlas is not None:
        weights = qd_to_numpy(atlas.regions.state.weights, transpose=True)
        result.update(
            active_groups=qd_to_numpy(atlas.state.active, transpose=True).tolist(),
            coverage_fallbacks=qd_to_numpy(atlas.state.coverage_fallbacks).tolist(),
            numerical_fallbacks=qd_to_numpy(atlas.state.numerical_fallbacks).tolist(),
            mother_weight_sum=float(weights[:, 0, :][loaded].sum()),
            spin_invalid=qd_to_numpy(atlas.state.spin_invalid).tolist(),
        )
        assert not qd_to_numpy(atlas.state.spin_invalid).any()
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--envs", type=int, default=4)
    cli.add_argument("--level", type=int, choices=(1, 2), default=1)
    cli.add_argument("--scope", choices=("module", "scene"), default="module")
    cli.add_argument("--output-mode", choices=("max", "full"), default="full")
    cli.add_argument("--warmup", type=int, default=130)
    cli.add_argument("--steps", type=int, default=10)
    cli.add_argument("--atlas-regions", type=int, default=3)
    cli.add_argument("--atlas-basis", type=Path)
    cli.add_argument("--atlas-core-scale", type=float, default=1.0)
    cli.add_argument("--atlas-halo", type=int, default=0)
    cli.add_argument("--atlas-transition", type=float, default=0.003)
    cli.add_argument("--atlas-solver", choices=("factor", "inverse", "response"), default="factor")
    cli.add_argument("--atlas-space", choices=("local", "coarse"), default="local")
    cli.add_argument("--atlas-shell-bending", action="store_true")
    cli.add_argument("--inverse-max-bytes", type=int, default=512 * 1024 * 1024)
    cli.add_argument("--quadrature", type=int, default=16)
    cli.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    cli.add_argument("--synchronize-calls", action="store_true")
    cli.add_argument("--sample-memory", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if min(args.envs, args.warmup, args.steps) < 1:
        cli.error("envs, warmup and steps must be positive")
    assert os.environ.get("SLURM_JOB_ID"), "Scientific execution requires a Slurm job."
    assert os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid", "Use rtx-mid."
    assert os.environ.get("GENESIS_IMAGE_VER") == "1_26", "Use image 1_26."
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        primary_timing=False,
        identity=runtime_identity(),
        arguments={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        conditions=file_id(args.conditions),
        graph_environment=os.environ.get("QD_GRAPH", "default"),
        profiler_environment=os.environ.get("QD_KERNEL_PROFILER", "default"),
        category_rules=[dict(category=label, kernel_substrings=needles) for label, needles in KERNEL_CATEGORIES],
        scope_note="Intrusive actual recovery/scene trace, with original runtime gates. CUDA task sums can overlap and are not scene latency. Solve category includes feature and exact-fallback factor calls; their raw kernel names remain available. Source-space numerical acceptance does not certify spatial approximation.",
    )
    write_json(args.output / "result.json", result)
    sampler = ProcessMemory(args.sample_memory)
    sampler.start()
    try:
        started = time.perf_counter()
        gs.init(backend=gs.cuda, precision="64", logging_level="warning")
        basis_path = args.atlas_basis
        if args.stress_backend == "shell" and args.atlas_regions and basis_path is None:
            assets = Path(gs.__file__).resolve().parent.parent / f"examples/rigid/assets/hollow_egg/level{args.level}"
            parents, _xyz, provenance = source_parents(assets, True)
            basis_path = args.output / "basis.npz"
            np.savez(basis_path, parent_nodes=parents)
            result["basis_provenance"] = provenance
        result["basis"] = file_id(basis_path) if basis_path is not None else None
        workload = FrankaEgg(
            args.envs,
            level=args.level,
            conditions=args.conditions,
            output_mode=args.output_mode,
            egg_geometry="mesh64",
            radius_model="hertz",
            pressure_profile="hertz",
            box_edge_policy="planar",
            quadrature=args.quadrature,
            pressure_quadrature=args.pressure_quadrature,
            stress_backend=args.stress_backend,
            atlas_regions=args.atlas_regions,
            atlas_basis=basis_path,
            atlas_core_scale=args.atlas_core_scale,
            atlas_halo=args.atlas_halo,
            atlas_transition=args.atlas_transition,
            atlas_solver=args.atlas_solver,
            atlas_space=args.atlas_space,
            atlas_shell_bending=args.atlas_shell_bending,
            inverse_max_bytes=args.inverse_max_bytes,
        )
        qd.sync()
        result["setup_seconds_including_JIT"] = time.perf_counter() - started
        solver = workload.scene.rigid_solver
        entry = solver.stress_recovery.links[0]
        result["options"] = entry.options.model_dump(mode="json")
        result["atlas"] = asdict(entry.feature_atlas.metadata()) if entry.feature_atlas is not None else None
        result["mother_build_timings_including_JIT"] = asdict(entry.model.build_timings)
        geometry = None
        with torch.inference_mode():
            for tick in range(args.warmup):
                if tick == args.warmup - 1 and args.scope == "module":
                    solver.update_forward_pos()
                    geometry = GeometryFrame(
                        np.ascontiguousarray(qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True, copy=True)),
                        np.ascontiguousarray(qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True, copy=True)),
                    )
                workload.step()
            qd.sync()
            result["warmup_checks"] = runtime_checks(workload)
            if args.scope == "module":
                original_force = qd_to_numpy(entry.state.force, copy=True)
                original_q = qd_to_numpy(entry.state.displacement, copy=True)
                original_peak = qd_to_numpy(entry.state.peak, copy=True)
                restore_frame(workload)
                kernel_restore_geometry(geometry.position, geometry.quaternion, solver.dyn_state)
                solver.stress_recovery.recover(0)
                qd.sync()
                np.testing.assert_allclose(qd_to_numpy(entry.state.force), original_force, rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(qd_to_numpy(entry.state.displacement), original_q, rtol=1e-9, atol=1e-13)
                np.testing.assert_allclose(qd_to_numpy(entry.state.peak), original_peak, rtol=1e-9, atol=1e-6)
                result["frozen_input_reproduction_passed"] = True
            invalid_before = qd_to_numpy(entry.state.invalid_steps, copy=True)
            with torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                record_shapes=False,
                with_stack=False,
            ) as profiler:
                started = time.perf_counter()
                for _ in range(args.steps):
                    with torch.profiler.record_function("LocalFeatureAtlas." + args.scope):
                        if args.scope == "module":
                            solver.stress_recovery.recover(0)
                        else:
                            workload.step()
                        if args.synchronize_calls:
                            qd.sync()
                qd.sync()
                result["intrusive_wall_seconds"] = time.perf_counter() - started
            result["final_checks"] = runtime_checks(workload)
            np.testing.assert_array_equal(qd_to_numpy(entry.state.invalid_steps), invalid_before)
        trace_path = args.output / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        result["trace"] = file_id(trace_path)
        events = json.loads(trace_path.read_text())["traceEvents"]
        device = [
            event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in event
        ]
        assert any(event.get("cat") == "kernel" for event in device), "No CUDA kernel trace was captured."
        counts = Counter((event["cat"], event["name"]) for event in device)
        result["device_kernels"] = [
            dict(
                name=name,
                trace_category=kind,
                category=category(name) if kind == "kernel" else kind,
                calls=count,
                sum_duration_us=sum(event["dur"] for event in device if event["cat"] == kind and event["name"] == name),
            )
            for (kind, name), count in counts.most_common()
        ]
        labels = sorted({row["category"] for row in result["device_kernels"]})
        result["device_categories"] = [
            dict(
                category=label,
                calls=sum(row["calls"] for row in result["device_kernels"] if row["category"] == label),
                sum_duration_us=sum(
                    row["sum_duration_us"] for row in result["device_kernels"] if row["category"] == label
                ),
                sum_duration_us_per_call=sum(
                    row["sum_duration_us"] for row in result["device_kernels"] if row["category"] == label
                )
                / args.steps,
            )
            for label in labels
        ]
        allocations, buffers = {}, []
        for item in solver.stress_recovery.data:
            if item.kind == DataKind.CONFIG:
                continue
            if math.prod(item.value.shape) == 0:
                buffers.append(
                    dict(name=item.name, bytes=0, kind=item.kind.name, shape=list(item.value.shape), alias_of=None)
                )
                continue
            view = qd_to_torch(item.value, copy=False)
            pointer, size = view.data_ptr(), view.numel() * view.element_size()
            alias_of = allocations[pointer][0] if pointer in allocations else None
            if alias_of is None:
                allocations[pointer] = (item.name, size, item.kind)
            buffers.append(
                dict(name=item.name, bytes=size, kind=item.kind.name, shape=list(view.shape), alias_of=alias_of)
            )
        result.update(
            native_buffers=buffers,
            native_unique_bytes=sum(row[1] for row in allocations.values()),
            native_unique_bytes_by_kind={
                kind.name: sum(row[1] for row in allocations.values() if row[2] == kind)
                for kind in DataKind
                if kind != DataKind.CONFIG
            },
            native_shared_constant_bytes=sum(row[1] for row in allocations.values() if row[2] == DataKind.CONSTANT),
            native_nonconstant_bytes_per_environment=sum(
                row[1] for row in allocations.values() if row[2] != DataKind.CONSTANT
            )
            / args.envs,
            memory_inventory_note="Unique exposed native buffers by allocation pointer; raw shapes remain available. Per-environment bytes divide observed nonconstant storage by envs, including any fixed scratch. Compiler/CUDA allocations are excluded; process polling and Torch peaks have separate scopes.",
            process_maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            device_free_bytes=torch.cuda.mem_get_info()[0],
            complete=True,
            passed=True,
        )
    finally:
        result["process_memory"] = sampler.finish()
        write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
