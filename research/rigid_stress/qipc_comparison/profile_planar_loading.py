"""Trace actual shell loading/recovery calls, separately from primary throughput.

Scientific execution requires Slurm rtx-mid and image 1_26. The module scope
replays a validated current contact snapshot; scene scope advances the native
controller and rigid solver. Neither scope measures policy training.
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = (
    ("fused_shell_recovery", ("kernel_recover_",)),
    ("proxy_query", ("kernel_planar_query",)),
    ("material_guard", ("kernel_planar_material", "kernel_material_")),
    ("proxy_geometry_preparation", ("kernel_planar_prepare_moments",)),
    ("proxy_moments", ("kernel_planar_moments", "kernel_planar_radial_moments")),
    ("proxy_fit", ("kernel_planar_fit",)),
    ("proxy_validation", ("kernel_planar_validate",)),
    ("proxy_scatter", ("kernel_planar_scatter",)),
    ("anchor", ("kernel_anchor",)),
    ("association_radius", ("kernel_associate", "kernel_estimate_radii")),
    ("original_loading", ("kernel_pressure", "kernel_fit_scatter_moments", "kernel_scatter")),
    ("balance", ("kernel_balance",)),
    ("response", ("kernel_apply", "kernel_solve", "kernel_direct_solve")),
    ("stress_output", ("kernel_peak", "kernel_stress")),
    ("acceptance_state", ("kernel_full_residual", "kernel_accept", "kernel_begin_step", "kernel_direct_init")),
)


def _category(name):
    for label, needles in CATEGORIES:
        if any(needle in name for needle in needles):
            return label
    return "other_rigid_control_and_runtime"


def _checks(api, workload):
    solver = workload.scene.rigid_solver
    solver.check_errno()
    entry = solver.stress_recovery.links[0]
    valid = api.qd_to_numpy(entry.contacts.valid, transpose=True)
    magnitude = api.np.linalg.norm(api.qd_to_numpy(entry.contacts.force, transpose=True), axis=-1)
    loaded = valid & (magnitude > 1e-30)
    radius = api.qd_to_numpy(entry.contacts.radius, transpose=True)
    force_error = api.qd_to_numpy(entry.contacts.force_error, transpose=True)[loaded] / magnitude[loaded]
    moment_error = api.qd_to_numpy(entry.contacts.moment_error, transpose=True)[loaded] / (
        magnitude[loaded] * radius[loaded]
    )
    residual = api.np.sqrt(api.qd_to_numpy(entry.state.residual_norm_squared))
    rhs = api.np.sqrt(api.qd_to_numpy(entry.state.rhs_norm_squared))
    gate = api.np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * rhs)
    checks = {
        "native_valid": bool(api.qd_to_numpy(entry.state.valid).all()),
        "step_valid": bool(api.qd_to_numpy(entry.state.step_valid).all()),
        "finite_output": bool(api.np.isfinite(api.qd_to_numpy(entry.state.peak)).all()),
        "original_unscaled_residual": bool((residual <= gate).all()),
        "positive_loaded_radii": bool((radius[loaded] > 0).all()),
        "original_force": bool((force_error <= 1e-8).all()),
        "original_point_moment": bool((moment_error <= 1e-8).all()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "loaded_contacts": int(loaded.sum()),
        "maximum_relative_force_error": float(force_error.max(initial=0)),
        "maximum_relative_moment_error": float(moment_error.max(initial=0)),
        "maximum_residual_over_gate": float((residual / gate).max(initial=0)),
    }


def _api():
    from types import SimpleNamespace

    import numpy as np
    import quadrants as qd
    import torch

    import genesis as gs
    from examples.rigid.franka_egg_stress import FrankaEgg
    from examples.speed_benchmark.rigid_stress import (
        GeometryFrame,
        anchor_execution_modes,
        planar_query_execution_modes,
        prepare_recovery_snapshot,
    )
    from examples.speed_benchmark.rigid_stress_snapshot import (
        restore_module_snapshot,
        snapshot_bindings,
        verify_module_snapshot,
    )
    from genesis.utils.misc import qd_to_numpy

    from .artifacts import runtime_identity

    return SimpleNamespace(**locals())


def _routes(api, workload):
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    if entry.planar is None:
        return None
    return {
        name: api.qd_to_numpy(field, transpose=True).tolist()
        for name, field in (
            ("attempted_loaded_contacts", entry.planar.state.attempted_loaded_contacts),
            ("proxy_contacts", entry.planar.state.proxy_contacts),
            ("fallback_contacts", entry.planar.state.fallback_contacts),
            ("analytic_sections", entry.planar.state.analytic_sections),
            ("overflow_steps", entry.planar.state.overflow_steps),
            ("material_queries", entry.planar.material_state.query_contacts),
            ("material_rejections", entry.planar.material_state.rejected_contacts),
        )
    }


def _task_work(api, workload):
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    if entry.planar is None:
        return None
    state = entry.planar.state
    tasks = api.qd_to_numpy(state.task_count, transpose=True)
    samples = api.qd_to_numpy(state.sample_count, transpose=True)
    eligible = api.qd_to_numpy(state.eligible, transpose=True)
    reasons = api.qd_to_numpy(state.reason, transpose=True)
    return {
        "scope": "Untimed terminal workload diagnostics; task capacity is allocated per environment. Positive integration nodes count angular rays for exact radial integration and point samples for 2D Gauss integration.",
        "task_capacity_per_environment": state.tasks.shape[0],
        "tasks_per_environment": tasks.tolist(),
        "maximum_tasks_per_environment": int(tasks.max(initial=0)),
        "total_tasks": int(tasks.sum()),
        "positive_samples": int(samples.sum()),
        "eligible_sources": int(eligible.sum()),
        "source_reason_counts": dict(Counter(map(int, reasons.flat))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--scope", choices=("module", "scene"), default="module")
    parser.add_argument("--shell-loading", choices=("q4", "planar"), default="planar")
    parser.add_argument("--loading-subdivisions", type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument("--planar-integration", choices=("radial", "gauss"), default="radial")
    parser.add_argument("--planar-query", choices=("cells", "blocks"), default="cells")
    parser.add_argument("--cooperative-anchor", action="store_true")
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=130)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--module-snapshot", type=Path, help="Trace native recovery on common raw rigid inputs.")
    args = parser.parse_args()
    if min(args.envs, args.warmup, args.steps, args.tasks_per_env) < 1:
        parser.error("Environment, warmup, trace-step and task counts must be positive.")
    if args.module_snapshot is not None and args.scope != "module":
        parser.error("A common raw module snapshot cannot drive a moving scene trace.")
    recipe = None
    if args.module_snapshot is not None:
        recipe = json.loads((args.module_snapshot / "manifest.json").read_text())["bindings"]["recipe"]
    args.output.mkdir(parents=True, exist_ok=False)
    api = _api()
    api.gs.init(
        backend=api.gs.gpu,
        precision="64",
        seed=510000 if recipe is None else recipe["seed"],
        logging_level="warning",
    )
    started = time.perf_counter()
    workload = api.FrankaEgg(
        args.envs,
        level=1 if recipe is None else recipe["level"],
        seed=510000 if recipe is None else recipe["seed"],
        varied=False if recipe is None else recipe["varied"],
        conditions=args.conditions,
        stress_backend="shell",
        shell_loading=args.shell_loading,
        loading_subdivisions=args.loading_subdivisions,
        output_mode=args.output_mode,
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        egg_geometry="mesh64" if recipe is None else recipe["egg_geometry"],
        use_gjk_collision=None
        if recipe is None
        else {"auto": None, "mpr": False, "gjk": True}[recipe["collision_method"]],
        enable_contact_patch=None
        if recipe is None
        else {"auto": None, "on": True, "off": False}[recipe["contact_patch"]],
        native_controller=True if recipe is None else recipe["native_controller"],
        timestep=0.01 if recipe is None else recipe["dt"],
        substeps=1 if recipe is None else recipe["substeps"],
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        scatter_tasks_per_env=args.tasks_per_env,
    )
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    entry.cooperative_anchor = args.cooperative_anchor
    if entry.planar is not None:
        entry.planar.exact_radial = args.planar_integration == "radial"
        entry.planar.hierarchical_query = args.planar_query == "blocks"
    setup_seconds = time.perf_counter() - started
    geometry = None
    scene_warmup_steps = args.warmup if args.module_snapshot is None else 0
    for tick in range(scene_warmup_steps):
        if tick == args.warmup - 1 and args.scope == "module":
            solver = workload.scene.rigid_solver
            solver.update_forward_pos()
            geometry = api.GeometryFrame(
                api.np.ascontiguousarray(api.qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True)),
                api.np.ascontiguousarray(api.qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True)),
            )
        workload.step()
    module_snapshot = None
    module_warmup_calls = 0
    snapshot_prepare_seconds = None
    if args.module_snapshot is not None:
        prepare_started = time.perf_counter()
        bindings = api.snapshot_bindings(workload, recipe)
        module_snapshot = api.restore_module_snapshot(workload, args.module_snapshot, bindings)
        solver = workload.scene.rigid_solver
        solver.stress_recovery.recover(0)
        api.qd.sync()
        solver.check_errno()
        module_warmup_calls += 1
        geometry = api.GeometryFrame(
            api.np.ascontiguousarray(api.qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True)),
            api.np.ascontiguousarray(api.qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True)),
        )
        snapshot = api.prepare_recovery_snapshot(workload, geometry, restore_input=False)
        module_warmup_calls += 1
        module_snapshot["raw_inputs_exact_after_preparation"] = (
            api.verify_module_snapshot(workload, args.module_snapshot) == module_snapshot["raw_input_sha256"]
        )
        snapshot_prepare_seconds = time.perf_counter() - prepare_started
    else:
        snapshot = api.prepare_recovery_snapshot(workload, geometry) if args.scope == "module" else None
    before = _checks(api, workload)
    routes_before = _routes(api, workload)
    api.qd.sync()
    trace_start = time.perf_counter()
    with api.torch.profiler.profile(
        activities=(api.torch.profiler.ProfilerActivity.CPU, api.torch.profiler.ProfilerActivity.CUDA),
    ) as profiler:
        for _ in range(args.steps):
            with api.torch.profiler.record_function("PlanarLoading." + args.scope):
                if args.scope == "module":
                    workload.scene.rigid_solver.stress_recovery.recover(0)
                else:
                    workload.step()
        api.qd.sync()
    trace_seconds = time.perf_counter() - trace_start
    trace = args.output / "trace.json"
    profiler.export_chrome_trace(str(trace))
    after = _checks(api, workload)
    if args.module_snapshot is not None:
        module_snapshot["raw_inputs_exact_after_measurement"] = (
            api.verify_module_snapshot(workload, args.module_snapshot) == module_snapshot["raw_input_sha256"]
        )
    routes_after = _routes(api, workload)
    groups, kernels, counts = defaultdict(float), defaultdict(float), Counter()
    for event in json.loads(trace.read_text())["traceEvents"]:
        if event.get("ph") == "X" and event.get("cat") == "kernel" and "dur" in event:
            name = event["name"]
            groups[_category(name)] += event["dur"]
            kernels[name] += event["dur"]
            counts[name] += 1
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    result = {
        "passed": before["passed"] and after["passed"],
        "complete": True,
        "primary_timing": False,
        "identity": api.runtime_identity(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "conditions_sha256": hashlib.sha256(args.conditions.read_bytes()).hexdigest(),
        "actual_native_package": str(Path(api.gs.__file__).resolve()),
        "actual_worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup_seconds_including_JIT": setup_seconds,
        "intrusive_trace_seconds": trace_seconds,
        "snapshot_check": snapshot,
        "module_snapshot": module_snapshot,
        "snapshot_rigid_recipe": recipe,
        "scene_warmup_steps": scene_warmup_steps,
        "module_warmup_calls": module_warmup_calls,
        "snapshot_preparation_seconds_including_IO_JIT": snapshot_prepare_seconds,
        "before_checks": before,
        "after_checks": after,
        "routes_before": routes_before,
        "routes_after": routes_after,
        "terminal_task_work": _task_work(api, workload),
        "actual_method": entry.model.selected_method,
        "cooperative_anchor_requested": args.cooperative_anchor,
        "effective_anchor_modes": api.anchor_execution_modes(workload),
        "planar_query_requested": args.planar_query if entry.planar is not None else None,
        "effective_planar_query_modes": api.planar_query_execution_modes(workload),
        "surface_inverse_available": entry.model.surface_inverse is not None,
        "device_categories": [{"category": name, "sum_duration_us": value} for name, value in sorted(groups.items())],
        "device_kernels": [
            {"name": name, "sum_duration_us": duration, "calls": counts[name]}
            for name, duration in sorted(kernels.items(), key=lambda row: -row[1])
        ],
        "scope": "Intrusive actual recovery/scene kernel trace. Device durations may overlap. Module replays one real frozen contact state; scene includes native controller and rigid work, without policy inference. Neither supplies a primary wall-time speedup or spatial qualification.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    assert result["passed"], "The traced path must preserve original numerical and load acceptance."


if __name__ == "__main__":
    main()
