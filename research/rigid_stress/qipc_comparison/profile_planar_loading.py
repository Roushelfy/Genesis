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
    ("proxy_moments", ("kernel_planar_moments",)),
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
    from examples.speed_benchmark.rigid_stress import GeometryFrame, prepare_recovery_snapshot
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
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--scope", choices=("module", "scene"), default="module")
    parser.add_argument("--shell-loading", choices=("q4", "planar"), default="planar")
    parser.add_argument("--loading-subdivisions", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    parser.add_argument("--tasks-per-env", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=130)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if min(args.envs, args.warmup, args.steps, args.tasks_per_env) < 1:
        parser.error("Environment, warmup, trace-step and task counts must be positive.")
    args.output.mkdir(parents=True, exist_ok=False)
    api = _api()
    api.gs.init(backend=api.gs.gpu, precision="64", seed=510000, logging_level="warning")
    started = time.perf_counter()
    workload = api.FrankaEgg(
        args.envs,
        conditions=args.conditions,
        stress_backend="shell",
        shell_loading=args.shell_loading,
        loading_subdivisions=args.loading_subdivisions,
        output_mode=args.output_mode,
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        egg_geometry="mesh64",
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        scatter_tasks_per_env=args.tasks_per_env,
    )
    setup_seconds = time.perf_counter() - started
    geometry = None
    for tick in range(args.warmup):
        if tick == args.warmup - 1 and args.scope == "module":
            solver = workload.scene.rigid_solver
            solver.update_forward_pos()
            geometry = api.GeometryFrame(
                api.np.ascontiguousarray(api.qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True)),
                api.np.ascontiguousarray(api.qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True)),
            )
        workload.step()
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
        "before_checks": before,
        "after_checks": after,
        "routes_before": routes_before,
        "routes_after": routes_after,
        "actual_method": entry.model.selected_method,
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
