"""Measure automatic shell preparation and the native Franka module/scene, with untimed acceptance and field checks."""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import quadrants as qd
import torch
from PIL import Image

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.rigid.stress_field_view import StressFieldSurface
from examples.rigid.stress_mesh_asset import prepare_stress_mesh, write_rigid_asset
from examples.speed_benchmark.rigid_stress import GeometryFrame, prepare_recovery_snapshot
from genesis.utils.array_class import DataKind
from genesis.utils.geom import trans_quat_to_T
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array
from research.rigid_stress.qipc_comparison.artifacts import file_id, runtime_identity, write_json
from research.rigid_stress.qipc_comparison.profile_planar_loading import _api, _checks
from research.rigid_stress.shell_comparison.automatic_shell_geometry import surface_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=("egg", "grasp", "round", "neck", "u", "plate"), default="grasp")
    parser.add_argument("--mesh-mode", choices=("automatic", "explicit", "authored"), default="automatic")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--warmup", type=int, default=130)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--audit-substeps", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--asset-cache", type=Path)
    parser.add_argument("--loading-subdivisions", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "identity": runtime_identity(),
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "actual_genesis_source": str(Path(gs.__file__).resolve()),
    }
    gs.init(backend=gs.gpu, precision="64", seed=510000, logging_level="warning")
    options = gs.options.RigidStressOptions.dpr(
        mesh=Path(gs.__file__).resolve().parents[1] / "examples/rigid/assets/hollow_egg/level1/shell.npz",
        backend="shell",
        shell_loading="planar",
        loading_subdivisions=args.loading_subdivisions,
        quadrature=10,
        scatter_tasks_per_env=128,
        box_edge_policy="planar",
        output_mode=args.output_mode,
    )
    rigid_mesh = None
    if args.mesh_mode != "authored":
        surface = surface_recipe(args.shape, 1 if args.shape == "neck" else 0)
        source = args.output / "midsurface.obj"
        surface.export(source, digits=17)
        asset_cache = args.asset_cache or args.output / "cache"
        existing_assets = set(asset_cache.glob("*/elastic.npz"))
        started = time.perf_counter()
        prepared = prepare_stress_mesh(source, None, "shell", 1.0, None, asset_cache)
        results["conversion_cold_seconds"] = time.perf_counter() - started
        results["preparation_first_cache_hit"] = prepared.elastic in existing_assets
        started = time.perf_counter()
        prepare_stress_mesh(source, None, "shell", 1.0, None, asset_cache)
        results["conversion_warm_seconds"] = time.perf_counter() - started
        if args.mesh_mode == "explicit":
            started = time.perf_counter()
            prepared = prepare_stress_mesh(prepared.elastic, None, "shell", 1.0, None, asset_cache)
            results["explicit_preparation_seconds"] = time.perf_counter() - started
        options.mesh = prepared.elastic
        started = time.perf_counter()
        field = StressFieldSurface(prepared.elastic, "shell", options.shell_thickness)
        rigid_file = write_rigid_asset(prepared.collision, field.material_mesh, options.density)
        results["material_urdf_seconds"] = time.perf_counter() - started
        results["material_mass"] = field.material_mesh.volume * options.density
        results["collision"] = file_id(prepared.collision)
        rigid_mesh = gs.morphs.URDF(
            pos=(0.65, 0, 0.001 - prepared.minimum_z),
            file=rigid_file,
            align=False,
            convexify=False,
            decimate=False,
            watertighten=None,
            preserve_mesh_precision=True,
        )
    results["elastic"] = file_id(options.mesh)
    write_json(args.output / "progress.json", results)
    viewer_options = None
    if args.viewer:
        if args.envs != 1 or args.output_mode != "full":
            raise ValueError("Viewer capture needs --envs 1 --output-mode full.")
        field = StressFieldSurface(options.mesh, "shell", options.shell_thickness)
        if rigid_mesh is not None:
            rigid_mesh.visualization = False
        viewer_options = gs.options.ViewerOptions(
            res=(800, 600),
            run_in_thread=False,
            realtime_factor=None,
            camera_pos=(0.80, -0.23, 0.07),
            camera_lookat=(0.65, 0, 0.04),
        )
    started = time.perf_counter()
    workload = FrankaEgg(
        n_envs=args.envs,
        substeps=args.substeps,
        stress_options=options,
        rigid_mesh=rigid_mesh,
        seed=510000,
        viewer=args.viewer,
        stress_visualization=args.viewer,
        viewer_options=viewer_options,
    )
    qd.sync()
    results["scene_build_seconds_including_JIT"] = time.perf_counter() - started
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    results["shell_build_seconds_including_JIT"] = asdict(entry.model.build_timings)
    results["quads"] = entry.model.shell.quads.shape[0]
    results["dofs"] = 3 * entry.model.info.vertices.shape[0]
    write_json(args.output / "progress.json", results)
    if args.audit_substeps:
        workload.scene.register_post_substep_callback(lambda index: solver.check_errno())

    geometry = None

    def capture_geometry(index):
        nonlocal geometry
        if index == args.substeps - 1:
            solver.update_forward_pos()
            geometry = GeometryFrame(
                np.ascontiguousarray(qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True)),
                np.ascontiguousarray(qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True)),
            )

    for tick in range(args.warmup):
        if tick == args.warmup - 1:
            workload.scene.register_pre_substep_callback(capture_geometry)
        try:
            workload.step()
        except gs.GenesisException as error:
            write_json(args.output / "failure.json", {"step": tick + 1, "error": str(error)})
            arrays = {
                item.name: qd_to_numpy(item.value, transpose=True)
                for item in solver.stress_recovery.data
                if item.kind != DataKind.CONFIG
                and math.prod(item.value.shape) > 0
                and any(
                    word in item.name
                    for word in ("contacts", "sources", "hertz", "radius", "curvature", "stress.0.state.")
                )
            }
            for item in solver.collider.data:
                if item.kind != DataKind.CONFIG and math.prod(item.value.shape) > 0:
                    arrays[item.name] = qd_to_numpy(item.value, transpose=True)
            arrays["geom_position"] = qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True)
            arrays["geom_quaternion"] = qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True)
            np.savez_compressed(args.output / "failure-state.npz", **arrays)
            raise
    workload.scene._pre_substep_callbacks.remove(capture_geometry)
    qd.sync()
    results["acceptance_before"] = _checks(_api(), workload)
    assert results["acceptance_before"]["passed"] and results["acceptance_before"]["loaded_contacts"] > 0
    results["snapshot"] = prepare_recovery_snapshot(workload, geometry)
    results["warmup_peak_Pa"] = tensor_to_array(workload.link.get_max_stress()).tolist()
    np.savez_compressed(
        args.output / "contacts.npz",
        valid=qd_to_numpy(entry.contacts.valid, transpose=True),
        position=qd_to_numpy(entry.contacts.position, transpose=True),
        force=qd_to_numpy(entry.contacts.force, transpose=True),
        radius=qd_to_numpy(entry.contacts.radius, transpose=True),
        task_count=qd_to_numpy(entry.planar.state.task_count, transpose=True),
        sample_count=qd_to_numpy(entry.planar.state.sample_count, transpose=True),
        affine=qd_to_numpy(entry.planar.state.affine, transpose=True),
        fit_evaluations=qd_to_numpy(entry.planar.state.fit_evaluations, transpose=True),
    )
    if args.output_mode == "full":
        tensor, vm = map(tensor_to_array, workload.link.get_stress_field())
        np.savez_compressed(args.output / "warmup-fields.npz", tensor=tensor, von_mises=vm)
    if args.viewer:
        started = time.perf_counter()
        colored = field.colored_mesh(vm[0], vm.max())
        transform = trans_quat_to_T(workload.link.get_pos(envs_idx=[0])[0], workload.link.get_quat(envs_idx=[0])[0])
        with workload.scene.visualizer.viewer_lock:
            overlay = workload.scene.draw_debug_mesh(colored, T=transform)
            overlay.is_marker = False
        workload.scene.visualizer.update(force=True)
        viewer = workload.scene.visualizer.viewer._pyrender_viewer
        viewer.switch_to()
        pixels = viewer._renderer.jit.read_color_buf(*viewer.get_size(), rgba=False)
        Image.fromarray(pixels).save(args.output / "viewer.png")
        results["viewer_readback_seconds"] = time.perf_counter() - started
        results["viewer_surface"] = file_id(args.output / "viewer.png")
    module_times = []
    for _ in range(args.repeats):
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.steps):
            solver.stress_recovery.recover(0)
        qd.sync()
        module_times.append((time.perf_counter() - started) / args.steps)
    results["module_seconds_per_step"] = module_times
    results["module_acceptance_after"] = _checks(_api(), workload)
    if args.profile:
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA)
        ) as profiler:
            for _ in range(5):
                solver.stress_recovery.recover(0)
            qd.sync()
        profiler.export_chrome_trace(str(args.output / "module-trace.json"))
    # A reset re-establishes matching moving inputs after the frozen module measurements.
    workload.restart()
    for _ in range(args.warmup):
        workload.step()
    scene_times = []
    for _ in range(args.repeats):
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.steps):
            workload.step()
        qd.sync()
        scene_times.append((time.perf_counter() - started) / args.steps)
        solver.check_errno()
    results["scene_seconds_per_step"] = scene_times
    results["scene_acceptance_after"] = _checks(_api(), workload)
    results["maximum_stress_Pa"] = tensor_to_array(workload.link.get_max_stress()).tolist()
    results["last_substep_peak_Pa"] = qd_to_numpy(entry.state.peak).tolist()
    if args.output_mode == "full":
        tensor, vm = map(tensor_to_array, workload.link.get_stress_field())
        np.savez_compressed(args.output / "fields.npz", tensor=tensor, von_mises=vm)
        latest_peak = vm.reshape((args.envs, -1)).max(axis=1)
        np.testing.assert_allclose(latest_peak, results["last_substep_peak_Pa"], rtol=1e-12)
        assert (np.array(results["maximum_stress_Pa"]) >= latest_peak * (1 - 1e-12)).all()
        if args.substeps == 1:
            np.testing.assert_allclose(latest_peak, results["maximum_stress_Pa"], rtol=1e-12)
    allocations = {}
    buffers = []
    for item in solver.stress_recovery.data:
        if item.kind == DataKind.CONFIG or math.prod(item.value.shape) == 0:
            continue
        view = qd_to_torch(item.value, copy=False)
        pointer, size = view.data_ptr(), view.numel() * view.element_size()
        if pointer not in allocations:
            allocations[pointer] = size
            buffers.append({"name": item.name, "shape": list(view.shape), "bytes": size, "kind": int(item.kind)})
    results["native_allocated_bytes"] = sum(allocations.values())
    results["native_buffers"] = buffers
    # Public partial reset must preserve other environments and invalidate the reset output.
    before = tensor_to_array(workload.link.get_max_stress()).copy()
    workload.reset(np.array([0]))
    after = tensor_to_array(workload.link.get_max_stress())
    np.testing.assert_array_equal(after[1:], before[1:])
    np.testing.assert_array_equal(after[0], 0)
    results["partial_reset_passed"] = True
    results["passed"] = all(
        results[key]["passed"] for key in ("acceptance_before", "module_acceptance_after", "scene_acceptance_after")
    )
    write_json(args.output / "results.json", results)
    assert results["passed"], results


if __name__ == "__main__":
    main()
