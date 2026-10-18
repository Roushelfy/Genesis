"""Profile a continuous short QIPC contact window with stress recovery only at declared keyframes."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import qipc._src.native.solver.qipc_solver_ext as extension
from qipc import tetmesh
from qipc.geometry import ground

from .artifacts import file_id, runtime_identity, write_json
from .pose_robot import PoseRobot, pose_contact_table
from .reference import ReferenceConfig, SolidObserver, preserve_contact_failure, solver_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.16)
    parser.add_argument("--window-start", type=float, default=1.05)
    parser.add_argument("--keyframes", type=float, nargs="+", default=[1.10, 1.13, 1.16])
    parser.add_argument("--velocity-tolerance", type=float)
    parser.add_argument("--linear-tolerance", type=float)
    parser.add_argument("--max-pcg", type=int)
    parser.add_argument("--linear-solver", choices=("linear_pcg", "partition_pcg"))
    parser.add_argument("--newton-aware-pcg", action="store_true")
    parser.add_argument("--timing-mode", choices=("segmented", "whole-window"), default="segmented")
    parser.add_argument("--prewarm-elastic-observer", action="store_true")
    parser.add_argument("--defer-field-io", action="store_true")
    parser.add_argument("--stress-every-step", action="store_true", help="Supplementary dense-observer cost control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.window_start < args.duration <= 1.5, "This entrypoint is a bounded short-window control"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    original = json.loads((args.baseline / "result.json").read_text())
    inputs = json.loads((args.baseline / "inputs.json").read_text())
    assert inputs["native"]["sha256"] == file_id(extension.__file__)["sha256"]
    config = ReferenceConfig(**original["config"])
    overrides = {
        key: value
        for key, value in (
            ("velocity_tolerance", args.velocity_tolerance),
            ("linear_tolerance", args.linear_tolerance),
            ("max_pcg", args.max_pcg),
            ("linear_solver", args.linear_solver),
        )
        if value is not None
    }
    config = replace(config, **overrides)
    if args.newton_aware_pcg:
        assert config.linear_solver == "partition_pcg", "Pinned native gate is eligible only for partition_pcg"
        config = replace(config, newton_aware_pcg=True)
        overrides["newton_aware_pcg"] = True
    steps, start = round(args.duration / config.dt), round(args.window_start / config.dt)
    keyframes = {round(time / config.dt) for time in args.keyframes}
    assert all(
        abs(round(time / config.dt) * config.dt - time) < 1e-12 and 0 < time <= args.duration for time in args.keyframes
    )
    with np.load(inputs["motion"]["path"]) as saved:
        motion = {key: saved[key] for key in saved.files}
    sample_dt = motion["time_s"][1] - motion["time_s"][0]
    stride = round(config.dt / sample_dt)
    assert stride > 0 and abs(stride * sample_dt - config.dt) < 1e-12
    with np.load(args.baseline / "rest.npz") as saved:
        rest = {key: saved[key] for key in saved.files}
    setup_started = perf_counter()
    scene = config.scene()
    descriptor_scene = config.scene(contact=False)
    descriptor = descriptor_scene.add_mjcf(args.baseline / "panda-bank-initial.xml", link_type="rigid")
    robot = PoseRobot.create(scene, descriptor, inputs["dbc_stiffness_ratio"])
    geometry = tetmesh(rest["vertices_world_m"], rest["tetrahedra"])
    config.material().apply_to(
        geometry, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
    )
    scene.geometries.create("egg", geometry)
    floor = ground(height=0, N=(0, 0, 1))
    scene.geometries.create("floor", floor)
    pose_contact_table(scene, robot, floor, config.resistance)
    scene.init()
    from qipc._src.native.solver import NewtonAwarePCGGate

    gate = scene.sim_systems.get(NewtonAwarePCGGate)
    assert (gate is not None) == config.newton_aware_pcg, "Requested native gate did not activate as declared"
    with torch.cuda.stream(scene.solver.cuda_stream):
        observer = SolidObserver.allocate(scene, config)
        robot.allocate_targets(scene, motion)
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - setup_started
    observer_activation_seconds = None
    if args.prewarm_elastic_observer:
        x_before, v_before = scene.finite_element.x.clone(), scene.finite_element.velocities.clone()
        begun = perf_counter()
        with torch.cuda.stream(scene.solver.cuda_stream):
            initial_field = observer.update()
            initial_gradient = observer.elastic_gradient(initial_field)
        torch.cuda.synchronize()
        observer_activation_seconds = perf_counter() - begun
        torch.testing.assert_close(scene.finite_element.x, x_before, rtol=0, atol=0)
        torch.testing.assert_close(scene.finite_element.velocities, v_before, rtol=0, atol=0)
        del initial_field, initial_gradient, x_before, v_before
    scene.solver.accd_iter_stats = True
    scene.solver.min_gap_stats = True
    np.savez_compressed(args.output / "rest.npz", **rest)
    write_json(
        args.output / "inputs.json",
        {
            "identity": runtime_identity(),
            "baseline": file_id(args.baseline / "result.json"),
            "baseline_inputs": file_id(args.baseline / "inputs.json"),
            "native": file_id(extension.__file__),
            "motion": inputs["motion"],
            "conditions": inputs["conditions"],
            "config": asdict(config),
            "overrides": overrides,
            "keyframes_s": args.keyframes,
            "window_s": [args.window_start, args.duration],
            "start_state": "Original initial state; uninterrupted physical/contact history through the window",
            "observation_mode": "dense compute, sparse export"
            if args.stress_every_step
            else "sparse compute and export",
            "native_gap_and_CCD_diagnostics_enabled": True,
            "timing_mode": args.timing_mode,
            "initialization_only_elastic_warmups": int(args.prewarm_elastic_observer),
            "defer_field_compression_and_IO_until_after_window": args.defer_field_io,
        },
    )
    timings, rows, observed, payloads = [], [], [], []
    completed, failure = False, None
    observer_calls = 0
    deferred_fields = []
    window_started = None
    total_started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(1, steps + 1):
                sample = step * stride
                timed = step > start
                if timed and window_started is None:
                    torch.cuda.synchronize()
                    window_started = perf_counter()
                stage_times = {}

                def measure(name, function):
                    if not timed:
                        with torch.cuda.stream(scene.solver.cuda_stream):
                            return function()
                    if args.timing_mode == "whole-window":
                        begun = perf_counter()
                        with torch.cuda.stream(scene.solver.cuda_stream):
                            result = function()
                        stage_times[name] = {"wall_seconds": perf_counter() - begun, "cuda_stream_span_ms": None}
                        return result
                    torch.cuda.synchronize()
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record(scene.solver.cuda_stream)
                    begun = perf_counter()
                    with torch.cuda.stream(scene.solver.cuda_stream):
                        result = function()
                    after.record(scene.solver.cuda_stream)
                    torch.cuda.synchronize()
                    stage_times[name] = {
                        "wall_seconds": perf_counter() - begun,
                        "cuda_stream_span_ms": before.elapsed_time(after),
                    }
                    return result

                observe = args.stress_every_step or step in keyframes
                previous = (
                    measure("observer_prepare", lambda: scene.finite_element.velocities.clone()) if observe else None
                )
                measure("controller", lambda: robot.push(sample))
                measure("solver_step", scene.step)

                def status():
                    result = solver_statistics(scene)
                    assert (
                        result["edge_triangle_check_enabled"] and result["edge_triangle_intersection_count"] is not None
                    )
                    result["positions_and_velocities_finite"] = bool(
                        torch.isfinite(scene.finite_element.x).all()
                        & torch.isfinite(scene.finite_element.velocities).all()
                    )
                    x, tets = scene.finite_element.x, observer.tetrahedra
                    deformation = (x[tets[:, 1:]] - x[tets[:, :1]]).transpose(-1, -2) @ observer.inverse_rest
                    determinant = torch.linalg.det(deformation)
                    determinant_finite = bool(torch.isfinite(determinant).all())
                    result["minimum_det_F"] = determinant.min().item() if determinant_finite else None
                    result["ccd_maximum_iterations"] = scene.solver.max_accd_iters
                    gap = scene.solver.min_gap_ratio
                    result["minimum_reported_gap_ratio"] = gap if np.isfinite(gap) else None
                    return result

                info = measure("native_status_and_geometry_check", status)
                failed = (
                    not info["positions_and_velocities_finite"]
                    or info["edge_triangle_intersection_count"] > 0
                    or info["newton"] >= config.max_newton
                    or info["line_search_max"] >= 64
                    or info["minimum_det_F"] is None
                    or info["minimum_det_F"] <= 0
                )
                info.update(
                    time_s=step * config.dt, invalid_step=failed, pcg_cap_reached=info["pcg_max"] >= config.max_pcg
                )
                state = None
                if observe or failed:
                    if previous is None:
                        # Failure-only diagnostic: acceleration closure needs the real pre-step velocity.
                        # Do not manufacture it for an unsampled rejected state.
                        info["failure_diagnostic_velocity_unavailable"] = True
                    else:
                        state = measure("stress_and_contact_observer", lambda: observer.read_step(previous))
                        observer_calls += 1
                        metrics = measure("scalar_metrics_host_transfer", state.metrics)
                        info["sampled_stress_metrics"] = metrics
                        failed = failed or not metrics["all_fields_finite"] or metrics["minimum_det_F"] <= 0
                        info["invalid_step"] = failed
                if step in keyframes and state is not None:

                    def transfer():
                        return {
                            "time_s": step * config.dt,
                            "step": step - 1,
                            "positions_m": scene.finite_element.x.cpu().numpy(),
                            "velocities_m_per_s": scene.finite_element.velocities.cpu().numpy(),
                            "cauchy_pa": state.field.cauchy_pa.cpu().numpy(),
                            "det_F": state.field.determinant.cpu().numpy(),
                            "contact_forces_n": state.contact.fem_forces_n.cpu().numpy(),
                            "inertial_reaction_n": state.inertial_reaction_n.cpu().numpy(),
                            "active_global_ids": state.contact.global_vertex_ids.cpu().numpy(),
                            "active_global_forces_n": state.contact.global_forces_n.cpu().numpy(),
                            "active_global_positions_m": state.contact.global_positions_m[
                                state.contact.global_vertex_ids
                            ]
                            .cpu()
                            .numpy(),
                            "target_q": motion["q"][sample],
                            "actual_robot_com_world_m": scene.rigid_body.t.cpu().numpy(),
                            "actual_robot_material_quaternion_wxyz": scene.rigid_body.quat.cpu().numpy(),
                        }

                    payload = measure("full_field_host_transfer", transfer)
                    payloads.append(
                        {
                            "time_s": step * config.dt,
                            "cauchy_field_bytes": payload["cauchy_pa"].nbytes,
                            "all_exported_numeric_array_bytes": sum(
                                np.asarray(value).nbytes for value in payload.values()
                            ),
                        }
                    )
                    destination = args.output / "frames" / f"step{step - 1:06d}.npz"
                    if args.defer_field_io:
                        deferred_fields.append((destination, payload))
                    else:
                        begun = perf_counter()
                        np.savez_compressed(destination, **payload)
                        if timed:
                            stage_times["field_compression_and_IO"] = {
                                "wall_seconds": perf_counter() - begun,
                                "cuda_stream_span_ms": None,
                            }
                    observed.append(step * config.dt)
                rows.append(info)
                if timed:
                    timings.append({"time_s": step * config.dt, "stages": stage_times})
                begun = perf_counter()
                journal.write(json.dumps(info, allow_nan=False) + "\n")
                if timed:
                    stage_times["journal_IO"] = {"wall_seconds": perf_counter() - begun, "cuda_stream_span_ms": None}
                if failed:
                    failure = info
                    preserve_contact_failure(scene, args.output)
                    break
            completed = len(rows) == steps and failure is None
        finally:
            torch.cuda.synchronize()
            ended = perf_counter()
            elapsed = ended - total_started
            begun = perf_counter()
            for destination, payload in deferred_fields:
                np.savez_compressed(destination, **payload)
            post_window_field_io_seconds = perf_counter() - begun if deferred_fields else 0.0
            gate_stats = {"active": gate is not None, "attempts": None, "accepts": None, "last_ratio": None}
            if gate is not None:
                ratio = float(gate.last_ratio)
                gate_stats.update(
                    attempts=gate.attempts,
                    accepts=gate.accepts,
                    last_ratio=ratio if np.isfinite(ratio) else None,
                    last_ratio_is_finite=bool(np.isfinite(ratio)),
                )
            write_json(args.output / "timings.json", timings)
            summary = {}
            for name in sorted({name for row in timings for name in row["stages"]}):
                values = np.array([row["stages"][name]["wall_seconds"] for row in timings if name in row["stages"]])
                summary[name] = {
                    "calls": len(values),
                    "total_wall_seconds": float(values.sum()),
                    "median_wall_ms": float(np.median(values) * 1000),
                    "p95_wall_ms": float(np.quantile(values, 0.95) * 1000),
                }
            write_json(
                args.output / "result.json",
                {
                    "kind": "continuous-short-contact-window-segmented-profile",
                    "config": asdict(config),
                    "requested_steps": steps,
                    "executed_steps": len(rows),
                    "completed_requested_history": completed,
                    "setup_seconds": setup_seconds,
                    "elastic_observer_cold_activation_seconds": observer_activation_seconds,
                    "initialization_only_elastic_warmups": int(args.prewarm_elastic_observer),
                    "timing_mode": args.timing_mode,
                    "field_IO_deferred": args.defer_field_io,
                    "post_window_field_IO_seconds": post_window_field_io_seconds,
                    "total_intrusive_seconds": elapsed,
                    "window_intrusive_seconds": ended - window_started if window_started is not None else None,
                    "window_observations": len(timings),
                    "stage_summary": summary,
                    "stress_observer_calls": observer_calls,
                    "exported_keyframes_s": observed,
                    "keyframe_payloads": payloads,
                    "pcg_cap_steps": sum(row["pcg_cap_reached"] for row in rows),
                    "failure": failure,
                    "newton_aware_gate": gate_stats,
                    "reference_qualified": False,
                    "scope": "Segmented mode explicitly synchronizes each phase; CUDA stream spans include dispatch/idle intervals rather than kernel sums. Whole-window mode adds only boundary synchronization and records host-call spans without interpreting those spans as GPU latency. Both retain necessary native/geometry checks and sparse observation costs. Setup and optional one-time read-only elastic-observer activation are outside the contact window. Optional deferred compression/IO is measured after the window; complete spatial host transfer remains declared in-window. Continuous mechanics/contact history is unchanged, with stress/contact observation only at selected physical steps. No temporal stress peak or whole-history stress norm is computed. Accepted-window checkpoint overhead is absent; rejected states receive native dumps.",
                },
            )
    if not completed:
        raise RuntimeError("Short-window candidate failed; retained evidence is unqualified")


if __name__ == "__main__":
    main()
