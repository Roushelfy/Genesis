"""Measure normal recovery with warmed resident inputs and separate audits."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.hertz import StressHertz, StressHertzState
from genesis.engine.solvers.rigid.stress.hertz_recovery import (
    StressHertzFrames,
    StressHertzRecovery,
    StressHertzRecoveryState,
)
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array, configure


@qd.kernel
def inputs(
    step: int,
    nominal: qd.Tensor,
    state: StressHertzState,
    frames: StressHertzFrames,
    info: StressInfo,
    n_envs: int,
    moving_centers: qd.template(),
):
    for i_p in range(state.force.shape[0]):
        i_b = i_p % n_envs
        force = nominal[i_p] * (1 + 0.2 * qd.sin(step * 0.37 + i_b * 0.11))
        if (step + i_b) % 13 == 0:
            force = 0.0
        state.force[i_p] = force
        if qd.static(moving_centers):
            anchor = frames.anchor[i_p]
            bary = qd.Vector(
                [0.3 + 0.08 * qd.sin(0.41 * step + 0.13 * i_b), 0.3 + 0.08 * qd.cos(0.17 * step + 0.11 * i_p), 0.0]
            )
            bary[2] = 1 - bary[0] - bary[1]
            center = qd.Vector.zero(gs.qd_float, 3)
            for i in qd.static(range(3)):
                i_n = info.surface_nodes[anchor, i]
                center += bary[i] * info.vertices[i_n]
            frames.center[i_p] = center
            angle = 0.17 * (4 * (i_p // n_envs) + i_b % 4) + 0.037 * step
            c, s = qd.cos(angle), qd.sin(angle)
            state.curvature[i_p] = qd.Matrix(
                [[50 * c * c + 80 * s * s, -30 * c * s], [-30 * c * s, 50 * s * s + 80 * c * c]]
            )


@qd.kernel
def capture_inputs(step: int, centers: qd.Tensor, axes: qd.Tensor, frames: StressHertzFrames, hertz: StressHertzState):
    for i_p in range(hertz.force.shape[0]):
        centers[step, i_p], axes[step, i_p] = frames.center[i_p], hertz.axes[i_p]


@qd.kernel
def capture(
    step: int,
    statuses: qd.Tensor,
    loaded: qd.Tensor,
    vm: qd.Tensor,
    tensile: qd.Tensor,
    wrench: qd.Tensor,
    elastic: qd.Tensor,
    stress: StressState,
    hertz: StressHertzState,
    state: StressHertzRecoveryState,
):
    for i_b in range(state.status.shape[0]):
        statuses[step, i_b] = state.status[i_b]
        loaded[step, i_b] = False
        for i_r in range(hertz.force.shape[0] // state.status.shape[0]):
            i_p = i_r * state.status.shape[0] + i_b
            if hertz.force[i_p] > 0:
                loaded[step, i_b] = True
        vm[step, i_b], tensile[step, i_b] = stress.peak[i_b], state.tensile_peak[i_b]
        wrench[step, i_b] = state.wrench_error[i_b]
        elastic[step, i_b] = stress.valid[i_b]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--far-field", action="store_true")
    cli.add_argument("--moving-probes", action="store_true")
    cli.add_argument("--moving-centers", action="store_true")
    cli.add_argument("--quadrature", type=int, default=4)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "control_mesh": file_id(args.control / "mesh.npz"),
            "control_fields": file_id(args.control / "fields.npz"),
            "far_field_four_moment_sources": args.far_field,
            "moving_probes": args.moving_probes,
            "moving_centers_and_rotating_curvature": args.moving_centers,
            "boundary_quadrature": args.quadrature,
            "scope": "Supplied flat normal regions on the finite control box, independently phased normal loads and release. Inclusive resident input/Hertz/scatter/solve/combined max/full module work. Optional resident center/curvature motion and source-relative material probes are explicit. No collision association, friction, moving rigid scene or general physical qualification.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    started = perf_counter()
    model = StressModel(RigidStressOptions(mesh=args.control / "mesh.npz", method="auto"))
    surface = StressSurface(args.quadrature, model.info)
    qd.sync()
    preparation = perf_counter() - started
    with np.load(args.control / "fields.npz") as data:
        nv = len(data["volume_weights_m3"])
        centers, frames, anchors = data["centers_m"], data["frames"], data["anchors"]
        positions, elements, barycentric = (
            data["positions_m"][nv:],
            data["query_elements"][nv:],
            data["query_barycentric"][nv:],
        )
    if args.moving_probes:
        positions, elements, barycentric = positions[:0], elements[:0], barycentric[:0]
    angles = np.arange(8) * 0.17
    rotations = np.stack(
        [np.stack([np.cos(angles), -np.sin(angles)], axis=1), np.stack([np.sin(angles), np.cos(angles)], axis=1)],
        axis=1,
    )
    curvature = rotations @ np.diag([50.0, 80.0]) @ rotations.transpose(0, 2, 1)
    force = np.array([0.2, 0.12, 0.18, 0, 0, 0.05, 0.11, 0])
    rows = []
    for batch in [1, 64, 256, 1024]:
        indices = np.r_[np.arange(batch) % 4, 4 + np.arange(batch) % 4]
        law = StressHertz(2 * batch)
        law.state.modulus.fill(1e10 / 0.91)
        law.state.curvature.from_numpy(curvature[indices])
        nominal = V(dtype=gs.qd_float, shape=(2 * batch,))
        nominal.from_numpy(force[indices])
        omega = V_VEC(3, dtype=gs.qd_float, shape=(batch,))
        omega.fill(0)
        for mode in ["max", "full"]:
            module = StressHertzRecovery(
                batch, 2, len(positions), mode, far_field=args.far_field, moving_probes=args.moving_probes
            )
            if args.moving_probes:
                module.frames.center.from_numpy(centers[indices])
                module.frames.basis.from_numpy(frames[indices])
                module.frames.anchor.from_numpy(anchors[indices].astype(gs.np_int))
            else:
                configure(module, centers[indices], frames[indices], anchors[indices], positions, elements, barycentric)
            state = model.create_state(batch, mode)
            errno = V(dtype=gs.qd_int, shape=(batch,))
            errno.fill(0)
            for step in range(4):
                kernel_begin_step(state)
                inputs(step, nominal, law.state, module.frames, model.info, batch, args.moving_centers)
                law.update()
                module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
            qd.sync()
            elapsed = []
            for repeat in range(3):
                began = perf_counter()
                for frame in range(8):
                    step = repeat * 8 + frame
                    kernel_begin_step(state)
                    inputs(step, nominal, law.state, module.frames, model.info, batch, args.moving_centers)
                    law.update()
                    module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
                qd.sync()
                elapsed.append((perf_counter() - began) / 8)
            # errno latches failures throughout all timed frames, even when a later normal update recovers.
            assert not np.any(array(errno)), (batch, mode, array(errno))
            status, loaded, vm, tensile, wrench, elastic = (
                V(dtype=dtype, shape=(24, batch))
                for dtype in [gs.qd_int, gs.qd_bool, gs.qd_float, gs.qd_float, gs.qd_float, gs.qd_bool]
            )
            source_centers = V_VEC(3, dtype=gs.qd_float, shape=(24, 2 * batch))
            source_axes = V_VEC(2, dtype=gs.qd_float, shape=(24, 2 * batch))
            for step in range(24):
                kernel_begin_step(state)
                inputs(step, nominal, law.state, module.frames, model.info, batch, args.moving_centers)
                law.update()
                module.recover(omega, law.state, state, model, surface.info, 1e-12, errno)
                capture(step, status, loaded, vm, tensile, wrench, elastic, state, law.state, module.state)
                capture_inputs(step, source_centers, source_axes, module.frames, law.state)
            qd.sync()
            statuses, loads, solves = array(status), array(loaded), array(elastic)
            vms, tensions, wrenches = array(vm), array(tensile), array(wrench)
            np.savez_compressed(
                args.output / f"{mode}-b{batch}-audit.npz",
                status=statuses,
                loaded=loads,
                elastic_valid=solves,
                vm_pa=vms,
                tensile_pa=tensions,
                wrench_error=wrenches,
                centers_m=array(source_centers),
                axes_m=array(source_axes),
            )
            assert np.all(statuses <= 1) and np.all(solves) and not np.any(array(errno))
            assert np.isfinite(vms).all() and np.isfinite(tensions).all() and np.max(wrenches) < 1e-4
            diagnostics = {}
            for phase in ["input_and_boundary", "regular_solve", "combined_field"]:
                samples = []
                for repeat in range(3):
                    began = perf_counter()
                    for _ in range(8):
                        if phase == "input_and_boundary":
                            inputs(1, nominal, law.state, module.frames, model.info, batch, args.moving_centers)
                            law.update()
                            module.scatter(law.state, state, model.info, surface.info, 0.3, 1e-12, errno)
                        elif phase == "regular_solve":
                            model.recover(omega, state, surface_load=True, compute_peak=False)
                        else:
                            module.evaluate(law.state, state, model, 1e-12, errno)
                    qd.sync()
                    samples.append((perf_counter() - began) / 8)
                diagnostics[phase] = {"seconds": samples, "median_seconds": float(np.median(samples))}
            assert not np.any(array(errno))
            seconds = float(np.median(elapsed))
            row = {
                "batch": batch,
                "mode": mode,
                "seconds_per_update": elapsed,
                "median_seconds": seconds,
                "raw_environment_updates_per_second": batch / seconds,
                "audit_environment_frames": int(statuses.size),
                "admitted_loaded_audit_frames": int(loads.sum()),
                "admitted_all_audit_frames": int(statuses.size),
                "maximum_wrench_error": float(wrenches.max()),
                "exclusive_fixed_input_phase_diagnostics": diagnostics,
                "complete_tensor_output_bytes": (
                    model.info.elements.shape[0] * 4 * 7 + (len(positions) + (178 if args.moving_probes else 0)) * 6
                )
                * batch
                * 8
                if mode == "full"
                else 0,
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "shared_preparation_seconds": preparation,
            "shared_build_timings": asdict(model.build_timings),
            "additional_query_count": len(positions),
            "moving_probe_count_per_region": 89 if args.moving_probes else 0,
            "element_count": model.info.elements.shape[0],
            "phase_times_are_separate_fixed_input_diagnostics_not_additive": True,
            "moving_rigid_scene_qualified": False,
            "general_contact_friction_physics_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
