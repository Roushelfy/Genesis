"""Retain PR22 full-field capped histories, with optional existing read-only solver/contact diagnostics."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison import diagnostics
from ..shell_comparison.application import ShellEgg
from ..shell_comparison.rollout import controller_step, make_parser
from .artifacts import file_id, runtime_identity, write_json
from .prescribed_motion import PrescribedMotion


def main():
    parser = make_parser()
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--tracking-damping", type=float, default=1e10)
    parser.add_argument("--control-interval", type=float, default=0.01)
    parser.add_argument("--substep-peaks", action="store_true")
    parser.add_argument("--keyframes", type=float, nargs="+")
    args = parser.parse_args()
    assert args.model == "shell" and args.envs == 1 and args.output_mode == "full"
    key_steps = None
    if args.keyframes is not None:
        assert not args.substep_peaks
        key_steps = {round(time / args.sim_dt) for time in args.keyframes}
        assert all(
            0 < time <= args.steps * args.sim_dt and abs(round(time / args.sim_dt) * args.sim_dt - time) < 1e-12
            for time in args.keyframes
        )
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    with np.load(args.conditions) as bank:
        coefficient = 0.6 * bank["friction"][0, 0].item()
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    started = perf_counter()
    workload = ShellEgg(args)
    assert np.isclose(coefficient, args.coupling_friction, rtol=0, atol=1e-15)
    motion = (
        PrescribedMotion.configure(workload.robot, args.motion, args.sim_dt, args.tracking_damping)
        if args.motion is not None
        else None
    )
    if motion is not None:
        assert args.substeps == 1 and args.steps * motion.stride < len(motion.q)
    else:
        control_stride = round(args.control_interval / args.sim_dt)
        assert args.control_interval == 0.01 and control_stride > 0
        assert abs(control_stride * args.sim_dt - args.control_interval) < 1e-12
    torch.cuda.synchronize()
    faces = tensor_to_array(workload.egg.get_faces())[0]
    face_mass = workload.egg.topology.faces_mass
    vertex_mass = np.bincount(faces.reshape(-1), weights=np.repeat(face_mass / 3, 3), minlength=workload.egg.n_verts)
    thickness = qd_to_numpy(workload.scene.sim.shell_solver.shell_state.faces_thickness, transpose=True)[0]
    np.savez_compressed(
        args.output / "rest.npz",
        world_initial_vertices_m=tensor_to_array(workload.initial_vertices)[0],
        faces=faces,
        vertices_m=workload.egg.init_verts - np.array([0.65, 0, 0.031]),
        vertex_masses_kg=vertex_mass,
        face_thickness_m=thickness,
    )
    write_json(
        args.output / "start.json",
        {
            "genesis_import": gs.__file__,
            "identity": runtime_identity(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "condition_bank": file_id(args.conditions),
            "midpoint_mesh": file_id(args.inputs / f"egg-mid-level{args.level}.obj"),
            "method": "shell",
            "dt": args.sim_dt,
            "protocol": "shared prescribed motion" if motion is not None else "original capped-control schedule",
            "prescribed_motion": motion.identity() if motion is not None else None,
            "substeps": args.substeps,
            "precision": args.precision,
            "pcg_cap": args.iterations,
            "pcg_threshold": args.tolerance,
            "coarse_interval": args.coarse_interval,
            "coarse_patches": args.patches,
            "damping": args.damping,
            "coupling_friction": args.coupling_friction,
            "coupling_softness_m": args.coupling_softness,
            "influence_0_1_cutoff_beyond_half_thickness_m": (args.coupling_softness * np.log(10)).item(),
            "setup_seconds": perf_counter() - started,
            "source_scope": "PR22 plus existing read-only diagnostics"
            if args.instrumented
            else "original PR22 plus full-field observer",
            "stress_frame": "world Cauchy observer, nominal bottom/middle/top fibers per face",
            "scope": "intrusive complete capped history, excluded from primary timing",
            "full_spatial_keyframes_s": args.keyframes,
            "nominal_thickness_mass_kg": (workload.egg.topology.faces_rest_area.sum() * 2000 * 0.0005).item(),
            "actual_face_mass_kg": face_mass.sum().item(),
            "wall_input": workload.wall_input.description() if workload.wall_input is not None else None,
            "fracture_enabled": workload.scene.sim.shell_solver._static_config.has_fracture,
            "plasticity_enabled": workload.scene.sim.shell_solver._static_config.has_plasticity,
        },
    )
    completed = False
    physics_steps_returned = 0
    observed_fields = 0
    count = 0
    nonfinite = False
    peak = -np.inf
    principal_peak = -np.inf
    physical_observations = 0
    invalid_physical_observations = 0
    micro_journal = None
    if args.substep_peaks:
        assert args.instrumented and motion is None
        micro_journal = (args.output / "physical-step-peaks.jsonl").open("w", buffering=1)

        def observe_physical_step(solver):
            nonlocal peak, principal_peak, physical_observations, invalid_physical_observations
            physical_observations += 1
            workload.observation.update(solver)
            tensor = workload.observation.tensors[:, 0].reshape(-1, 3, 3, 3)
            finite = torch.isfinite(tensor).all().item()
            vm_peak = workload.observation.von_mises[:, 0].max().item() if finite else None
            tensile_peak = torch.linalg.eigvalsh(tensor)[..., -1].max().item() if finite else None
            time = physical_observations * args.sim_dt / args.substeps
            micro_journal.write(
                json.dumps(
                    {
                        "physical_step": physical_observations - 1,
                        "time_s": time,
                        "all_fields_finite": bool(finite),
                        "raw_vm_peak_pa": vm_peak,
                        "raw_maximum_principal_pa": tensile_peak,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            invalid_physical_observations += not finite
            if not finite or vm_peak > peak or tensile_peak > principal_peak:
                payload = {
                    "time_s": time,
                    "physical_step": physical_observations - 1,
                    "cauchy_world_pa": tensor_to_array(tensor).copy(),
                    "vm_pa": tensor_to_array(workload.observation.von_mises)[:, 0].reshape(-1, 3),
                    "vertices_world_m": tensor_to_array(workload.egg.get_verts_pos())[0],
                    "q": tensor_to_array(workload.robot.get_qpos())[0],
                }
                if not finite:
                    np.savez_compressed(args.output / "invalid-physical-step-state.npz", **payload)
                else:
                    if vm_peak > peak:
                        peak = vm_peak
                        np.savez_compressed(args.output / "raw-vm-peak-state.npz", **payload)
                    if tensile_peak > principal_peak:
                        principal_peak = tensile_peak
                        np.savez_compressed(args.output / "raw-principal-peak-state.npz", **payload)

        diagnostics.SUBSTEP_OBSERVER = observe_physical_step
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(args.steps):
                observe = key_steps is None or step + 1 in key_steps
                if args.instrumented:
                    diagnostics.ACTIVE.clear()
                if motion is not None:
                    phase = motion.advance(workload, step, False)
                elif control_stride == 1:
                    phase = controller_step(workload, True, observe_stress=key_steps is None)
                else:
                    if step % control_stride == 0:
                        phase = controller_step(workload, True, advance=False)
                    workload.scene.step()
                physics_steps_returned += 1
                if observe and (motion is not None or control_stride != 1 or key_steps is not None):
                    workload.observation.update(workload.scene.sim.shell_solver)
                if key_steps is not None:
                    workload.scene.rigid_solver.check_errno()
                    finite_geometry = bool(
                        torch.isfinite(workload.egg.get_verts_pos()).all()
                        & torch.isfinite(workload.robot.get_qpos()).all()
                    )
                    if not observe or not finite_geometry:
                        row = {
                            "step": step,
                            "time_s": (step + 1) * args.sim_dt,
                            "stress_sampled": False,
                            "mechanical_geometry_finite": finite_geometry,
                            "phase": tensor_to_array(phase)[0].item(),
                            "reset_epoch": workload.reset_count,
                        }
                        if args.instrumented:
                            row["cap_exhausted_substeps"] = tensor_to_array(diagnostics.ACTIVE.caps)[0].item()
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
                        if not finite_geometry:
                            nonfinite = True
                            np.savez_compressed(
                                args.output / "invalid-mechanics-state.npz",
                                time_s=(step + 1) * args.sim_dt,
                                vertices_world_m=tensor_to_array(workload.egg.get_verts_pos())[0],
                                q=tensor_to_array(workload.robot.get_qpos())[0],
                            )
                            break
                        continue
                tensor = tensor_to_array(workload.observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
                observed_fields += 1
                vm = tensor_to_array(workload.observation.von_mises)[:, 0].reshape(-1, 3)
                vertices = tensor_to_array(workload.egg.get_verts_pos())[0]
                q = tensor_to_array(workload.robot.get_qpos())[0]
                finite = bool(np.isfinite(tensor).all() and np.isfinite(vertices).all() and np.isfinite(q).all())
                payload = {
                    "time_s": (step + 1) * args.sim_dt,
                    "step": step,
                    "cauchy_world_pa": tensor,
                    "vm_pa": vm,
                    "vertices_world_m": vertices,
                    "q": q,
                }
                if args.instrumented:
                    payload["contact_impulses_ns"] = tensor_to_array(diagnostics.ACTIVE.impulses)[:, 0]
                    payload["contact_positions_world_m"] = tensor_to_array(diagnostics.ACTIVE.points)[:, 0]
                    payload["contact_moment_impulses_world_origin_nms"] = tensor_to_array(diagnostics.ACTIVE.moments)[
                        :, 0
                    ]
                    payload["contact_impulse_kinetic_work_j"] = tensor_to_array(diagnostics.ACTIVE.work)[:, 0]
                if (
                    key_steps is not None
                    or (step + 1) % max(args.record_every, 1) == 0
                    or step == args.steps - 1
                    or not finite
                ):
                    np.savez_compressed(frames / f"step{step:06d}.npz", **payload)
                row = {
                    "step": step,
                    "stress_sampled": True,
                    "time_s": (step + 1) * args.sim_dt,
                    "phase": tensor_to_array(phase)[0].item(),
                    "q": q.tolist(),
                    "position_mean_m": vertices.mean(axis=0).tolist(),
                    "minimum_vertex_height_m": vertices[:, 2].min().item() if finite else None,
                    "all_fields_finite": finite,
                    "raw_vm_peak_pa": vm.max().item() if finite else None,
                    "raw_maximum_principal_pa": np.linalg.eigvalsh(tensor)[..., -1].max().item() if finite else None,
                }
                if args.instrumented:
                    impulse = payload["contact_impulses_ns"]
                    row.update(
                        iteration_counts=tensor_to_array(diagnostics.ACTIVE.counts)[0].item(),
                        cap_exhausted_substeps=tensor_to_array(diagnostics.ACTIVE.caps)[0].item(),
                        actual_residuals=tensor_to_array(diagnostics.ACTIVE.residuals)[0].tolist(),
                        contact_impulse_over_scene_step_ns=impulse.sum(axis=0).tolist(),
                        contact_average_force_over_scene_step_n=(impulse.sum(axis=0) / args.sim_dt).tolist(),
                        contact_interval_s=args.sim_dt,
                        contact_average_force_scope="Impulse average over this scene step's physical substeps",
                        contact_average_moment_world_origin_over_scene_step_nm=(
                            payload["contact_moment_impulses_world_origin_nms"].sum(axis=0) / args.sim_dt
                        ).tolist(),
                        contact_impulse_kinetic_work_j=payload["contact_impulse_kinetic_work_j"].sum().item(),
                    )
                if motion is not None:
                    row.update(motion.metrics(workload.robot, step))
                if key_steps is None and finite and row["raw_vm_peak_pa"] > peak:
                    peak = row["raw_vm_peak_pa"]
                    np.savez_compressed(args.output / "raw-vm-peak-state.npz", **payload)
                if key_steps is None and finite and row["raw_maximum_principal_pa"] > principal_peak:
                    principal_peak = row["raw_maximum_principal_pa"]
                    np.savez_compressed(args.output / "raw-principal-peak-state.npz", **payload)
                count += 1
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                if not finite:
                    nonfinite = True
                    break
            completed = count == args.steps and not nonfinite
            if args.substep_peaks:
                completed = (
                    completed and physical_observations == count * args.substeps and not invalid_physical_observations
                )
        finally:
            exception_type, exception, _ = sys.exc_info()
            if args.substep_peaks:
                diagnostics.SUBSTEP_OBSERVER = None
                micro_journal.close()
            write_json(
                args.output / "result.json",
                {
                    "kind": "PR22-prescribed-full-field-history"
                    if motion is not None
                    else "PR22-capped-full-field-history",
                    "scene_step_dt_s": args.sim_dt,
                    "full_spatial_keyframes_s": args.keyframes,
                    "requested_control_ticks": args.steps,
                    "executed_control_ticks": count,
                    "physics_steps_returned": physics_steps_returned,
                    "observed_fields": observed_fields,
                    "python_exception": {"type": exception_type.__name__, "message": str(exception)}
                    if exception_type is not None
                    else None,
                    "completed_requested_history": completed,
                    "nonfinite_output": nonfinite,
                    "intrusive_seconds": perf_counter() - started,
                    "reference_accuracy_qualified": False,
                    "physical_peak_observations": physical_observations,
                    "invalid_physical_peak_observations": invalid_physical_observations,
                },
            )
    workload.scene.rigid_solver.check_errno()
    if not completed:
        raise RuntimeError("Shell history stopped with retained nonfinite state")


if __name__ == "__main__":
    main()
