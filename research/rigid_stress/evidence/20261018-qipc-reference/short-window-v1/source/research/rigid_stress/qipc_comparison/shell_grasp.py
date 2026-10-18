"""Retain PR22 full-field capped histories, with optional existing read-only solver/contact diagnostics."""

import json
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
    args = parser.parse_args()
    assert args.model == "shell" and args.envs == 1 and args.output_mode == "full"
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
            "nominal_thickness_mass_kg": (workload.egg.topology.faces_rest_area.sum() * 2000 * 0.0005).item(),
            "actual_face_mass_kg": face_mass.sum().item(),
            "wall_input": workload.wall_input.description() if workload.wall_input is not None else None,
            "fracture_enabled": workload.scene.sim.shell_solver._static_config.has_fracture,
            "plasticity_enabled": workload.scene.sim.shell_solver._static_config.has_plasticity,
        },
    )
    completed = False
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
                if args.instrumented:
                    diagnostics.ACTIVE.clear()
                if motion is not None:
                    phase = motion.advance(workload, step, False)
                elif control_stride == 1:
                    phase = controller_step(workload, True)
                else:
                    if step % control_stride == 0:
                        phase = controller_step(workload, True, advance=False)
                    workload.scene.step()
                    workload.observation.update(workload.scene.sim.shell_solver)
                if motion is not None:
                    workload.observation.update(workload.scene.sim.shell_solver)
                tensor = tensor_to_array(workload.observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
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
                if (step + 1) % max(args.record_every, 1) == 0 or step == args.steps - 1 or not finite:
                    np.savez_compressed(frames / f"step{step:06d}.npz", **payload)
                row = {
                    "step": step,
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
                        contact_impulse_over_control_interval_ns=impulse.sum(axis=0).tolist(),
                        contact_average_force_over_control_interval_n=(impulse.sum(axis=0) / args.sim_dt).tolist(),
                        contact_average_force_scope="Accumulated impulses over all substeps; cannot be labeled an instantaneous final-substep force",
                        contact_average_moment_world_origin_over_interval_nm=(
                            payload["contact_moment_impulses_world_origin_nms"].sum(axis=0) / args.sim_dt
                        ).tolist(),
                        contact_impulse_kinetic_work_j=payload["contact_impulse_kinetic_work_j"].sum().item(),
                    )
                if motion is not None:
                    row.update(motion.metrics(workload.robot, step))
                if finite and row["raw_vm_peak_pa"] > peak:
                    peak = row["raw_vm_peak_pa"]
                    np.savez_compressed(args.output / "raw-vm-peak-state.npz", **payload)
                if finite and row["raw_maximum_principal_pa"] > principal_peak:
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
                    "requested_control_ticks": args.steps,
                    "executed_control_ticks": count,
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
