"""Retain native full-field capped Franka histories at the same control times as QIPC."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.rollout import controller_step
from .artifacts import file_id, runtime_identity, write_json
from .common_metric import voigt_tensor
from .prescribed_motion import PrescribedMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dpr", "scr"), required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--precision", choices=("32", "64"), default="64")
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--tracking-damping", type=float, default=1e10)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--control-interval", type=float, default=0.01)
    parser.add_argument("--keyframes", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.steps > 0 and args.record_every > 0
    control_stride = round(args.control_interval / args.sim_dt)
    assert control_stride > 0 and abs(control_stride * args.sim_dt - args.control_interval) < 1e-12
    assert args.control_interval == 0.01
    key_steps = None
    if args.keyframes is not None:
        key_steps = {round(time / args.sim_dt) for time in args.keyframes}
        assert all(
            abs(round(time / args.sim_dt) * args.sim_dt - time) < 1e-12 and 0 < time <= args.steps * args.sim_dt
            for time in args.keyframes
        )
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    started = perf_counter()
    workload = FrankaEgg(
        1,
        level=1,
        stress=True,
        conditions=args.conditions,
        substeps=args.substeps,
        output_mode="full",
        stress_update_mode="manual" if key_steps is not None else "every_step",
        native_controller=False,
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        contact_model="pressure" if args.method == "dpr" else "intrinsic",
        intrinsic_load_partition="body_source_prior",
        timestep=args.sim_dt,
    )
    motion = (
        PrescribedMotion.configure(workload.robot, args.motion, args.sim_dt, args.tracking_damping)
        if args.motion is not None
        else None
    )
    if motion is not None:
        assert args.substeps == 1 and args.steps * motion.stride < len(motion.q)
    torch.cuda.synchronize()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    write_json(
        args.output / "start.json",
        {
            "genesis_import": gs.__file__,
            "identity": runtime_identity(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "conditions": file_id(args.conditions),
            "elastic_mesh": file_id(workload.link.stress_options.mesh),
            "method": args.method,
            "effective_options": workload.link.stress_options.model_dump(mode="json"),
            "dt": args.sim_dt,
            "substeps": args.substeps,
            "control_interval_s": args.control_interval,
            "control_stride_scene_steps": control_stride,
            "precision": args.precision,
            "setup_seconds": perf_counter() - started,
            "protocol": "shared prescribed motion" if motion is not None else "original capped-control schedule",
            "prescribed_motion": motion.identity() if motion is not None else None,
            "stress_frame": "authored link frame",
            "field_measure": "linear elastic tensor, Voigt xx yy zz xy xz yz; last physical substep per scene step. One-substep runs retain every physical-step peak state; a multi-substep final field cannot be called its interior peak field.",
            "scope": "intrusive full-field/force history; excluded from primary timing",
            "full_spatial_keyframes_s": args.keyframes,
            "unsampled_stress_policy": "hold previous snapshot; no recovery/export" if key_steps is not None else None,
            "rigid_link_names": [{"id": link.idx, "name": link.name} for link in workload.scene.rigid_solver.links],
        },
    )
    with np.load(workload.link.stress_options.mesh) as mesh:
        np.savez_compressed(args.output / "rest.npz", **{key: mesh[key] for key in mesh.files})
    invalid_initial = qd_to_numpy(entry.state.invalid_steps, transpose=True).copy()
    completed = False
    executed = 0
    invalid = 0
    invalid_mechanics = 0
    physics_steps_returned = 0
    observed_fields = 0
    peak = -np.inf
    principal_peak = -np.inf
    control_force_sum = np.zeros(3)
    control_moment_sum = np.zeros(3)
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(args.steps):
                observe = key_steps is None or step + 1 in key_steps
                if motion is not None:
                    phase = motion.advance(
                        workload, step, True, update_stress=observe if key_steps is not None else None
                    )
                elif control_stride == 1 and key_steps is None:
                    phase = controller_step(workload, False)
                else:
                    if step % control_stride == 0:
                        phase = controller_step(workload, False, advance=False)
                    if key_steps is not None and observe:
                        workload.link.request_stress_update()
                    workload.scene.step()
                physics_steps_returned += 1
                if key_steps is not None:
                    workload.scene.rigid_solver.check_errno()
                    finite = bool(
                        torch.isfinite(workload.robot.get_qpos()).all()
                        & torch.isfinite(workload.robot.get_dofs_velocity()).all()
                        & torch.isfinite(workload.egg.get_pos()).all()
                        & torch.isfinite(workload.egg.get_vel()).all()
                    )
                    if not observe or not finite:
                        invalid_mechanics += int(not finite)
                        journal.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "time_s": (step + 1) * args.sim_dt,
                                    "stress_sampled": False,
                                    "phase": tensor_to_array(phase)[0].item(),
                                    "reset_epoch": workload.reset_count,
                                    "mechanical_state_finite": finite,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        executed += 1
                        if not finite:
                            break
                        continue
                stress, vm = workload.link.get_stress_field(copy=False)
                observed_fields += 1
                stress, vm = tensor_to_array(stress)[0], tensor_to_array(vm)[0]
                contacts = workload.egg.get_contacts(is_padded=False)
                contacts = {key: tensor_to_array(value)[0] for key, value in contacts.items()}
                valid = contacts["valid_mask"]
                egg_is_a = contacts["link_a"] == workload.link.idx
                forces = np.where(egg_is_a[:, None], contacts["force_a"], contacts["force_b"])[valid]
                points = contacts["position"][valid]
                normals = contacts["normal"][valid]
                normal_force = np.einsum("ci,ci->c", forces, normals)
                tangent_force = forces - normal_force[:, None] * normals
                friction_utilization = np.full(len(forces), np.nan)
                normal_admitted = np.abs(normal_force) > 1e-9
                friction_utilization[normal_admitted] = np.linalg.norm(tangent_force[normal_admitted], axis=1) / (
                    contacts["friction"][valid][normal_admitted] * np.abs(normal_force[normal_admitted])
                )
                penetration = contacts["penetration"][valid]
                control_force_sum += forces.sum(axis=0)
                control_moment_sum += np.cross(points, forces).sum(axis=0)
                finite = bool(np.isfinite(stress).all() and np.isfinite(vm).all())
                principal = np.linalg.eigvalsh(voigt_tensor(stress))[..., -1].max().item() if finite else None
                count = qd_to_numpy(entry.state.invalid_steps, transpose=True) - invalid_initial
                invalid = count.sum().item()
                q = tensor_to_array(workload.robot.get_qpos())[0]
                position = tensor_to_array(workload.egg.get_pos())[0]
                quaternion = tensor_to_array(workload.egg.get_quat())[0]
                residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
                rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
                payload = dict(
                    time_s=(step + 1) * args.sim_dt,
                    step=step,
                    stress_voigt_link_pa=stress,
                    vm_pa=vm,
                    q=q,
                    link_position_m=position,
                    link_quaternion_wxyz=quaternion,
                    contact_positions_world_m=points,
                    contact_forces_world_n=forces,
                    contact_normals=contacts["normal"][valid],
                    contact_penetration_m=penetration,
                    contact_friction_utilization=friction_utilization,
                    contact_counterpart_link=np.where(egg_is_a, contacts["link_b"], contacts["link_a"])[valid],
                    contact_friction=contacts["friction"][valid],
                    contact_source_geom_a=contacts["geom_a"][valid],
                    contact_source_geom_b=contacts["geom_b"][valid],
                )
                if key_steps is not None or (step + 1) % args.record_every == 0 or step == args.steps - 1 or not finite:
                    np.savez_compressed(frames / f"step{step:06d}.npz", **payload)
                if key_steps is None and finite and vm.max() > peak:
                    peak = vm.max().item()
                    np.savez_compressed(args.output / "raw-vm-peak-state.npz", **payload)
                if key_steps is None and finite and principal > principal_peak:
                    principal_peak = principal
                    np.savez_compressed(args.output / "raw-principal-peak-state.npz", **payload)
                row = {
                    "step": step,
                    "time_s": (step + 1) * args.sim_dt,
                    "stress_sampled": True,
                    "reset_epoch": workload.reset_count,
                    "phase": tensor_to_array(phase)[0].item(),
                    "position_m": position.tolist(),
                    "quaternion_wxyz": quaternion.tolist(),
                    "q": q.tolist(),
                    "all_fields_finite": finite,
                    "raw_maximum_principal_pa": principal,
                    "invalid_environment_steps": invalid,
                    "raw_vm_peak_last_physical_substep_pa": vm.max().item() if finite else None,
                    "raw_vm_peak_over_scene_substeps_pa": tensor_to_array(workload.link.get_max_stress())[0].item()
                    if finite
                    else None,
                    "complete_force_residual_l2_n": np.linalg.norm(residual).item(),
                    "complete_force_relative_residual": (np.linalg.norm(residual) / np.linalg.norm(rhs)).item()
                    if np.linalg.norm(rhs) > 1e-12
                    else None,
                    "contact_resultant_n": forces.sum(axis=0).tolist(),
                    "contact_moment_world_origin_nm": np.cross(points, forces).sum(axis=0).tolist(),
                    "contact_count": len(forces),
                    "maximum_reported_contact_penetration_m": penetration.max().item() if len(penetration) else None,
                    "maximum_contact_friction_utilization": friction_utilization[normal_admitted].max().item()
                    if normal_admitted.any()
                    else None,
                    "friction_utilization_normal_force_floor_n": 1e-9,
                }
                if motion is not None:
                    row.update(motion.metrics(workload.robot, step))
                elif key_steps is None and args.substeps == 1 and (step + 1) % control_stride == 0:
                    row.update(
                        contact_average_force_over_control_interval_n=(control_force_sum / control_stride).tolist(),
                        contact_average_moment_world_origin_over_control_interval_nm=(
                            control_moment_sum / control_stride
                        ).tolist(),
                        control_force_scope="Average of actual end-of-physical-step contact forces and physical moments; never final point times accumulated force",
                    )
                    control_force_sum.fill(0)
                    control_moment_sum.fill(0)
                executed += 1
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                if not finite:
                    break
            completed = executed == args.steps and finite
        finally:
            exception_type, exception, _ = sys.exc_info()
            invalid = (qd_to_numpy(entry.state.invalid_steps, transpose=True) - invalid_initial).sum().item()
            write_json(
                args.output / "result.json",
                {
                    "kind": "native-prescribed-full-field-history"
                    if motion is not None
                    else "native-capped-full-field-history",
                    "method": args.method,
                    "requested_control_ticks": args.steps,
                    "scene_step_dt_s": args.sim_dt,
                    "executed_control_ticks": executed,
                    "physics_steps_returned": physics_steps_returned,
                    "observed_fields": observed_fields,
                    "completed_requested_history": completed,
                    "invalid_environment_steps": invalid,
                    "stress_invalid_counter_scope": "observed recovery substeps"
                    if key_steps is not None
                    else "all physical substeps",
                    "invalid_mechanical_steps": invalid_mechanics,
                    "full_spatial_keyframes_s": args.keyframes,
                    "python_exception": {"type": exception_type.__name__, "message": str(exception)}
                    if exception_type is not None
                    else None,
                    "intrusive_seconds": perf_counter() - started,
                    "reference_accuracy_qualified": False,
                },
            )
    workload.scene.rigid_solver.check_errno()
    if not completed:
        raise RuntimeError("Native candidate stopped with retained invalid full field")


if __name__ == "__main__":
    main()
