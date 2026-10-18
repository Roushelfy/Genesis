"""Retain native full-field capped Franka histories at the same control times as QIPC."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.rollout import controller_step
from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dpr", "scr"), required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--precision", choices=("32", "64"), default="64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
        native_controller=False,
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        contact_model="pressure" if args.method == "dpr" else "intrinsic",
        intrinsic_load_partition="body_source_prior",
    )
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
            "dt": 0.01,
            "substeps": args.substeps,
            "precision": args.precision,
            "setup_seconds": perf_counter() - started,
            "protocol": "original capped-control schedule",
            "stress_frame": "authored link frame",
            "field_measure": "linear elastic tensor, Voigt xx yy zz xy xz yz; last physical substep per control tick",
            "scope": "intrusive full-field/force history; excluded from primary timing",
        },
    )
    with np.load(workload.link.stress_options.mesh) as mesh:
        np.savez_compressed(args.output / "rest.npz", **{key: mesh[key] for key in mesh.files})
    invalid_initial = qd_to_numpy(entry.state.invalid_steps, transpose=True).copy()
    completed = False
    executed = 0
    invalid = 0
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(args.steps):
                phase = controller_step(workload, False)
                stress, vm = workload.link.get_stress_field(copy=False)
                stress, vm = tensor_to_array(stress)[0], tensor_to_array(vm)[0]
                contacts = workload.egg.get_contacts(is_padded=False)
                contacts = {key: tensor_to_array(value)[0] for key, value in contacts.items()}
                valid = contacts["valid_mask"]
                egg_is_a = contacts["link_a"] == workload.link.idx
                forces = np.where(egg_is_a[:, None], contacts["force_a"], contacts["force_b"])[valid]
                points = contacts["position"][valid]
                finite = bool(np.isfinite(stress).all() and np.isfinite(vm).all())
                count = qd_to_numpy(entry.state.invalid_steps, transpose=True) - invalid_initial
                invalid = count.sum().item()
                q = tensor_to_array(workload.robot.get_qpos())[0]
                position = tensor_to_array(workload.egg.get_pos())[0]
                quaternion = tensor_to_array(workload.egg.get_quat())[0]
                residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
                rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
                np.savez_compressed(
                    frames / f"step{step:06d}.npz",
                    time_s=(step + 1) * 0.01,
                    step=step,
                    stress_voigt_link_pa=stress,
                    vm_pa=vm,
                    q=q,
                    link_position_m=position,
                    link_quaternion_wxyz=quaternion,
                    contact_positions_world_m=points,
                    contact_forces_world_n=forces,
                    contact_normals=contacts["normal"][valid],
                    contact_friction=contacts["friction"][valid],
                    contact_source_geom_a=contacts["geom_a"][valid],
                    contact_source_geom_b=contacts["geom_b"][valid],
                )
                row = {
                    "step": step,
                    "time_s": (step + 1) * 0.01,
                    "phase": tensor_to_array(phase)[0].item(),
                    "position_m": position.tolist(),
                    "quaternion_wxyz": quaternion.tolist(),
                    "q": q.tolist(),
                    "all_fields_finite": finite,
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
                }
                executed += 1
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                if not finite:
                    break
            completed = executed == args.steps
        finally:
            write_json(
                args.output / "result.json",
                {
                    "kind": "native-capped-full-field-history",
                    "method": args.method,
                    "requested_control_ticks": args.steps,
                    "executed_control_ticks": executed,
                    "completed_requested_history": completed,
                    "invalid_environment_steps": invalid,
                    "intrusive_seconds": perf_counter() - started,
                    "reference_accuracy_qualified": False,
                },
            )
    workload.scene.rigid_solver.check_errno()
    if not completed:
        raise RuntimeError("Native candidate stopped with retained invalid full field")


if __name__ == "__main__":
    main()
