"""Retain PR22 full-field capped histories, with optional existing read-only solver/contact diagnostics."""

import json
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..shell_comparison import diagnostics
from ..shell_comparison.application import ShellEgg
from ..shell_comparison.rollout import controller_step, make_parser
from .artifacts import file_id, runtime_identity, write_json


def main():
    args = make_parser().parse_args()
    assert args.model == "shell" and args.envs == 1
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    with np.load(args.conditions) as bank:
        coefficient = 0.6 * bank["friction"][0, 0].item()
    assert np.isclose(coefficient, args.coupling_friction, rtol=0, atol=1e-15)
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    started = perf_counter()
    workload = ShellEgg(args)
    torch.cuda.synchronize()
    faces = tensor_to_array(workload.egg.get_faces())[0]
    np.savez_compressed(
        args.output / "rest.npz",
        world_initial_vertices_m=tensor_to_array(workload.initial_vertices)[0],
        faces=faces,
        vertices_m=workload.egg.init_verts - np.array([0.65, 0, 0.031]),
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
            "dt": 0.01,
            "substeps": args.substeps,
            "precision": args.precision,
            "pcg_cap": args.iterations,
            "pcg_threshold": args.tolerance,
            "coarse_interval": args.coarse_interval,
            "coarse_patches": args.patches,
            "damping": args.damping,
            "coupling_friction": args.coupling_friction,
            "setup_seconds": perf_counter() - started,
            "source_scope": "PR22 plus existing read-only diagnostics"
            if args.instrumented
            else "original PR22 plus full-field observer",
            "stress_frame": "world Cauchy observer, nominal bottom/middle/top fibers per face",
            "scope": "intrusive complete capped history, excluded from primary timing",
            "nominal_thickness_mass_kg": workload.egg.topology.faces_mass.sum().item(),
            "fracture_enabled": workload.scene.sim.shell_solver._static_config.has_fracture,
            "plasticity_enabled": workload.scene.sim.shell_solver._static_config.has_plasticity,
        },
    )
    completed = False
    count = 0
    nonfinite = False
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(args.steps):
                if args.instrumented:
                    diagnostics.ACTIVE.clear()
                phase = controller_step(workload, True)
                tensor = tensor_to_array(workload.observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
                vm = tensor_to_array(workload.observation.von_mises)[:, 0].reshape(-1, 3)
                vertices = tensor_to_array(workload.egg.get_verts_pos())[0]
                q = tensor_to_array(workload.robot.get_qpos())[0]
                finite = bool(np.isfinite(tensor).all() and np.isfinite(vertices).all() and np.isfinite(q).all())
                payload = {
                    "time_s": (step + 1) * 0.01,
                    "step": step,
                    "cauchy_world_pa": tensor,
                    "vm_pa": vm,
                    "vertices_world_m": vertices,
                    "q": q,
                }
                if args.instrumented:
                    payload["contact_impulses_ns"] = tensor_to_array(diagnostics.ACTIVE.impulses)[:, 0]
                    payload["contact_positions_world_m"] = tensor_to_array(diagnostics.ACTIVE.points)[:, 0]
                np.savez_compressed(frames / f"step{step:06d}.npz", **payload)
                row = {
                    "step": step,
                    "time_s": (step + 1) * 0.01,
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
                        contact_average_force_over_control_interval_n=(impulse.sum(axis=0) / 0.01).tolist(),
                        contact_average_force_scope="Accumulated impulses over all substeps; cannot be labeled an instantaneous final-substep force",
                    )
                count += 1
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                if not finite:
                    nonfinite = True
                    break
            completed = count == args.steps and not nonfinite
        finally:
            write_json(
                args.output / "result.json",
                {
                    "kind": "PR22-capped-full-field-history",
                    "requested_control_ticks": args.steps,
                    "executed_control_ticks": count,
                    "completed_requested_history": completed,
                    "nonfinite_output": nonfinite,
                    "intrusive_seconds": perf_counter() - started,
                    "reference_accuracy_qualified": False,
                },
            )
    workload.scene.rigid_solver.check_errno()
    if not completed:
        raise RuntimeError("Shell history stopped with retained nonfinite state")


if __name__ == "__main__":
    main()
