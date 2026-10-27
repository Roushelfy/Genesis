"""Locate the first rejected step in the matched volume-DPR Franka baseline.

Run in Slurm rtx-mid/image 1_26. Every step reads the original solver errno,
so these synchronized diagnostics never supply primary throughput timings.
The recipe uses the native controller and the benchmark's seeded policy.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def _api():
    import numpy as np
    import torch

    import genesis as gs
    from examples.rigid.franka_egg_stress import FrankaEgg
    from genesis.utils.misc import qd_to_numpy, tensor_to_array

    from .artifacts import file_id, runtime_identity, write_json

    return SimpleNamespace(**locals())


def _source_files(api):
    root = Path(api.gs.__file__).resolve().parent
    paths = [
        *sorted((root / "engine/solvers/rigid/stress").glob("*.py")),
        root / "engine/simulator.py",
        root / "engine/solvers/rigid/rigid_solver.py",
        root / "options/rigid_stress.py",
        root.parent / "examples/rigid/franka_egg_stress.py",
        root.parent / "examples/rigid/egg_stress_controller.py",
        Path(__file__).resolve(),
    ]
    return {str(path): api.file_id(path) for path in paths}


def _snapshot(api, workload):
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    contacts, state, collider = entry.contacts, entry.state, solver.collider.collider_state
    arrays = {
        name: api.qd_to_numpy(value, transpose=True, copy=True)
        for name, value in (
            ("errno", solver._errno),
            ("contact_status", contacts.status),
            ("contact_valid", contacts.valid),
            ("contact_radius_status", contacts.radius_status),
            ("contact_force", contacts.force),
            ("contact_point", contacts.position),
            ("contact_center", contacts.center),
            ("contact_radius", contacts.radius),
            ("contact_normal", contacts.normal),
            ("contact_friction", contacts.friction),
            ("contact_force_error", contacts.force_error),
            ("contact_point_moment_error", contacts.moment_error),
            ("contact_anchor_face", contacts.anchor_face),
            ("contact_refined", contacts.is_refined),
            ("contact_coefficient", contacts.coefficient),
            ("contact_frame_position", contacts.frame_position),
            ("contact_frame_quaternion", contacts.frame_quaternion),
            ("state_valid", state.valid),
            ("state_step_valid", state.step_valid),
            ("state_invalid_steps", state.invalid_steps),
            ("state_residual_norm_squared", state.residual_norm_squared),
            ("state_rhs_norm_squared", state.rhs_norm_squared),
            ("state_step_peak", state.step_peak),
            ("state_raw_force", state.force),
            ("state_rhs", state.rhs),
            ("state_q", state.displacement),
            ("body_omega", entry.omega),
            ("rigid_links_position", solver.dyn_state.links.pos),
            ("rigid_links_quaternion", solver.dyn_state.links.quat),
            ("rigid_links_angular_velocity", solver.dyn_state.links.cd_ang),
            ("rigid_applied_force", solver.dyn_state.links.cfrc_applied_vel),
            ("rigid_applied_moment", solver.dyn_state.links.cfrc_applied_ang),
            ("rigid_coupling_force", solver.dyn_state.links.cfrc_coupling_vel),
            ("rigid_coupling_moment", solver.dyn_state.links.cfrc_coupling_ang),
            ("collider_contact_count", collider.n_contacts),
            ("collider_contact_order", collider.contact_sort_idx),
            ("collider_contact_link_a", collider.contact_data.link_a),
            ("collider_contact_link_b", collider.contact_data.link_b),
            ("collider_contact_position", collider.contact_data.pos),
            ("collider_contact_force", collider.contact_data.force),
            ("collider_contact_normal", collider.contact_data.normal),
            ("elastic_mass_properties", entry.model.info.mass_properties),
        )
    }
    arrays["rigid_egg_mass"] = api.tensor_to_array(workload.link.get_mass())
    arrays["initial_joints"] = api.tensor_to_array(workload.initial_joints)
    arrays["initial_position"] = api.tensor_to_array(workload.initial_position)
    arrays["initial_quaternion"] = api.tensor_to_array(workload.initial_quaternion)
    arrays["delays"] = workload.delays.copy()
    return arrays


def _finite_max(api, values):
    finite = api.np.isfinite(values)
    return float(api.np.max(api.np.abs(values[finite]))) if finite.any() else None


def _summary(api, arrays):
    valid = arrays["contact_valid"]
    status = arrays["contact_status"]
    return {
        "errno_per_environment": arrays["errno"].tolist(),
        "native_valid_per_environment": arrays["state_valid"].tolist(),
        "step_valid_per_environment": arrays["state_step_valid"].tolist(),
        "invalid_steps_per_environment": arrays["state_invalid_steps"].tolist(),
        "contacts_per_environment": valid.sum(axis=1).tolist(),
        "current_failed_contacts": api.np.argwhere(valid & (status != 0)).tolist(),
        "force_error_finite_max_N": _finite_max(api, arrays["contact_force_error"][valid]),
        "point_moment_error_finite_max_Nm": _finite_max(api, arrays["contact_point_moment_error"][valid]),
        "nonfinite_raw_values": {
            name: int((~api.np.isfinite(value)).sum()) for name, value in arrays.items() if value.dtype.kind in "fc"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=730)
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--pressure-quadrature", choices=("edge", "angle"), default="edge")
    controller = parser.add_mutually_exclusive_group()
    controller.add_argument("--policy", dest="policy", action="store_true", default=True)
    controller.add_argument("--scripted", dest="policy", action="store_false")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION", "rtx-mid") != "rtx-mid":
        parser.error("Scientific execution requires Slurm rtx-mid/image 1_26.")
    if min(args.envs, args.steps, args.quadrature) < 1:
        parser.error("Use positive environment, step and quadrature counts.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Choose a fresh output under /mnt/data/zhaofeng.")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.conditions, args.output / "conditions.npz")
    api, started = _api(), time.perf_counter()
    api.gs.init(backend=api.gs.cuda, precision="64", seed=510000, logging_level="warning")
    source_before = _source_files(api)
    checkout = subprocess.check_output(["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema": "genesis.planar-volume-baseline-diagnostic.v1",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "identity": api.runtime_identity(),
        "checkout_commit": checkout,
        "source_files": source_before,
        "conditions": api.file_id(args.conditions),
        "complete": False,
        "partial": True,
        "passed": False,
        "completed_steps": 0,
        "primary_timing": False,
        "scope": "Continuous native Franka volume-DPR warmup with per-step original errno synchronization. No throughput or spatial-accuracy qualification.",
        "policy": {"enabled": args.policy, "seed": 99173, "layers": [26, 128, 128, 7], "dtype": "float32"},
        "error_polling": "Read original solver errno after every completed step; stop and preserve the first rejection without clearing or weakening it.",
    }
    api.write_json(args.output / "result.json", result)
    workload = api.FrankaEgg(
        args.envs,
        level=1,
        seed=510000,
        conditions=args.conditions,
        egg_geometry="mesh64",
        stress_backend="volume",
        output_mode="max",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        quadrature=args.quadrature,
        pressure_quadrature=args.pressure_quadrature,
        scatter_tasks_per_env=32,
        method="auto",
        inverse_precision="64",
        history_size=0,
    )
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    mesh_before = api.file_id(Path(entry.options.mesh))
    result.update(
        mesh=mesh_before,
        stress_options=entry.options.model_dump(mode="json"),
        condition_count=workload.condition_count,
        dt_seconds=0.01,
        substeps=1,
        link_index=workload.link.idx,
        selected_method=entry.model.selected_method,
        fused_pipeline=entry.fused_pipeline,
        reuse_contact_moments=entry.reuse_contact_moments,
    )
    initial = _snapshot(api, workload)
    api.np.savez_compressed(args.output / "initial.npz", **initial)
    api.write_json(args.output / "result.json", result)
    api.torch.manual_seed(99173)
    policy = (
        api.torch.nn.Sequential(
            api.torch.nn.Linear(26, 128),
            api.torch.nn.Tanh(),
            api.torch.nn.Linear(128, 128),
            api.torch.nn.Tanh(),
            api.torch.nn.Linear(128, 7),
        )
        .to(device=api.gs.device, dtype=api.gs.tc_float)
        .eval()
        .to(dtype=api.torch.float32)
    )
    failure = None
    with (args.output / "steps.jsonl").open("x") as journal, api.torch.inference_mode():
        for step in range(args.steps):
            tick = workload.tick
            resets_before = workload.reset_count
            scheduled = api.np.flatnonzero((tick > workload.delays) & ((tick - workload.delays) % 600 == 0))
            try:
                action = policy(workload.observation().to(dtype=api.torch.float32)) if args.policy else None
                workload.step(action)
                solver.check_errno()
            except (api.gs.GenesisException, ArithmeticError, RuntimeError) as error:
                failure = error
                result["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            record = {
                "attempted_tick": tick,
                "workload_tick_after_attempt": workload.tick,
                "diagnostic_iteration": step,
                "scheduled_reset_environments": scheduled.tolist(),
                "reset_count": workload.reset_count,
                "resets_during_attempt": workload.reset_count - resets_before,
                "completed_without_rejection": failure is None,
            }
            if failure is not None or step % 10 == 0 or step == args.steps - 1:
                snapshot = _snapshot(api, workload)
                record.update(_summary(api, snapshot))
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            if failure is not None:
                api.np.savez_compressed(args.output / "first-rejection.npz", **snapshot)
                result["first_rejection"] = record
                break
            result["completed_steps"] += 1
    final = _snapshot(api, workload)
    api.np.savez_compressed(args.output / "terminal.npz", **final)
    source_after = _source_files(api)
    result.update(
        complete=failure is None and result["completed_steps"] == args.steps,
        partial=failure is not None,
        terminal_tick=workload.tick,
        terminal_reset_count=workload.reset_count,
        terminal_summary=_summary(api, final),
        source_files_after=source_after,
        source_unchanged=source_before == source_after,
        input_files_unchanged=api.file_id(args.conditions) == result["conditions"]
        and api.file_id(Path(entry.options.mesh)) == mesh_before,
        seconds_including_intrusive_diagnostics=time.perf_counter() - started,
    )
    result["passed"] = result["complete"] and result["source_unchanged"] and result["input_files_unchanged"]
    api.write_json(args.output / "result.json", result)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
