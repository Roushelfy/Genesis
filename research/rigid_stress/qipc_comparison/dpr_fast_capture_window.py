"""Capture exact nearby strict-volume frames before any rejected policy feedback.

The historical221 recipe uses its original scripted URDF source/64-condition
bank; policy606 uses the archived FP64-to-FP32 MLP recipe and mesh64. This is
intrusive diagnostic capture, never primary timing or stress qualification.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace


def _api():
    import numpy as np
    import torch

    import genesis as gs
    from examples.rigid.franka_egg_stress import FrankaEgg
    from genesis.utils.misc import qd_to_numpy, tensor_to_array

    from .artifacts import runtime_identity

    return SimpleNamespace(**locals())


def _identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _snapshot(api, workload):
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    result = {
        "contact_" + ("point" if field.name == "position" else field.name): api.qd_to_numpy(
            vars(entry.contacts)[field.name], transpose=True, copy=True
        )
        for field in dataclasses.fields(entry.contacts)
    }
    result.update(
        {
            "state_" + field.name: api.qd_to_numpy(vars(entry.state)[field.name], transpose=True, copy=True)
            for field in dataclasses.fields(entry.state)
        }
    )
    result.update(
        body_omega=api.qd_to_numpy(entry.omega, transpose=True, copy=True),
        errno=api.qd_to_numpy(solver._errno, transpose=True, copy=True),
        current_body_position=api.tensor_to_array(workload.egg.get_pos()),
        current_body_quaternion=api.tensor_to_array(workload.egg.get_quat()),
        rigid_links_position=api.qd_to_numpy(solver.dyn_state.links.pos, transpose=True, copy=True),
        rigid_links_quaternion=api.qd_to_numpy(solver.dyn_state.links.quat, transpose=True, copy=True),
        rigid_links_angular_velocity=api.qd_to_numpy(solver.dyn_state.links.cd_ang, transpose=True, copy=True),
        rigid_applied_force=api.qd_to_numpy(solver.dyn_state.links.cfrc_applied_vel, transpose=True, copy=True),
        rigid_applied_moment=api.qd_to_numpy(solver.dyn_state.links.cfrc_applied_ang, transpose=True, copy=True),
        rigid_coupling_force=api.qd_to_numpy(solver.dyn_state.links.cfrc_coupling_vel, transpose=True, copy=True),
        rigid_coupling_moment=api.qd_to_numpy(solver.dyn_state.links.cfrc_coupling_ang, transpose=True, copy=True),
    )
    return result


def _witness(api, archived, current, env, slot):
    prefix = "contact_" if "contact_valid" in archived else ""
    rows = {}
    for key in (
        "valid",
        "status",
        "radius_status",
        "position",
        "force",
        "normal",
        "friction",
        "radius",
        "center",
        "anchor_face",
    ):
        alias = "point" if prefix and key == "position" else key
        first = archived[prefix + alias][env, slot]
        second = current["contact_" + ("point" if key == "position" else key)][env, slot]
        row = {
            "historical": first.tolist(),
            "recaptured": second.tolist(),
            "bitwise_equal": first.tobytes() == second.tobytes(),
        }
        if first.dtype.kind == "f":
            delta = api.np.abs(first - second)
            scale = max(float(api.np.max(api.np.abs(first))), float(api.np.max(api.np.abs(second))), 1e-30)
            row.update(
                maximum_absolute_difference=float(api.np.max(delta)),
                relative_to_component_max=float(api.np.max(delta)) / scale,
                relative_1e_12_diagnostic=bool(api.np.max(delta) <= 1e-12 * scale),
            )
        rows[key] = row
    return {
        "selection_environment_slot": [env, slot],
        "fields": rows,
        "all_bitwise_equal": all(row["bitwise_equal"] for row in rows.values()),
        "meaning": "Recorded identity and numerical continuity witness, not an automatic same-frame or application-accuracy certification; no source input is adjusted to match the historical sample.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("historical221", "policy606"), required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--historical-failure", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "rtx-mid"
        or os.environ.get("GENESIS_IMAGE_VER") != "1_26"
    ):
        parser.error("Scientific capture requires Slurm rtx-mid/image 1_26.")
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(Path("/mnt/data/zhaofeng")):
        parser.error("Use a fresh data-side capture directory.")
    args.output.mkdir(parents=True, exist_ok=False)
    policy_mode = args.recipe == "policy606"
    batch, target_env, failure_tick = (4, 2, 606) if policy_mode else (64, 1, 221)
    ticks = (failure_tick - 2, failure_tick - 1, failure_tick)
    api = _api()
    if policy_mode:
        api.gs.init(backend=api.gs.cuda, precision="64", seed=510000, logging_level="warning")
    else:
        # Match the original check_automatic_dpr_scene invocation, which did not override init seed.
        api.gs.init(backend=api.gs.cuda, precision="64", logging_level="warning")
    source = Path(api.gs.__file__).resolve().parent
    source_paths = [
        source / "engine/solvers/rigid/stress/contact.py",
        source / "engine/solvers/rigid/stress/recovery.py",
        source.parent / "examples/rigid/franka_egg_stress.py",
        source.parent / "examples/rigid/egg_stress_controller.py",
        Path(__file__).resolve(),
    ]
    source_before = [_identity(path) for path in source_paths]
    result = {
        "schema": "genesis.dpr-fast-strict-nearby-capture.v1",
        "complete": False,
        "passed": False,
        "goal_complete": False,
        "primary_timing": False,
        "spatial_qualification": False,
        "recipe": args.recipe,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "identity": api.runtime_identity(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(args.git_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources_before": source_before,
        "conditions": _identity(args.conditions),
        "historical_failure": _identity(args.historical_failure),
        "requested_attempt_ticks": ticks,
        "mechanical_recipe": {
            "batch": batch,
            "egg_geometry": "mesh64" if policy_mode else "urdf",
            "output_mode": "max" if policy_mode else "full",
            "Franka_seed": 510000,
            "quadrature": 16,
            "pressure_quadrature": "historical edge rule, no angle override",
            "Hertz_radius_and_pressure": True,
            "fused_pipeline": True,
            "contact_moment_reuse": "auto",
            "policy_seed": 99173 if policy_mode else None,
        },
        "stop_policy": "Preserve first rejection and raw snapshot immediately; never clear errno or execute the next action from rejected/NaN stress feedback.",
        "captures": [],
    }
    _write(args.output / "result.json", result)
    # Every keyword is supported by both original5c44042 and baseline d661.
    workload = api.FrankaEgg(
        batch,
        conditions=args.conditions,
        output_mode="max" if policy_mode else "full",
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=16,
        stress_backend="volume",
        shell_gauge="translation",
        method="auto",
        inverse_precision="64",
        history_size=0,
        surface_inverse=True,
        box_edge_policy="planar",
        fused_pipeline=True,
        contact_moment_reuse=None,
        egg_geometry="mesh64" if policy_mode else "urdf",
    )
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    mesh = Path(entry.options.mesh)
    result.update(
        stress_options=entry.options.model_dump(mode="json"),
        selected_method=entry.model.selected_method,
        actual_fused_pipeline=entry.fused_pipeline,
        actual_contact_moment_reuse=entry.reuse_contact_moments,
        mesh=_identity(mesh),
        dt_seconds=0.01,
        substeps=1,
    )
    with api.np.load(args.historical_failure, allow_pickle=False) as saved:
        archived = {name: saved[name] for name in saved.files}
    model = {
        field.name: api.qd_to_numpy(vars(entry.model.info)[field.name], transpose=True, copy=True)
        for field in dataclasses.fields(entry.model.info)
    }
    api.np.savez_compressed(args.output / "native-model.npz", **model)
    api.np.savez_compressed(
        args.output / "initial-context.npz",
        initial_joints=api.tensor_to_array(workload.initial_joints),
        initial_position=api.tensor_to_array(workload.initial_position),
        initial_quaternion=api.tensor_to_array(workload.initial_quaternion),
        delays=workload.delays.copy(),
    )
    policy = None
    if policy_mode:
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
    if policy_mode:
        digest = hashlib.sha256()
        policy_arrays = {}
        for name, tensor in policy.state_dict().items():
            value = api.tensor_to_array(tensor)
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
            policy_arrays[name] = value
        api.np.savez_compressed(args.output / "actual-policy-weights.npz", **policy_arrays)
        result["actual_policy_weights_sha256"] = digest.hexdigest()
        result["policy_control_scope"] = (
            "Seed99173 FP64 construction then FP32 inference; actual weights saved, not inferred from seed alone."
        )
    rejected_at = None
    with (args.output / "steps.jsonl").open("x") as journal, api.torch.inference_mode():
        for step in range(failure_tick + 1):
            tick, resets_before = workload.tick, workload.reset_count
            if tick != step:
                raise RuntimeError("Recipe tick no longer matches the historical zero-origin frame convention.")
            pregeometry = {}
            if not policy_mode:
                # Preserve the historical per-step preintegration FK and geometry read schedule.
                solver.update_forward_pos()
                pregeometry = {
                    "preintegration_geoms_pos": api.qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True, copy=True),
                    "preintegration_geoms_quat": api.qd_to_numpy(
                        solver.dyn_state.geoms.quat, transpose=True, copy=True
                    ),
                }
            exception, observation_before, action = None, None, None
            try:
                if policy_mode:
                    observation_before = workload.observation()
                    action = policy(observation_before.to(dtype=api.torch.float32))
                workload.step(action)
                solver.check_errno()
            except (api.gs.GenesisException, ArithmeticError, RuntimeError) as error:
                exception = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                rejected_at = tick
            row = {
                "attempted_tick": tick,
                "tick_after_attempt": workload.tick,
                "reset_count": workload.reset_count,
                "resets_during_attempt": workload.reset_count - resets_before,
                "rejected": exception is not None,
                "error": exception,
            }
            if tick in ticks or exception is not None:
                raw = _snapshot(api, workload)
                raw.update(pregeometry)
                if observation_before is not None:
                    raw["policy_observation_before_step"] = api.tensor_to_array(observation_before)
                if action is not None:
                    raw["policy_action_before_step"] = api.tensor_to_array(action)
                path = args.output / f"frame-{tick:06d}.npz"
                api.np.savez_compressed(path, **raw)
                valid, status = raw["contact_valid"], raw["contact_status"]
                row.update(
                    snapshot=_identity(path),
                    errno=raw["errno"].tolist(),
                    loaded_failed_sources=api.np.argwhere(valid & (status != 0)).tolist(),
                    original_native_valid=raw["state_valid"].tolist(),
                    original_step_valid=raw["state_step_valid"].tolist(),
                    nonfinite_fields={
                        key: int((~api.np.isfinite(value)).sum())
                        for key, value in raw.items()
                        if value.dtype.kind in "fc"
                    },
                )
                result["captures"].append(row)
                if tick == failure_tick:
                    result["historical_source_witness"] = _witness(api, archived, raw, target_env, 0)
                _write(args.output / "result.json", result)
            if tick % 10 == 0 or tick in ticks or exception is not None:
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
            if exception is not None:
                break
    sources_after = [_identity(path) for path in source_paths]
    captured = {row["attempted_tick"] for row in result["captures"]}
    result.update(
        complete=set(ticks).issubset(captured),
        expected_strict_rejection_reproduced=rejected_at == failure_tick,
        first_rejected_tick=rejected_at,
        terminal_tick=workload.tick,
        terminal_reset_count=workload.reset_count,
        source_files_unchanged=source_before == sources_after,
        input_files_unchanged=_identity(args.conditions) == result["conditions"]
        and _identity(args.historical_failure) == result["historical_failure"]
        and _identity(mesh) == result["mesh"],
        sources_after=sources_after,
        orientation="Contact tensors environment,slot[,component]; frame/omega environment,component. Internal active_pairs/pool arrays and constant model fields retain their explicitly named native shapes. State arrays move final batch axis to front.",
    )
    result["passed"] = (
        result["complete"]
        and result["expected_strict_rejection_reproduced"]
        and result["source_files_unchanged"]
        and result["input_files_unchanged"]
    )
    result["pass_semantics"] = (
        "Capture and preserved-failure contract only. Recorded native rejection remains a failure, and historical numerical identity witness is not silently promoted to an exact physical/frame qualification."
    )
    _write(args.output / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("Capture window or preserved strict failure differed; keep all raw evidence for diagnosis.")


if __name__ == "__main__":
    main()
