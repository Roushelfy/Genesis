"""Solve a declared prefix of the actual original controls from zero velocity and real ground support."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .original_franka_ground import GroundedOriginalContact, build_original, contact_groups
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol-run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--macro-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--recovery-run", type=Path)
    parser.add_argument("--recovery-audit", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    static, frame, report, protocol = build_original(args.capture, args.protocol_run, args.mjcf, args.dt, args.dhat)
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    assert 0 < args.macro_steps <= len(controls)
    substeps = round(protocol["macro_dt_s"] / args.dt)
    assert abs(substeps * args.dt - protocol["macro_dt_s"]) < 1e-14
    system = GroundedOriginalContact(static, frame, report, args.mjcf, args.dt)
    q = system.initial_q.copy()
    body_u, body_velocity, velocity = np.zeros(system.n, dtype=np.longdouble), np.zeros(system.n), np.zeros(9)
    previous = system.pack(body_u, q)[0]
    np.savez_compressed(
        args.output / "initial_state.npz",
        body_displacement=body_u,
        joint_position=q,
        body_velocity=body_velocity,
        joint_velocity=velocity,
        full_displacement=previous,
    )
    np.savez_compressed(
        args.output / "reference_geometry.npz",
        vertices=static.body.mesh.p.T,
        tetrahedra=static.body.mesh.t.T,
        rest_positions=static.rest,
        faces=static.faces,
        groups=static.groups,
        ground_nodes=static.ground_nodes,
        body_component_permutation=static.parts[0].permutation,
        world_coordinate_origin_m=static.world_coordinate_origin_m,
    )
    rows = []
    resume_step = 0
    if args.resume_run is not None:
        assert args.recovery_run is not None and args.recovery_audit is not None
        for source in [args.resume_run, args.recovery_run, args.recovery_audit]:
            closed_source(source)
        assert int((args.resume_run / "returncode.txt").read_text()) != 0
        assert int((args.recovery_run / "returncode.txt").read_text()) == 0
        assert int((args.recovery_audit / "returncode.txt").read_text()) == 0
        source_config = json.loads((args.resume_run / "results/configuration.json").read_text())
        for key in ["capture", "protocol_run"]:
            assert Path(source_config[key]).resolve() == getattr(args, key).resolve()
        for key in ["dt", "dhat"]:
            assert source_config[key] == getattr(args, key)
        assert (
            hashlib.sha256(Path(source_config["mjcf"]).read_bytes()).digest()
            == hashlib.sha256(args.mjcf.read_bytes()).digest()
        )
        audit = json.loads((args.recovery_audit / "results/summary.json").read_text())
        assert Path(audit["run"]).resolve() == args.resume_run.resolve()
        assert Path(audit["recovery_run"]).resolve() == args.recovery_run.resolve()
        resume_step = audit["independently_verified_recovered_step"]
        assert audit["steps_checked"] == resume_step and resume_step < args.macro_steps * substeps
        prefix_history = args.resume_run / "results/prefix_history.json"
        if prefix_history.exists():
            rows.extend(json.loads(prefix_history.read_text()))
        for line in (args.resume_run / "reference.log").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "step" in row and "time_s" in row:
                rows.append(row)
        assert [row["step"] for row in rows] == list(range(1, resume_step))
        provenance = []
        for name in ["initial_state.npz", "reference_geometry.npz"]:
            with np.load(args.resume_run / "results" / name) as source, np.load(args.output / name) as actual:
                assert source.files == actual.files
                for key in source.files:
                    np.testing.assert_array_equal(source[key], actual[key])
        for step in range(1, resume_step + 1):
            source = (
                args.recovery_run / "results/recovered.npz"
                if step == resume_step
                else args.resume_run / f"results/step{step:05d}.npz"
            )
            destination = args.output / f"step{step:05d}.npz"
            shutil.copy2(source, destination)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            provenance.append({"step": step, "source": str(source), "sha256": digest})
        recovery = json.loads((args.recovery_run / "results/summary.json").read_text())
        with np.load(args.output / f"step{resume_step:05d}.npz") as saved:
            previous = saved["displacement"].copy()
            body_u = previous[: system.n].copy()
            q, velocity, body_velocity = [
                saved[key].copy() for key in ["joint_position", "joint_velocity", "body_velocity"]
            ]
        rows.append(
            {
                "step": resume_step,
                "macro_tick": (resume_step - 1) // substeps,
                "time_s": resume_step * args.dt,
                "fresh_robot_step_effort_residual": recovery["attempts"][-1]["fresh_robot_effort_residual"],
                **recovery["attempts"][-1]["residuals"],
            }
        )
        (args.output / "prefix_history.json").write_text(json.dumps(rows, indent=2) + "\n")
        (args.output / "resume_provenance.json").write_text(
            json.dumps(
                {
                    "source_run": str(args.resume_run),
                    "recovery_run": str(args.recovery_run),
                    "independent_audit": str(args.recovery_audit),
                    "copied_states": provenance,
                    "pseudo_stages_excluded_from_physical_history": True,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        assert args.recovery_run is None and args.recovery_audit is None
    for macro in range(args.macro_steps):
        control = controls[macro]
        assert control["tick"] == macro
        system.target[:] = control["target"]
        system.caps[7:] = control["finger_cap_N"]
        for micro in range(substeps):
            step = macro * substeps + micro + 1
            if step <= resume_step:
                continue
            old_body, old_q = body_u.copy(), q.copy()
            old_body_velocity, old_velocity = body_velocity.copy(), velocity.copy()
            system.begin_dynamic_step(old_q, old_velocity, old_body + args.dt * old_body_velocity)
            try:
                body_u, velocity, full, contact = system.solve(body_u, velocity, previous, False, f"original{step:05d}")
                fresh = system.verify_robot_step(velocity, contact)
                groups, ground_points = contact_groups(system, full)
            except Exception:
                np.savez_compressed(
                    args.output / "failure.npz",
                    **system.failure_state,
                    previous_full_displacement=previous,
                    previous_joint_position=old_q,
                    previous_joint_velocity=old_velocity,
                    previous_body_displacement=old_body,
                    previous_body_velocity=old_body_velocity,
                    target=system.target,
                    caps=system.caps,
                )
                raise
            q = system.previous_q + np.longdouble(args.dt) * velocity
            body_velocity = (body_u - old_body) / args.dt
            row = {
                "step": step,
                "macro_tick": macro,
                "time_s": step * args.dt,
                "fresh_robot_step_effort_residual": fresh,
                "contact_groups": groups,
                "ground_contact_points_local_m": ground_points,
                **system.history[-1],
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"step{step:05d}.npz",
                displacement=full,
                joint_position=q,
                joint_velocity=velocity,
                body_velocity=body_velocity,
                body_acceleration=(body_velocity - old_body_velocity) / args.dt,
                contact_force_N=-contact,
                applied_force_N=system.gravity,
                controller_target=system.target,
                controller_caps=system.caps,
            )
            previous = full.copy()
            print(json.dumps(row, default=float), flush=True)
    vector = np.empty(system.n)
    vector[static.parts[0].permutation] = body_u
    tensor, vm, upper = system.contact.bulk.stress_bounds(vector)
    np.savez_compressed(args.output / "body_fields.npz", tensor_Pa=tensor, vm_Pa=vm, upper_peak_Pa=upper)
    summary = {
        "scope": __doc__,
        "steps": rows,
        "original_macro_steps_solved": args.macro_steps,
        "available_original_macro_steps": len(controls),
        "raw_global_peak_lower_Pa": float(vm.max()),
        "raw_global_peak_upper_Pa": float(upper.max()),
        "whole_volume_j_lower_bound": system.contact.bulk.determinant_lower_bound(vector),
        "goal_completion": False,
        "limitations": "This is an original ground-supported history prefix, not the complete grasp/lift. Complete independent force/state/control audits, temporal and spatial/contact convergence, corresponding rigid feedback, native models and verified accuracy/cost remain required. The ground triangle is used only where accepted contact points are strictly interior; non-body contact pairs would be rejected.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
