"""Solve a declared prefix of the actual original controls from zero velocity and real ground support."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .interrupted_source_verification import verify_interrupted_source
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
    parser.add_argument("--resume-audited-prefix", type=Path)
    parser.add_argument("--prefix-audit", type=Path)
    parser.add_argument("--interrupted-source-observation", type=Path)
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
    prefix_only = args.resume_audited_prefix is not None
    source_run = args.resume_audited_prefix if prefix_only else args.resume_run
    if source_run is not None:
        if prefix_only:
            assert args.resume_run is None and args.recovery_run is None and args.recovery_audit is None
            assert args.prefix_audit is not None
            if args.interrupted_source_observation is None:
                closed_source(source_run)
            else:
                verify_interrupted_source(source_run, args.interrupted_source_observation)
            audit_run = args.prefix_audit
        else:
            assert args.prefix_audit is None and args.interrupted_source_observation is None
            assert args.recovery_run is not None and args.recovery_audit is not None
            for source in [source_run, args.recovery_run]:
                closed_source(source)
            assert int((source_run / "returncode.txt").read_text()) != 0
            assert int((args.recovery_run / "returncode.txt").read_text()) == 0
            audit_run = args.recovery_audit
        closed_source(audit_run)
        assert int((audit_run / "returncode.txt").read_text()) == 0
        source_config = json.loads((source_run / "results/configuration.json").read_text())
        for key, value in [("capture", args.capture), ("protocol_run", args.protocol_run)]:
            assert Path(source_config[key]).resolve() == value.resolve()
        for key, value in [("dt", args.dt), ("dhat", args.dhat)]:
            assert source_config[key] == value
        assert (
            hashlib.sha256(Path(source_config["mjcf"]).read_bytes()).digest()
            == hashlib.sha256(args.mjcf.read_bytes()).digest()
        )
        audit = json.loads((audit_run / "results/summary.json").read_text())
        assert Path(audit["run"]).resolve() == source_run.resolve()
        if prefix_only:
            assert audit["recovery_run"] is None
            resume_step = audit["steps_checked"]
            files = sorted((source_run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
            assert len(files) == resume_step
            assert all(path.name == f"step{step:05d}.npz" for step, path in enumerate(files, 1))
            if args.interrupted_source_observation is not None:
                assert audit["source_closure_kind"] == "separate_post_termination_observation"
                assert (
                    audit["source_observation_sha256"]
                    == hashlib.sha256(args.interrupted_source_observation.read_bytes()).hexdigest()
                )
            # These are independently audited physical-state records, not
            # invented Newton iteration histories. Original logs remain intact.
            rows.extend(
                {**row, "prefix_record_origin": str(audit_run), "phase": "audited_saved_prefix"}
                for row in audit["rows"]
            )
            assert [row["step"] for row in rows] == list(range(1, resume_step + 1))
        else:
            assert Path(audit["recovery_run"]).resolve() == args.recovery_run.resolve()
            resume_step = audit["independently_verified_recovered_step"]
            prefix_history = source_run / "results/prefix_history.json"
            if prefix_history.exists():
                rows.extend(json.loads(prefix_history.read_text()))
            for line in (source_run / "reference.log").read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and "step" in row and "time_s" in row:
                    rows.append(row)
            assert [row["step"] for row in rows] == list(range(1, resume_step))
        assert audit["steps_checked"] == resume_step and resume_step < args.macro_steps * substeps
        provenance = []
        for name in ["initial_state.npz", "reference_geometry.npz"]:
            with np.load(source_run / "results" / name) as source, np.load(args.output / name) as actual:
                assert source.files == actual.files
                for key in source.files:
                    np.testing.assert_array_equal(source[key], actual[key])
        for step in range(1, resume_step + 1):
            source = (
                args.recovery_run / "results/recovered.npz"
                if step == resume_step and not prefix_only
                else source_run / f"results/step{step:05d}.npz"
            )
            destination = args.output / f"step{step:05d}.npz"
            shutil.copy2(source, destination)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            provenance.append({"step": step, "source": str(source), "sha256": digest})
        with np.load(args.output / f"step{resume_step:05d}.npz") as saved:
            previous = saved["displacement"].copy()
            body_u = previous[: system.n].copy()
            q, velocity, body_velocity = [
                saved[key].copy() for key in ["joint_position", "joint_velocity", "body_velocity"]
            ]
        if not prefix_only:
            recovery = json.loads((args.recovery_run / "results/summary.json").read_text())
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
                    "source_run": str(source_run),
                    "recovery_run": None if prefix_only else str(args.recovery_run),
                    "independent_audit": str(audit_run),
                    "resume_mode": "audited_saved_prefix" if prefix_only else "audited_recovered_next_step",
                    "source_observation": None
                    if args.interrupted_source_observation is None
                    else str(args.interrupted_source_observation),
                    "original_solver_history_preserved_at": str(source_run / "reference.log"),
                    "copied_states": provenance,
                    "pseudo_stages_excluded_from_physical_history": True,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        assert args.recovery_run is None and args.recovery_audit is None
        assert args.prefix_audit is None and args.interrupted_source_observation is None
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
