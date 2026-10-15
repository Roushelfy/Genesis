"""Solve held initialization then free FEM/complete Panda joint contact dynamics."""

import argparse
import json
from pathlib import Path

import numpy as np

from .articulated_franka_contact import ArticulatedContact
from .franka_contact_reference import build_from_capture
from .summarize_fine_references import closed_source


def check_tangent(system, body_u, variable, previous):
    result = system.evaluate(body_u, variable, previous, True)
    bb, bq, qb, qq = result[-1]
    rng = np.random.default_rng(20261015)
    body_direction = rng.normal(size=system.n)
    body_direction /= np.linalg.norm(body_direction)
    joint_direction = rng.normal(size=9)
    joint_direction /= np.linalg.norm(joint_direction)
    expected = np.r_[bb @ body_direction + bq @ joint_direction, qb @ body_direction + qq @ joint_direction]
    rows = []
    for step in [1e-8, 3e-9, 1e-9]:
        plus = system.evaluate(body_u + step * body_direction, variable + step * joint_direction, previous, False)
        minus = system.evaluate(body_u - step * body_direction, variable - step * joint_direction, previous, False)
        observed = (np.r_[plus[0], plus[1]] - np.r_[minus[0], minus[1]]) / (2 * step)
        error = float(np.linalg.norm(observed - expected) / np.linalg.norm(expected))
        rows.append({"perturbation": step, "relative_complete_tangent_error": error})
    assert rows[-1]["relative_complete_tangent_error"] < 1e-3, rows
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=64)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--check-jacobian", action="store_true")
    parser.add_argument("--closure-run", type=Path)
    parser.add_argument("--fixed-arm", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    static, frame = build_from_capture(args.capture, args.dhat)
    report = json.loads((args.capture / "results/summary.json").read_text())
    system = ArticulatedContact(static, frame, report, args.mjcf, args.dt)
    body_u, q = np.zeros(system.n), system.initial_q.copy()
    previous = system.pack(body_u, q)[0]
    tangents = []
    if args.closure_run is not None:
        closed_source(args.closure_run)
        assert int((args.closure_run / "returncode.txt").read_text()) == 0
        initial_config = json.loads((args.closure_run / "results/configuration.json").read_text())
        for key in ["capture", "dhat", "control_steps", "dt"]:
            assert initial_config[key] == vars(args)[key] or initial_config[key] == str(vars(args)[key])
        with np.load(args.closure_run / "results/closure_state.npz") as saved:
            body_u, q = saved["body_displacement"].copy(), saved["joint_position"].copy()
            previous = saved["full_displacement"].copy()
            system.target[:] = saved["target"]
        np.testing.assert_allclose(system.pack(body_u, q)[0], previous, rtol=0, atol=1e-12)
    for step in range(1, args.control_steps + 1) if args.closure_run is None else ():
        system.target[7:] = 0.04 + (0.018 - 0.04) * step / args.control_steps
        try:
            body_u, q, full, contact = system.solve(body_u, q, previous, True, f"closure{step:04d}")
        except Exception:
            np.savez_compressed(
                args.output / "failure.npz",
                **system.failure_state,
                previous_full_displacement=previous,
                target=system.target,
            )
            raise
        previous = full.copy()
        np.savez_compressed(
            args.output / "last_accepted_state.npz",
            body_displacement=body_u,
            joint_position=q,
            full_displacement=full,
            contact_force_N=-contact,
            target=system.target,
            phase="closure",
            step=step,
        )
        if args.check_jacobian and step == args.control_steps:
            tangents.append({"phase": "held_contact", "rows": check_tangent(system, body_u, q, previous)})
    np.savez_compressed(
        args.output / "closure_state.npz",
        body_displacement=body_u,
        joint_position=q,
        full_displacement=previous,
        target=system.target,
    )
    np.savez_compressed(
        args.output / "reference_geometry.npz",
        vertices=static.body.mesh.p.T,
        tetrahedra=static.body.mesh.t.T,
        rest_positions=static.rest,
        faces=static.faces,
        body_component_permutation=static.parts[0].permutation,
        world_coordinate_origin_m=static.world_coordinate_origin_m,
    )
    velocity, body_velocity = np.zeros(9), np.zeros(system.n)
    system.fixed_arm = args.fixed_arm
    closure_q = q.copy()
    records = []
    for step in range(1, args.steps + 1):
        old_body, old_body_velocity = body_u.copy(), body_velocity.copy()
        old_q, old_velocity = q.copy(), velocity.copy()
        system.begin_dynamic_step(old_q, old_velocity, old_body + args.dt * old_body_velocity)
        if args.check_jacobian and step == 1:
            tangents.append({"phase": "free_inertial", "rows": check_tangent(system, body_u, velocity, previous)})
        try:
            body_u, velocity, full, contact = system.solve(body_u, velocity, previous, False, f"dynamic{step:04d}")
        except Exception:
            np.savez_compressed(
                args.output / "failure.npz",
                **system.failure_state,
                previous_joint_position=old_q,
                previous_joint_velocity=old_velocity,
                previous_body_displacement=old_body,
                previous_body_velocity=old_body_velocity,
                previous_full_displacement=previous,
            )
            raise
        q = old_q + args.dt * velocity
        body_velocity = (body_u - old_body) / args.dt
        direct = system.verify_robot_step(velocity, contact)
        row = {"step": step, "time_s": step * args.dt, "fresh_robot_step_effort_residual": direct, **system.history[-1]}
        records.append(row)
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=full,
            joint_position=q,
            joint_velocity=velocity,
            body_velocity=body_velocity,
            body_acceleration=(body_velocity - old_body_velocity) / args.dt,
            contact_force_N=-contact,
            applied_force_N=system.gravity,
            arm_support_generalized_effort=system.last_support_effort,
            body_component_permutation=static.parts[0].permutation,
        )
        previous = full.copy()
        print(json.dumps(row), flush=True)
    vector = np.empty(system.n)
    vector[static.parts[0].permutation] = body_u
    tensor, vm, upper = static.bulk.stress_bounds(vector)
    np.savez_compressed(args.output / "body_fields.npz", tensor_Pa=tensor, vm_Pa=vm, upper_peak_Pa=upper)
    output = {
        "scope": __doc__,
        "steps": records,
        "body_tetrahedra": static.body.mesh.nelements,
        "original_force_budgets": {"body_N": 1e-9, "joint_N_or_Nm": 1e-9},
        "all_accepted_residuals": system.history,
        "tangent_qualifications": tangents,
        "body_peak_lower_Pa": float(vm.max()),
        "body_peak_upper_Pa": float(upper.max()),
        "body_determinant_lower_bound": static.bulk.determinant_lower_bound(vector),
        "final_joint_position": q.tolist(),
        "final_joint_velocity": velocity.tolist(),
        "controller_target": system.target.tolist(),
        "controller_kp": system.kp.tolist(),
        "controller_kv": system.kv.tolist(),
        "controller_caps": system.caps.tolist(),
        "closure_joint_position": closure_q.tolist(),
        "fixed_arm_control": args.fixed_arm,
        "closure_source": str(args.closure_run) if args.closure_run is not None else None,
        "goal_completion": False,
        "limitations": "Complete nine-joint nonlinear kinematics/PD/equality and independent FEM/contact are coupled with original1e-9 body/joint force gates and fresh MuJoCo step checks. Held six-point quasi-static initialization followed by zero initial velocity differs from original approach/lift and captured angular history. Endpoint implicit body/contact/PD forces use a MuJoCo old-state step response; time-integration, history, spatial/contact/regularization and native feedback/model/accuracy-cost still need qualification. Newton CCD guards accepted endpoint chords; no continuous nonlinear rigid-motion CCD claim is made. This is an application pilot, not completion of full physical acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ["steps", "all_accepted_residuals"]}), flush=True)


if __name__ == "__main__":
    main()
