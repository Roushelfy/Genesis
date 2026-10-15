"""Solve a rigid-egg counterpart of the declared full-arm deformable-contact hold."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .captured_reference_motion import initial_motion
from .franka_contact_reference import build_from_capture
from .rigid_egg_articulated_contact import RigidEggArticulatedContact, rotation_terms
from .summarize_fine_references import closed_source


def qualify_kinematics(system):
    vector = np.array([0.3, -0.2, 0.5])
    _, rotation, _ = rotation_terms(vector)
    np.testing.assert_allclose(rotation, Rotation.from_rotvec(vector).as_matrix(), rtol=0, atol=1e-15)
    pose = np.r_[[0.001, -0.002, 0.003], vector]
    _, jacobian = system.body_kinematics(pose)
    direction = np.random.default_rng(2026101502).normal(size=6)
    step = 1e-6
    observed = (
        system.body_kinematics(pose + step * direction)[0] - system.body_kinematics(pose - step * direction)[0]
    ) / (2 * step)
    relative = float(np.linalg.norm(observed - jacobian @ direction) / np.linalg.norm(jacobian @ direction))
    assert relative < 1e-9
    return relative


def qualify_tangent(system, pose, variable, previous):
    result = system.evaluate_rigid(pose, variable, previous, True)
    bb, bq, qb, qq = result[-1]
    matrix = np.block([[bb, bq], [qb, qq]])
    direction = np.random.default_rng(20261015).normal(size=15)
    direction /= np.linalg.norm(direction)
    expected = matrix @ direction
    rows = []
    for step in [1e-8, 3e-9, 1e-9, 3e-10, 1e-10, 3e-11, 1e-11]:
        plus = system.evaluate_rigid(pose + step * direction[:6], variable + step * direction[6:], previous, False)
        minus = system.evaluate_rigid(pose - step * direction[:6], variable - step * direction[6:], previous, False)
        observed = (np.r_[plus[0], plus[1]] - np.r_[minus[0], minus[1]]) / (2 * step)
        error = float(np.linalg.norm(observed - expected) / np.linalg.norm(expected))
        rows.append({"perturbation": step, "relative_tangent_error": error})
    print(json.dumps({"tangent_qualification": rows}), flush=True)
    assert rows[-1]["relative_tangent_error"] < 1e-3, rows
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
    parser.add_argument("--held-closure-run", type=Path)
    parser.add_argument("--captured-initial-velocity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    static, frame = build_from_capture(args.capture, args.dhat)
    report = json.loads((args.capture / "results/summary.json").read_text())
    system = RigidEggArticulatedContact(static, frame, report, args.mjcf, args.dt)
    assert np.all(system.robot.model.eq_solref[:, 0] >= 2 * args.dt)
    geometry_error = qualify_kinematics(system)
    pose, q = np.zeros(6, dtype=np.longdouble), system.initial_q.copy()
    previous = system.pack(system.body_kinematics(pose)[0], q)[0]
    tangents = []
    closure_identity = None
    if args.held_closure_run is not None:
        closed_source(args.held_closure_run)
        parent_config = json.loads((args.held_closure_run / "results/configuration.json").read_text())
        for key in ["capture", "dt", "dhat", "control_steps"]:
            assert parent_config[key] == vars(args)[key] or parent_config[key] == str(vars(args)[key])
        checkpoint = args.held_closure_run / "results/closure_state.npz"
        if not checkpoint.exists():
            # A failed post-initialization diagnostic may still retain the
            # complete accepted64-target held closure. Never reuse a failed state.
            checkpoint = args.held_closure_run / "results/last_accepted_state.npz"
        with np.load(checkpoint) as saved:
            if "step" in saved:
                assert int(saved["step"]) == args.control_steps
            pose, q, previous = [saved[k].copy() for k in ["body_pose", "joint_position", "full_displacement"]]
        np.testing.assert_array_equal(pose, np.zeros(6))
        np.testing.assert_allclose(system.pack(system.body_kinematics(pose)[0], q)[0], previous, rtol=0, atol=1e-12)
        system.target[7:] = 0.018
        closure_identity = {"source": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    for step in range(1, args.control_steps + 1) if args.held_closure_run is None else ():
        system.target[7:] = 0.04 + (0.018 - 0.04) * step / args.control_steps
        try:
            pose, q, full, contact = system.solve_rigid(pose, q, previous, True, f"closure{step:04d}")
        except Exception:
            np.savez_compressed(
                args.output / "failure.npz", **system.failure_state, previous_full_displacement=previous
            )
            raise
        previous = full.copy()
        np.savez_compressed(
            args.output / "last_accepted_state.npz", body_pose=pose, joint_position=q, full_displacement=full, step=step
        )
    if args.check_jacobian:
        tangents.append({"phase": "held", "rows": qualify_tangent(system, pose, q, previous)})
    np.savez_compressed(
        args.output / "closure_state.npz",
        body_pose=pose,
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
    body_velocity, velocity, motion = initial_motion(static, previous, q, frame, args.captured_initial_velocity)
    np.savez_compressed(args.output / "initial_velocity.npz", body_velocity=body_velocity, joint_velocity=velocity)
    rows = []
    for step in range(1, args.steps + 1):
        old_q, old_velocity, old_full = q.copy(), velocity.copy(), previous.copy()
        old_body_velocity = body_velocity.copy()
        system.begin_dynamic_step(q, velocity, previous[: system.n] + args.dt * body_velocity)
        if args.check_jacobian and step == 1:
            tangents.append({"phase": "dynamic", "rows": qualify_tangent(system, pose, velocity, previous)})
        try:
            pose, velocity, full, contact = system.solve_rigid(pose, velocity, previous, False, f"dynamic{step:04d}")
            direct = system.verify_robot_step(velocity, contact)
        except Exception:
            np.savez_compressed(
                args.output / "failure.npz",
                **system.failure_state,
                previous_full_displacement=old_full,
                previous_joint_position=old_q,
                previous_joint_velocity=old_velocity,
                previous_body_velocity=old_body_velocity,
            )
            raise
        q = system.previous_q + np.longdouble(args.dt) * velocity
        body_velocity = (full[: system.n] - previous[: system.n]) / args.dt
        acceleration = (body_velocity - old_body_velocity) / args.dt
        body_balance = system.contact.body_mass @ acceleration - system.gravity[: system.n] + contact[: system.n]
        current_relative = (
            static.rest[: system.n // 3] + full[: system.n].reshape(-1, 3) - system.body_centre - pose[:3]
        )
        force = body_balance.reshape(-1, 3).sum(axis=0)
        moment = np.cross(current_relative, body_balance.reshape(-1, 3)).sum(axis=0)
        assert np.linalg.norm(force) <= 1e-9 and np.linalg.norm(moment) / system.radius <= 1e-9
        row = {
            "step": step,
            "time_s": step * args.dt,
            "fresh_robot_step_effort_residual": direct,
            "body_resultant_residual_N": float(np.linalg.norm(force)),
            "body_moment_residual_Nm": float(np.linalg.norm(moment)),
            **system.history[-1],
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=full,
            body_pose=pose,
            joint_position=q,
            joint_velocity=velocity,
            body_velocity=body_velocity,
            body_acceleration=acceleration,
            contact_force_N=-contact,
            applied_force_N=system.gravity,
            body_component_permutation=static.parts[0].permutation,
        )
        previous = full.copy()
        print(json.dumps(row), flush=True)
    summary = {
        "scope": __doc__,
        "reused_held_closure": closure_identity,
        "initial_motion": motion,
        "steps": rows,
        "all_accepted_residuals": system.history,
        "body_radius_m": system.radius,
        "kinematic_directional_error": geometry_error,
        "tangent_qualifications": tangents,
        "controller_target": system.target.tolist(),
        "controller_kp": system.kp.tolist(),
        "controller_kv": system.kv.tolist(),
        "controller_caps": system.caps.tolist(),
        "robot_solver_tolerance": system.robot.model.opt.tolerance,
        "precise_kinematic_max_discrepancy_m": system.kinematic_max_discrepancy_m,
        "goal_completion": False,
        "limitations": "Rigid-body motion restricts the same collision surface and consistent mass, preserving actual robot control/contact laws. Body rotational residuals are scaled by the fixed reference radius only in the numerical merit. Raw force/moment and fresh robot-step gates are explicit. Initial velocities are explicitly declared; the held initializer differs from original approach/lift/support history. Spatial/contact/regularization convergence and independent balance/recovery replay remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"steps": len(rows), "body_radius_m": system.radius}), flush=True)


if __name__ == "__main__":
    main()
