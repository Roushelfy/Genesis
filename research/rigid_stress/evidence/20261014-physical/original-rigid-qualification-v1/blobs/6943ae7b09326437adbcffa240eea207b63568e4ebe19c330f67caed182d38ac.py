"""Recover a rejected rigid counterpart equation without changing physical time or its audited previous state."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import ipctk
import numpy as np
from scipy.optimize import least_squares, root

from .original_franka_ground import build_original, contact_groups
from .run_original_rigid_franka import OriginalRigidEggContact
from .summarize_fine_references import closed_source


def recover_equation(system, pose, variable, previous, directory):
    directory.mkdir(parents=True, exist_ok=False)
    x0 = np.asarray(np.r_[pose, variable], dtype=float)
    fixed = {
        key: np.asarray(getattr(system, key)).copy() for key in ["previous_q", "previous_velocity", "target", "caps"]
    }
    dt, predictor = system.dt, system.contact.predictor.copy()

    def residual(x):
        physical = np.asarray(x, dtype=np.longdouble)
        result = system.evaluate_rigid(physical[:6], physical[6:], previous, False)
        return np.asarray(np.r_[result[0], result[1]], dtype=float)

    def difference_jacobian(function, x, steps):
        columns = []
        for index, step in enumerate(steps):
            direction = np.zeros_like(x)
            direction[index] = step
            columns.append((function(x + direction) - function(x - direction)) / (2 * step))
        return np.column_stack(columns)

    initial = system.evaluate_rigid(pose, variable, previous, True)
    bb, bq, qb, qq = initial[-1]
    analytic = np.block([[bb, bq], [qb, qq]])
    observations = []
    for step in [1e-4, 1e-5, 1e-6, 1e-7]:
        numerical = difference_jacobian(lambda y: residual(np.r_[x0[:6], y])[6:], x0[6:], np.full(9, step))
        observations.append(
            {
                "velocity_perturbation": step,
                "robot_block_relative_difference": float(np.linalg.norm(numerical - qq) / np.linalg.norm(numerical)),
                "numerical_robot_condition": float(np.linalg.cond(numerical)),
            }
        )
    attempts, candidate = [], None

    def check(x, label, success, message):
        nonlocal candidate
        physical = np.asarray(x, dtype=np.longdouble)
        result = system.evaluate_rigid(physical[:6], physical[6:], previous, False)
        body, joint = [float(np.linalg.norm(value)) for value in result[:2]]
        row = {
            "method": label,
            "solver_reported_success": bool(success),
            "message": str(message),
            "body_residual_N": body,
            "joint_effort_residual": joint,
        }
        valid = max(body, joint) <= 1e-9
        if valid:
            start = system.contact.mesh.vertices(system.contact.rest + previous.reshape(-1, 3))
            end = system.contact.mesh.vertices(system.contact.rest + result[2].reshape(-1, 3))
            row["original_step_CCD_fraction"] = float(
                ipctk.compute_collision_free_stepsize(system.contact.mesh, start, end)
            )
            valid = row["original_step_CCD_fraction"] >= 1 - 1e-12 and not ipctk.has_intersections(
                system.contact.mesh, end
            )
            if valid:
                row["fresh_actual_robot_effort_residual"] = system.verify_robot_step(physical[6:], result[3])
                contact_groups(system, result[2])
                candidate = (
                    np.asarray(x[:6], dtype=np.longdouble),
                    np.asarray(x[6:], dtype=np.longdouble),
                    result[2],
                    result[3],
                )
        row["original_physical_gates_passed"] = valid
        attempts.append(row)
        np.savez_compressed(
            directory / f"attempt{len(attempts):02d}.npz", unknown=x, unmodified_residual=np.r_[result[0], result[1]]
        )

    robot_function = lambda y: residual(np.r_[x0[:6], y])[6:]
    robot_jacobian = lambda y: difference_jacobian(robot_function, y, np.full(9, 1e-6))
    for method in ["hybr", "lm"]:
        try:
            solved = root(robot_function, x0[6:], jac=robot_jacobian, method=method, options={"xtol": 1e-12})
            check(np.r_[x0[:6], solved.x], "robot_only_" + method, solved.success, solved.message)
        except (RuntimeError, AssertionError, ValueError) as error:
            attempts.append(
                {"method": "robot_only_" + method, "error": str(error), "original_physical_gates_passed": False}
            )
        if candidate is not None:
            break
    if candidate is None:
        steps = np.r_[np.full(3, 1e-10), np.full(3, 1e-8), np.full(9, 1e-6)]
        solved = least_squares(
            residual,
            x0,
            jac=lambda x: difference_jacobian(residual, x, steps),
            x_scale="jac",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
            max_nfev=500,
        )
        check(solved.x, "full15_numerical_TRF", solved.success, solved.message)
    for key, expected in fixed.items():
        np.testing.assert_array_equal(getattr(system, key), expected)
    assert system.dt == dt
    np.testing.assert_array_equal(system.contact.predictor, predictor)
    record = {
        "original_dt_s": dt,
        "physical_previous_state_fixed": True,
        "initial_body_residual_N": float(np.linalg.norm(initial[0])),
        "initial_joint_effort_residual": float(np.linalg.norm(initial[1])),
        "analytic_matrix_condition": float(np.linalg.cond(analytic)),
        "robot_tangent_observations": observations,
        "attempts": attempts,
        "recovered_original_equation": candidate is not None,
        "numerical_iterations_are_physical_history": False,
    }
    (directory / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    assert candidate is not None, record
    return candidate, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prefix-audit", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.prefix_audit]:
        closed_source(run)
    assert (
        int((args.run / "returncode.txt").read_text()) != 0
        and int((args.prefix_audit / "returncode.txt").read_text()) == 0
    )
    audit = json.loads((args.prefix_audit / "results/summary.json").read_text())
    assert Path(audit["run"]).resolve() == args.run.resolve() and not audit["complete_requested_history"]
    count = audit["states_checked"]
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, capture, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    system = OriginalRigidEggContact(static, frame, capture, args.mjcf, config["dt"])
    with np.load(args.run / f"results/step{count:05d}.npz") as saved:
        previous, q, velocity, body_velocity = [
            saved[key].copy() for key in ["displacement", "joint_position", "joint_velocity", "body_velocity"]
        ]
    with np.load(args.run / "results/failure.npz") as saved:
        pose, variable = saved["body_pose"].copy(), saved["robot_variable"].copy()
        np.testing.assert_array_equal(saved["previous_full_displacement"], previous)
    substeps = round(protocol["macro_dt_s"] / config["dt"])
    control = [row for row in protocol["records"] if row["mode"] == "approach_lift"][count // substeps]
    system.target = np.array(control["target"])
    system.caps[7:] = control["finger_cap_N"]
    system.begin_dynamic_step(q, velocity, previous[: system.n] + config["dt"] * body_velocity)
    args.output.mkdir(parents=True, exist_ok=False)
    candidate, numerical = recover_equation(system, pose, variable, previous, args.output / "numerical_recovery")
    new_pose, new_velocity, full, contact = candidate
    new_q = q + np.longdouble(config["dt"]) * new_velocity
    new_body_velocity = (full[: system.n] - previous[: system.n]) / config["dt"]
    identities = []
    for original in sorted((args.run / "results").glob("*.npz")):
        if original.name not in ["failure.npz"]:
            os.link(original, args.output / original.name)
            identities.append({"source": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
    config.update(output=str(args.output), mjcf=str(args.mjcf))
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(
        args.output / f"step{count + 1:05d}.npz",
        body_pose=new_pose,
        displacement=full,
        joint_position=new_q,
        joint_velocity=new_velocity,
        body_velocity=new_body_velocity,
        body_acceleration=(new_body_velocity - body_velocity) / config["dt"],
        contact_force_N=-contact,
        body_component_permutation=static.parts[0].permutation,
        applied_force_N=system.gravity,
    )
    report = {
        "scope": __doc__,
        "source_run": str(args.run),
        "prefix_audit": str(args.prefix_audit),
        "byte_identical_prefix_states": count,
        "recovered_step": count + 1,
        "original_state_identities": identities,
        "numerical_recovery": numerical,
        "complete_requested_history": False,
        "goal_completion": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
