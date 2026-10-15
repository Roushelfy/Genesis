"""Recover the original rigid equation with lagged-friction numerical stages and unchanged final current-force gates."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ipctk
import numpy as np

from . import recover_original_rigid_step as driver
from .original_franka_ground import contact_groups
from .run_original_rigid_franka import qualify_physical_motion

ANCHOR = None


def recover_equation(system, pose, variable, previous, directory):
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(ANCHOR) as saved:
        np.testing.assert_array_equal(saved["displacement"], previous)
        pose = saved["body_pose"].copy()
    variable = np.zeros(9, dtype=np.longdouble)
    fixed = {
        key: np.asarray(getattr(system, key)).copy() for key in ["previous_q", "previous_velocity", "target", "caps"]
    }
    predictor, dt = system.contact.predictor.copy(), system.dt
    contact, mesh = system.contact, system.contact.mesh
    current_force_evaluation = contact.evaluate_updated_friction
    start = mesh.vertices(contact.rest + previous.reshape(-1, 3))
    rows, candidate = [], None
    for stage in range(120):
        full = system.pack(system.body_kinematics(pose)[0], system.previous_q + np.longdouble(dt) * variable)[0]
        normal, positions = contact.normal_collisions(full)
        frozen = ipctk.TangentialCollisions()
        frozen.build(mesh, positions, normal, contact.barrier, contact.friction)

        def lagged(displacement, old, external_force, hessian, frozen_friction=frozen):
            return contact.evaluate(displacement, old, frozen_friction, hessian=hessian, external_force=external_force)

        contact.evaluate_updated_friction = lagged
        try:
            solved = system.solve_rigid(pose.copy(), variable.copy(), previous, False, f"lagged_stage{stage:03d}")
        except (RuntimeError, AssertionError, ValueError) as error:
            rows.append(
                {
                    "stage": stage,
                    "inner_equation_solved": False,
                    "error": str(error),
                    "is_physical_history_state": False,
                }
            )
            break
        finally:
            contact.evaluate_updated_friction = current_force_evaluation
        new_pose, new_variable = solved[:2]
        true = system.evaluate_rigid(new_pose, new_variable, previous, False)
        body, robot = [float(np.linalg.norm(value)) for value in true[:2]]
        end = mesh.vertices(contact.rest + true[2].reshape(-1, 3))
        motion = qualify_physical_motion(mesh, start, end)
        row = {
            "stage": stage,
            "inner_equation_solved": True,
            "original_current_body_residual_N": body,
            "original_current_robot_effort_residual": robot,
            "original_motion_CCD": motion,
            "frozen_friction_stencils": len(frozen),
            "is_physical_history_state": False,
            "original_physical_gates_passed": False,
        }
        rows.append(row)
        np.savez_compressed(
            directory / f"stage{stage + 1:04d}.npz",
            body_pose=new_pose,
            robot_variable=new_variable,
            original_current_residual=np.r_[true[0], true[1]],
            contact_force_N=-true[3],
        )
        print(json.dumps(row), flush=True)
        if max(body, robot) <= 1e-9 and motion["passed"]:
            fresh = system.verify_robot_step(new_variable, true[3])
            contact_groups(system, true[2])
            candidate = (new_pose.copy(), new_variable.copy(), true[2], true[3])
            row["original_physical_gates_passed"] = True
            row["fresh_actual_robot_effort_residual"] = fresh
            break
        accepted = False
        for exponent in range(1, 22):
            relaxation = 0.5**exponent
            trial_pose = pose + relaxation * (new_pose - pose)
            trial_variable = variable + relaxation * (new_variable - variable)
            trial = system.evaluate_rigid(trial_pose, trial_variable, previous, False)
            end = mesh.vertices(contact.rest + trial[2].reshape(-1, 3))
            if qualify_physical_motion(mesh, start, end)["passed"]:
                pose, variable = trial_pose, trial_variable
                row["numerical_relaxation"] = relaxation
                accepted = True
                break
        if not accepted:
            row["no_feasible_relaxed_numerical_iterate"] = True
            break
    assert contact.evaluate_updated_friction == current_force_evaluation
    for key, value in fixed.items():
        np.testing.assert_array_equal(getattr(system, key), value)
    assert system.dt == dt
    np.testing.assert_array_equal(contact.predictor, predictor)
    report = {
        "scope": __doc__,
        "anchor": str(ANCHOR),
        "anchor_sha256": hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
        "original_previous_state_controls_dt_predictor_fixed": True,
        "attempts": rows,
        "recovered_original_equation": candidate is not None,
        "final_acceptance_uses_original_current_friction": True,
        "numerical_iterations_are_physical_history": False,
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert candidate is not None, report
    return candidate, report


def main():
    global ANCHOR
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lagged-anchor", type=Path, required=True)
    args, rest = parser.parse_known_args()
    ANCHOR = args.lagged_anchor
    sys.argv = [sys.argv[0], *rest]
    driver.recover_equation = recover_equation
    driver.main()


if __name__ == "__main__":
    main()
