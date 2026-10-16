"""Recover the original rigid equation by numerical residual continuation; no intermediate state is physical time."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

from . import recover_original_rigid_step as driver
from .original_franka_ground import contact_groups
from .run_original_rigid_franka import qualify_physical_motion

ANCHOR = None


def recover_equation(system, pose, variable, previous, directory):
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(ANCHOR) as saved:
        np.testing.assert_array_equal(saved["displacement"], previous)
        old_pose = saved["body_pose"].copy()
    fixed = {
        key: np.asarray(getattr(system, key)).copy() for key in ["previous_q", "previous_velocity", "target", "caps"]
    }
    predictor, dt = system.contact.predictor.copy(), system.dt
    # Zero new velocity recovers exactly the previous collision geometry. It
    # is a numerical anchor, not an alternative physical initial condition.
    anchor = np.r_[old_pose, np.zeros(9)].astype(np.longdouble)

    def evaluate(x, hessian=False):
        wide = np.asarray(x, dtype=np.longdouble)
        return system.evaluate_rigid(wide[:6], wide[6:], previous, hessian)

    def residual(x):
        evaluated = evaluate(x)
        return np.asarray(np.r_[evaluated[0], evaluated[1]], dtype=float)

    offset = residual(anchor)
    mesh = system.contact.mesh
    start = mesh.vertices(system.contact.rest + previous.reshape(-1, 3))
    rows, candidate = [], None
    total_evaluations = 0
    for perturbation in [1.0, 0.1]:
        x = np.asarray(anchor, dtype=float)
        position, increment = 0.0, 0.125
        stage = 0
        while position < 1 and stage < 240:
            target = min(1.0, position + increment)

            def equation(y, stage_target=target):
                nonlocal total_evaluations
                total_evaluations += 1
                return residual(y) - (1 - stage_target) * offset

            def numerical_jacobian(y, difference_scale=perturbation, equation_function=equation):
                steps = difference_scale * np.r_[np.full(3, 1e-12), np.full(3, 1e-10), np.full(9, 1e-8)]
                columns = []
                for column, step in enumerate(steps):
                    direction = np.eye(15)[column] * step
                    columns.append((equation_function(y + direction) - equation_function(y - direction)) / (2 * step))
                return np.column_stack(columns)

            solved = root(equation, x, jac=numerical_jacobian, method="hybr", options={"xtol": 1e-11, "maxfev": 300})
            stage_error = float(np.linalg.norm(equation(solved.x)))
            result = evaluate(solved.x)
            endpoint = mesh.vertices(system.contact.rest + result[2].reshape(-1, 3))
            motion = qualify_physical_motion(mesh, start, endpoint) if stage_error <= 1e-8 else {"passed": False}
            accepted = stage_error <= (1e-9 if target == 1 else 1e-8) and motion["passed"]
            row = {
                "numerical_stage": stage,
                "residual_continuation_parameter": target,
                "finite_difference_scale": perturbation,
                "solver_reported_success": bool(solved.success),
                "message": str(solved.message),
                "artificial_equation_residual": stage_error,
                "original_body_residual_N": float(np.linalg.norm(result[0])),
                "original_robot_effort_residual": float(np.linalg.norm(result[1])),
                "original_motion_CCD": motion,
                "numerical_stage_accepted": accepted,
                "is_physical_history_state": False,
            }
            rows.append(row)
            np.savez_compressed(directory / f"stage{len(rows):04d}.npz", unknown=solved.x, offset=(1 - target) * offset)
            print(json.dumps(row), flush=True)
            stage += 1
            if accepted:
                x, position = solved.x, target
                if target == 1:
                    fresh = system.verify_robot_step(np.asarray(x[6:], dtype=np.longdouble), result[3])
                    contact_groups(system, result[2])
                    assert max(np.linalg.norm(result[0]), np.linalg.norm(result[1]), fresh) <= 1e-9
                    candidate = (
                        np.asarray(x[:6], dtype=np.longdouble),
                        np.asarray(x[6:], dtype=np.longdouble),
                        result[2],
                        result[3],
                    )
                    row["final_original_physical_gates_passed"] = True
                    row["fresh_actual_robot_effort_residual"] = fresh
                    break
                increment = min(0.25, increment * 1.5)
            else:
                increment *= 0.5
                if increment < 2**-18:
                    break
        if candidate is not None:
            break
    for key, value in fixed.items():
        np.testing.assert_array_equal(getattr(system, key), value)
    assert system.dt == dt
    np.testing.assert_array_equal(system.contact.predictor, predictor)
    report = {
        "scope": __doc__,
        "anchor_file": str(ANCHOR),
        "anchor_sha256": hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
        "original_dt_s": dt,
        "original_previous_state_and_controls_fixed": True,
        "artificial_offset_removed_at_final_state": candidate is not None,
        "numerical_iterations_are_physical_history": False,
        "original_residual_evaluations": total_evaluations,
        "attempts": rows,
        "recovered_original_equation": candidate is not None,
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert candidate is not None, report
    return candidate, report


def main():
    global ANCHOR
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--homotopy-anchor", type=Path, required=True)
    args, rest = parser.parse_known_args()
    ANCHOR = args.homotopy_anchor
    sys.argv = [sys.argv[0], *rest]
    # The original driver retains its unchanged configuration/audit checks,
    # state writer and byte-identical prefix linking; only numerical recovery
    # is substituted in this isolated research process.
    driver.recover_equation = recover_equation
    driver.main()


if __name__ == "__main__":
    main()
