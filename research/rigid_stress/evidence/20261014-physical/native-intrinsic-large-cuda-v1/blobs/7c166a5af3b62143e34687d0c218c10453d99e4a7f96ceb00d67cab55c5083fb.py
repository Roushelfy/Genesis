"""Continue the unchanged original rigid equation in scaled coordinates using its measured analytic tangent."""

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
from .summarize_fine_references import closed_source

SEED_RUN = None


def recover_equation(system, pose, variable, previous, directory):
    directory.mkdir(parents=True, exist_ok=False)
    closed_source(SEED_RUN)
    assert int((SEED_RUN / "returncode.txt").read_text()) != 0
    report_path = SEED_RUN / "results/numerical_recovery/summary.json"
    seed_report = json.loads(report_path.read_text())
    anchor_path = Path(seed_report["anchor_file"])
    assert hashlib.sha256(anchor_path.read_bytes()).hexdigest() == seed_report["anchor_sha256"]
    with np.load(anchor_path) as saved:
        np.testing.assert_array_equal(saved["displacement"], previous)
    index, seed = max(
        ((i + 1, row) for i, row in enumerate(seed_report["attempts"]) if row["numerical_stage_accepted"]),
        key=lambda pair: pair[1]["residual_continuation_parameter"],
    )
    position = seed["residual_continuation_parameter"]
    assert 0 < position < 1
    seed_file = SEED_RUN / f"results/numerical_recovery/stage{index:04d}.npz"
    with np.load(seed_file) as saved:
        origin = np.asarray(saved["unknown"], dtype=np.longdouble)
        offset = saved["offset"] / (1 - position)
    fixed = {
        key: np.asarray(getattr(system, key)).copy() for key in ["previous_q", "previous_velocity", "target", "caps"]
    }
    predictor, dt = system.contact.predictor.copy(), system.dt
    scales = np.r_[np.full(3, 1e-4), np.full(3, 1e-2), np.full(9, 0.1)].astype(np.longdouble)
    mesh = system.contact.mesh
    start = mesh.vertices(system.contact.rest + previous.reshape(-1, 3))
    evaluations = 0

    def evaluate(z, hessian=False):
        nonlocal evaluations
        evaluations += 1
        x = origin + scales * z
        result = system.evaluate_rigid(x[:6], x[6:], previous, hessian)
        value = np.asarray(np.r_[result[0], result[1]], dtype=float)
        matrix = None
        if hessian:
            bb, bq, qb, qq = result[-1]
            matrix = np.asarray(np.block([[bb, bq], [qb, qq]]) * scales, dtype=float)
        return x, value, matrix, result

    z = np.zeros(15)
    assert np.linalg.norm(evaluate(z)[1] - (1 - position) * offset) <= 1e-8
    rows, candidate = [], None
    increment = 0.01
    for stage in range(400):
        target = min(1.0, position + increment)
        matrix = evaluate(z, True)[2]
        tangent = np.linalg.solve(matrix, -offset)
        predicted = z + (target - position) * tangent

        def equation(y, parameter=target):
            return evaluate(y)[1] - (1 - parameter) * offset

        def jacobian(y):
            return evaluate(y, True)[2]

        solved = root(equation, predicted, jac=jacobian, method="hybr", options={"xtol": 1e-11, "maxfev": 400})
        x, value, _, result = evaluate(solved.x)
        error = float(np.linalg.norm(value - (1 - target) * offset))
        endpoint = mesh.vertices(system.contact.rest + result[2].reshape(-1, 3))
        motion = qualify_physical_motion(mesh, start, endpoint) if error <= 1e-8 else {"passed": False}
        accepted = error <= (1e-9 if target == 1 else 1e-8) and motion["passed"]
        row = {
            "numerical_stage": stage,
            "continuation_parameter": target,
            "artificial_equation_residual": error,
            "original_body_residual_N": float(np.linalg.norm(value[:6])),
            "original_robot_effort_residual": float(np.linalg.norm(value[6:])),
            "solver_reported_success": bool(solved.success),
            "message": str(solved.message),
            "scaled_tangent_condition": float(np.linalg.cond(matrix)),
            "motion_CCD": motion,
            "numerical_stage_accepted": accepted,
            "is_physical_history_state": False,
        }
        rows.append(row)
        np.savez_compressed(
            directory / f"stage{stage + 1:04d}.npz",
            unknown=x,
            original_residual=value,
            artificial_offset=(1 - target) * offset,
            scaled_predictor_tangent=tangent,
        )
        print(json.dumps(row), flush=True)
        if accepted:
            z, position = solved.x, target
            if target == 1:
                fresh = system.verify_robot_step(x[6:], result[3])
                contact_groups(system, result[2])
                assert max(np.linalg.norm(value[:6]), np.linalg.norm(value[6:]), fresh) <= 1e-9
                candidate = (x[:6].copy(), x[6:].copy(), result[2], result[3])
                row["fresh_actual_robot_effort_residual"] = fresh
                row["final_original_physical_gates_passed"] = True
                break
            increment = min(0.1, increment * 1.5)
        else:
            increment *= 0.5
            if increment < 1e-8:
                break
    for key, value in fixed.items():
        np.testing.assert_array_equal(getattr(system, key), value)
    assert system.dt == dt
    np.testing.assert_array_equal(system.contact.predictor, predictor)
    report = {
        "scope": __doc__,
        "seed_run": str(SEED_RUN),
        "seed_state": str(seed_file),
        "seed_state_sha256": hashlib.sha256(seed_file.read_bytes()).hexdigest(),
        "seed_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "original_dt_s": dt,
        "original_previous_state_and_controls_fixed": True,
        "original_residual_evaluations": evaluations,
        "numerical_iterations_are_physical_history": False,
        "unknown_scales": scales.astype(float).tolist(),
        "attempts": rows,
        "final_parameter": position,
        "recovered_original_equation": candidate is not None,
        "artificial_offset_removed_at_final_state": candidate is not None,
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert candidate is not None, report
    return candidate, report


def main():
    global SEED_RUN
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--continuation-seed", type=Path, required=True)
    args, rest = parser.parse_known_args()
    SEED_RUN = args.continuation_seed
    sys.argv = [sys.argv[0], *rest]
    driver.recover_equation = recover_equation
    driver.main()


if __name__ == "__main__":
    main()
