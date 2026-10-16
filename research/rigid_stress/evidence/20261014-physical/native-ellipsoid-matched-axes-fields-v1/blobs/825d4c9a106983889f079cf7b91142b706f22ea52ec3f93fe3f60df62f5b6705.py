"""Use a matched deformable state's rigid frame only as a numerical guess for the unchanged original rigid equation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.spatial.transform import Rotation

from . import recover_original_rigid_step as driver
from .audit_objective_frame import finite_frame
from .original_franka_ground import contact_groups
from .run_original_rigid_franka import qualify_physical_motion
from .summarize_fine_references import closed_source

RIGID_RUN = None
DEFORMABLE_RUN = None


def recover_equation(system, pose, variable, previous, directory):
    directory.mkdir(parents=True, exist_ok=False)
    closed_source(DEFORMABLE_RUN)
    assert int((DEFORMABLE_RUN / "returncode.txt").read_text()) == 0
    config = json.loads((RIGID_RUN / "results/configuration.json").read_text())
    assert Path(config["matched_deformable"]).resolve() == DEFORMABLE_RUN.resolve()
    assert (DEFORMABLE_RUN / "results/reference_geometry.npz").read_bytes() == (
        RIGID_RUN / "results/reference_geometry.npz"
    ).read_bytes()
    paths = sorted((RIGID_RUN / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
    source = DEFORMABLE_RUN / f"results/step{len(paths) + 1:05d}.npz"
    with np.load(source) as saved:
        u = saved["displacement"].copy()
        seed_q, seed_velocity = saved["joint_position"].copy(), saved["joint_velocity"].copy()
    with np.load(paths[-1]) as saved:
        old_pose = saved["body_pose"].copy()
        np.testing.assert_array_equal(saved["displacement"], previous)
    reference = system.contact.body
    vector = np.empty(reference.basis.N)
    vector[system.contact.parts[0].permutation] = u[: system.n]
    rotation, translation, local = finite_frame(reference, vector)
    fitted_pose = np.r_[
        translation + (rotation - np.eye(3)) @ reference.com, Rotation.from_matrix(rotation).as_rotvec()
    ]
    scales = np.r_[np.full(3, 1e-4), np.full(3, 1e-2), np.full(9, 0.1)].astype(np.longdouble)
    fixed = {
        key: np.asarray(getattr(system, key)).copy() for key in ["previous_q", "previous_velocity", "target", "caps"]
    }
    predictor, dt = system.contact.predictor.copy(), system.dt
    mesh = system.contact.mesh
    start = mesh.vertices(system.contact.rest + previous.reshape(-1, 3))
    rows, candidate = [], None
    for label, seed_v in [
        ("matched_joint_position", (seed_q - system.previous_q) / dt),
        ("matched_joint_velocity", seed_velocity),
        ("rejected_joint_iterate", variable),
    ]:
        for exponent in range(21):
            alpha = 0.5**exponent
            seed_pose = old_pose + alpha * (fitted_pose - old_pose)
            initial_v = alpha * seed_v
            initial = system.evaluate_rigid(seed_pose, initial_v, previous, False)
            endpoint = mesh.vertices(system.contact.rest + initial[2].reshape(-1, 3))
            initial_motion = qualify_physical_motion(mesh, start, endpoint)
            if initial_motion["passed"]:
                break
        else:
            rows.append({"seed": label, "no_feasible_numerical_initializer": True})
            continue
        origin = np.r_[seed_pose, initial_v].astype(np.longdouble)

        def evaluate(z, hessian=False, centre=origin):
            x = centre + scales * z
            evaluated = system.evaluate_rigid(x[:6], x[6:], previous, hessian)
            value = np.asarray(np.r_[evaluated[0], evaluated[1]], dtype=float)
            jacobian = None
            if hessian:
                bb, bq, qb, qq = evaluated[-1]
                jacobian = np.asarray(np.block([[bb, bq], [qb, qq]]) * scales, dtype=float)
            return x, value, jacobian, evaluated

        solved = root(
            lambda z: evaluate(z)[1],
            np.zeros(15),
            jac=lambda z: evaluate(z, True)[2],
            method="hybr",
            options={"xtol": 1e-11, "maxfev": 600},
        )
        x, value, _, evaluated = evaluate(solved.x)
        body, robot = [float(np.linalg.norm(r)) for r in [value[:6], value[6:]]]
        motion = {"passed": False}
        fresh = None
        if max(body, robot) <= 1e-9:
            endpoint = mesh.vertices(system.contact.rest + evaluated[2].reshape(-1, 3))
            motion = qualify_physical_motion(mesh, start, endpoint)
            if motion["passed"]:
                fresh = system.verify_robot_step(x[6:], evaluated[3])
                contact_groups(system, evaluated[2])
                candidate = (x[:6].copy(), x[6:].copy(), evaluated[2], evaluated[3])
        row = {
            "seed": label,
            "initializer_scale": alpha,
            "initial_motion_CCD": initial_motion,
            "solver_reported_success": bool(solved.success),
            "message": str(solved.message),
            "body_force_equivalent_residual_N": body,
            "robot_effort_residual": robot,
            "original_motion_CCD": motion,
            "fresh_actual_robot_effort_residual": fresh,
            "original_physical_gates_passed": candidate is not None,
        }
        rows.append(row)
        np.savez_compressed(
            directory / f"attempt{len(rows):02d}.npz", unknown=x, original_residual=value, initial_unknown=origin
        )
        print(json.dumps(row), flush=True)
        if candidate is not None:
            break
    for key, value in fixed.items():
        np.testing.assert_array_equal(getattr(system, key), value)
    assert system.dt == dt
    np.testing.assert_array_equal(system.contact.predictor, predictor)
    report = {
        "scope": __doc__,
        "deformable_seed_file": str(source),
        "deformable_seed_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fitted_seed_pose": fitted_pose.tolist(),
        "seed_objective_maximum_displacement_m": float(np.linalg.norm(local[reference.component_dofs], axis=1).max()),
        "original_previous_state_controls_dt_predictor_fixed": True,
        "seed_is_not_prescribed_physical_motion_or_force": True,
        "attempts": rows,
        "recovered_original_equation": candidate is not None,
        "numerical_iterations_are_physical_history": False,
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert candidate is not None, report
    return candidate, report


def main():
    global RIGID_RUN, DEFORMABLE_RUN
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--deformable-seed", type=Path, required=True)
    args, rest = parser.parse_known_args()
    RIGID_RUN, DEFORMABLE_RUN = args.run, args.deformable_seed
    sys.argv = [sys.argv[0], "--run", str(args.run), *rest]
    driver.recover_equation = recover_equation
    driver.main()


if __name__ == "__main__":
    main()
