"""Recheck an unchanged direct Newton candidate against the original physical acceptance budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import GroundedOriginalContact, build_original, contact_groups
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    config = json.loads((args.run / "results/configuration.json").read_text())
    files = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
    assert files and all(path.name == f"step{index:05d}.npz" for index, path in enumerate(files, 1))
    step = len(files) + 1
    candidate_path = args.run / f"globalization/original{step:05d}/direct_failure.npz"
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, capture, _ = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    with np.load(args.run / "results/reference_geometry.npz") as saved:
        np.testing.assert_array_equal(saved["rest_positions"], static.rest)
        np.testing.assert_array_equal(saved["faces"], static.faces)
    with np.load(args.run / "results/failure.npz") as saved:
        old = {key: saved[key].copy() for key in saved.files}
    with np.load(files[-1]) as saved:
        np.testing.assert_array_equal(old["previous_full_displacement"], saved["displacement"])
        np.testing.assert_array_equal(old["previous_joint_position"], saved["joint_position"])
        np.testing.assert_array_equal(old["previous_joint_velocity"], saved["joint_velocity"])
        np.testing.assert_array_equal(old["previous_body_velocity"], saved["body_velocity"])
    with np.load(candidate_path) as saved:
        candidate = {key: saved[key].copy() for key in saved.files}
    system = GroundedOriginalContact(static, frame, capture, args.mjcf, config["dt"])
    system.target[:], system.caps[:] = old["target"], old["caps"]
    system.begin_dynamic_step(
        old["previous_joint_position"],
        old["previous_joint_velocity"],
        old["previous_body_displacement"] + config["dt"] * old["previous_body_velocity"],
    )
    body, velocity = candidate["body_displacement"], candidate["robot_variable"]
    rb, rj, full, contact, _, _, _ = system.evaluate(body, velocity, old["previous_full_displacement"], False)
    np.testing.assert_array_equal(full, candidate["full_displacement"])
    np.testing.assert_allclose(-contact, candidate["contact_force_N"], rtol=1e-12, atol=1e-14)
    body_error, joint_error = float(np.linalg.norm(rb)), float(np.linalg.norm(rj))
    assert body_error <= 1e-9 and joint_error <= 1e-9
    fresh = system.verify_robot_step(velocity, contact)
    groups, _ = contact_groups(system, full)
    args.output.mkdir(parents=True, exist_ok=False)
    body_velocity = (body - old["previous_body_displacement"]) / config["dt"]
    np.savez_compressed(
        args.output / "recovered.npz",
        displacement=full,
        joint_position=old["previous_joint_position"] + np.longdouble(config["dt"]) * velocity,
        joint_velocity=velocity,
        body_velocity=body_velocity,
        body_acceleration=(body_velocity - old["previous_body_velocity"]) / config["dt"],
        contact_force_N=-contact,
        applied_force_N=system.gravity,
        controller_target=system.target,
        controller_caps=system.caps,
    )
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "source_candidate": str(candidate_path),
        "source_candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "step": step,
        "final_fraction": 1.0,
        "original_dt_s": config["dt"],
        "recovered_original_equation": True,
        "physical_previous_state_fixed": True,
        "no_newton_or_continuation_steps_taken": True,
        "internal_newton_target_N": 1e-10,
        "original_physical_force_budget_N": 1e-9,
        "body_residual_N": body_error,
        "actual_robot_step_effort_residual": joint_error,
        "fresh_robot_step_effort_residual": fresh,
        "contact_groups": groups,
        "attempts": [
            {
                "operation": "Re-evaluate the saved direct candidate; no Newton or artificial continuation step",
                "accepted": True,
                "fraction": 1.0,
                "fresh_robot_effort_residual": fresh,
                "residuals": {
                    "phase": f"original{step:05d}_saved_direct_candidate",
                    "iteration": None,
                    "body_residual_N": body_error,
                    "joint_residual_N_or_Nm": joint_error,
                    "normal_stencils": len(groups),
                },
            }
        ],
        "goal_completion": False,
        "limitations": "This re-evaluates the unchanged direct fraction-one candidate saved before artificial residual continuation. It does not accept a defect-continuation stage or change the solver's internal target. Original physical1e-9 body/actual-robot/fresh-step gates are enforced separately. Full independent control/force/state auditing and the remaining original history are still required.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
