"""Recover an original-prefix state using the actual independent robot step as residual."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import GroundedOriginalContact, build_original
from .run_articulated_franka import check_tangent
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, report, _ = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    with np.load(args.run / "results/reference_geometry.npz") as geometry:
        np.testing.assert_array_equal(geometry["rest_positions"], static.rest)
        np.testing.assert_array_equal(geometry["faces"], static.faces)
    with np.load(args.run / "results/failure.npz") as saved:
        values = {key: saved[key].copy() for key in saved.files}
    system = GroundedOriginalContact(static, frame, report, args.mjcf, config["dt"])
    system.target[:] = values["target"]
    system.caps[:] = values["caps"]
    predictor = values["previous_body_displacement"] + config["dt"] * values["previous_body_velocity"]
    system.begin_dynamic_step(values["previous_joint_position"], values["previous_joint_velocity"], predictor)
    body_u, velocity = values["body_displacement"], values["robot_variable"]
    previous = values["previous_full_displacement"]
    initial = system.evaluate(body_u, velocity, previous, False)
    result = {
        "scope": __doc__,
        "initial_actual_robot_effort_residual": float(np.linalg.norm(initial[1])),
        "saved_joint_effort_residual": float(np.linalg.norm(values["joint_residual"])),
        "tangent": check_tangent(system, body_u, velocity, previous),
        "goal_completion": False,
    }
    evaluated = system.evaluate(body_u, velocity, previous, True)
    bb, bq, qb, qq = evaluated[6]
    modes = static.body.rigid_modes[static.parts[0].permutation].copy()
    modes[:, 3:] /= 0.02
    probes = []
    for index in range(15):
        db = modes[:, index] if index < 6 else np.zeros(system.n)
        dv = np.zeros(9) if index < 6 else np.eye(9)[index - 6]
        expected_b = bb @ db + bq @ dv
        expected_q = qb @ db + qq @ dv
        for step in [1e-8, 3e-9, 1e-9]:
            plus = system.evaluate(body_u + step * db, velocity + step * dv, previous, False)
            minus = system.evaluate(body_u - step * db, velocity - step * dv, previous, False)
            body_difference = (plus[0] - minus[0]) / (2 * step) - expected_b
            joint_difference = (plus[1] - minus[1]) / (2 * step) - expected_q
            probes.append(
                {
                    "direction": index,
                    "perturbation": step,
                    "body_absolute_tangent_difference": float(np.linalg.norm(body_difference)),
                    "body_expected_tangent_norm": float(np.linalg.norm(expected_b)),
                    "joint_absolute_tangent_difference": float(np.linalg.norm(joint_difference)),
                    "joint_expected_tangent_norm": float(np.linalg.norm(expected_q)),
                }
            )
    result["rigid_body_and_joint_tangent_probes"] = probes
    (args.output / "before_recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    body_u, velocity, full, contact = system.solve(body_u, velocity, previous, False, "original_saved_state")
    result["recovered_residuals"] = system.history[-1]
    result["fresh_robot_step_effort_residual"] = system.verify_robot_step(velocity, contact)
    np.savez_compressed(
        args.output / "recovered.npz",
        body_displacement=body_u,
        joint_velocity=velocity,
        displacement=full,
        contact_force_N=-contact,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
