"""Test wide-coordinate Newton recovery on a saved failed state without changing its physical inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .franka_contact_reference import build_from_capture
from .run_articulated_franka import check_tangent
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WideArticulatedContact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "results/configuration.json").read_text())
    capture = Path(config["capture"])
    static, frame = build_from_capture(capture, config["dhat"])
    report = json.loads((capture / "results/summary.json").read_text())
    system = WideArticulatedContact(static, frame, report, config["mjcf"], config["dt"])
    system.target[7:] = 0.018
    with np.load(args.run / "results/failure.npz") as saved:
        values = {key: saved[key].copy() for key in saved.files}
    previous = values["previous_full_displacement"]
    predictor = np.asarray(values["previous_body_displacement"], dtype=np.longdouble)
    predictor += config["dt"] * np.asarray(values["previous_body_velocity"], dtype=np.longdouble)
    system.begin_dynamic_step(values["previous_joint_position"], values["previous_joint_velocity"], predictor)
    body_u = np.asarray(values["body_displacement"], dtype=np.longdouble)
    velocity = values["robot_variable"].copy()
    tangent = check_tangent(system, body_u, velocity, previous)
    initial = system.evaluate(body_u, velocity, previous, False)
    result = {
        "initial_body_residual_N": float(np.linalg.norm(initial[0])),
        "tangent": tangent,
        "internal_target_N": 1e-10,
        "original_external_budget_N": 1e-9,
        "goal_completion": False,
    }
    try:
        body_u, velocity, full, contact = system.solve(body_u, velocity, previous, False, "wide_saved_state")
        result["recovered"] = True
        result["residuals"] = system.history[-1]
        result["fresh_robot_step_effort_residual"] = system.verify_robot_step(velocity, contact)
        result["body_coordinate_mantissa_bits"] = np.finfo(body_u.dtype).nmant
        np.savez_compressed(
            args.output / "recovered.npz",
            body_displacement=body_u,
            joint_velocity=velocity,
            displacement=full,
            contact_force_N=-contact,
        )
    except RuntimeError as error:
        result["recovered"] = False
        result["recovery_error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
