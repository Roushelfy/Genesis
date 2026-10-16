"""Compare original and extended force assembly at an archived rejected dynamic state."""

import argparse
import json
from pathlib import Path

import numpy as np

from .articulated_franka_contact import ArticulatedContact
from .extended_piola_reference import ExtendedPiolaBulk
from .franka_contact_reference import build_from_capture
from .summarize_fine_references import closed_source


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
    report = json.loads((capture / "results/summary.json").read_text())
    static, frame = build_from_capture(capture, config["dhat"])
    system = ArticulatedContact(static, frame, report, config["mjcf"], config["dt"])
    system.target[7:] = 0.018
    with np.load(args.run / "results/failure.npz") as saved:
        values = {key: saved[key].copy() for key in saved.files}
    previous = values["previous_full_displacement"]
    predictor = values["previous_body_displacement"] + config["dt"] * values["previous_body_velocity"]
    system.begin_dynamic_step(values["previous_joint_position"], values["previous_joint_velocity"], predictor)
    original = system.evaluate(values["body_displacement"], values["robot_variable"], previous, False)
    np.testing.assert_array_equal(original[0], values["body_gradient"])
    system.contact.bulk = ExtendedPiolaBulk(static.body)
    extended = system.evaluate(values["body_displacement"], values["robot_variable"], previous, False)
    results = {
        "original_body_residual_N": float(np.linalg.norm(original[0])),
        "extended_body_residual_N": float(np.linalg.norm(extended[0])),
        "force_contraction_difference_N": float(np.linalg.norm(original[0] - extended[0])),
        "force_contraction_max_difference_N": float(abs(original[0] - extended[0]).max()),
        "original_joint_residual": float(np.linalg.norm(original[1])),
        "extended_joint_residual": float(np.linalg.norm(extended[1])),
        "longdouble_mantissa_bits": np.finfo(np.longdouble).nmant,
        "physical_law_quadrature_and_force_budget_changed": False,
        "goal_completion": False,
    }
    try:
        body_u, velocity, full, contact = system.solve(
            values["body_displacement"], values["robot_variable"], previous, False, "same_rejected_state"
        )
        results["recovered"] = True
        results["recovered_residuals"] = system.history[-1]
        results["fresh_robot_step_effort_residual"] = system.verify_robot_step(velocity, contact)
        np.savez_compressed(
            args.output / "recovered.npz",
            body_displacement=body_u,
            joint_velocity=velocity,
            displacement=full,
            contact_force_N=-contact,
        )
    except RuntimeError as error:
        results["recovered"] = False
        results["recovery_error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
