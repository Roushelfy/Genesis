"""Continue a residual defect at fixed original dt, accepting only the unmodified final equation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import GroundedOriginalContact, build_original, contact_groups
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WideArticulatedContact


class ResidualContinuationContact(GroundedOriginalContact):
    fraction = 1.0
    defect = None

    def evaluate(self, body_u, variable, previous, hessian):
        values = list(super().evaluate(body_u, variable, previous, hessian))
        if self.defect is not None and self.fraction != 1:
            values[0] = values[0] - (1 - self.fraction) * self.defect[0]
            values[1] = values[1] - (1 - self.fraction) * self.defect[1]
        return tuple(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, report, _ = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    with np.load(args.run / "results/reference_geometry.npz") as saved:
        np.testing.assert_array_equal(saved["rest_positions"], static.rest)
        np.testing.assert_array_equal(saved["faces"], static.faces)
    with np.load(args.run / "results/failure.npz") as saved:
        old = {key: saved[key].copy() for key in saved.files}
    old_body = np.asarray(old["previous_body_displacement"], dtype=np.longdouble)
    previous = old["previous_full_displacement"]
    system = ResidualContinuationContact(static, frame, report, args.mjcf, config["dt"])
    system.target[:], system.caps[:] = old["target"], old["caps"]
    system.begin_dynamic_step(
        old["previous_joint_position"],
        old["previous_joint_velocity"],
        old_body + config["dt"] * old["previous_body_velocity"],
    )
    body, velocity = old_body.copy(), np.zeros(9)
    baseline = system.evaluate(body, velocity, previous, False)
    system.defect = (baseline[0].copy(), baseline[1].copy())
    fraction, increment, attempts = 0.0, 0.0625, []
    while fraction < 1:
        trial = min(1.0, fraction + increment)
        system.fraction = trial
        attempt = {"fraction": trial, "dt_s": config["dt"], "previous_fraction": fraction}
        try:
            candidate = WideArticulatedContact.solve(
                system, body.copy(), velocity.copy(), previous, False, f"residual_fraction_{trial:.8f}"
            )
            groups, _ = contact_groups(system, candidate[2])
            attempt.update(accepted=True, residuals=system.history[-1], contact_groups=groups)
            if trial == 1:
                attempt["fresh_robot_effort_residual"] = system.verify_robot_step(candidate[1], candidate[3])
        except (RuntimeError, AssertionError, ValueError) as error:
            attempt.update(accepted=False, error=str(error))
            np.savez_compressed(args.output / f"attempt{len(attempts):03d}-failure.npz", **system.failure_state)
            attempts.append(attempt)
            increment *= 0.5
            if increment < 1 / 1024:
                break
            continue
        attempts.append(attempt)
        body, velocity, fraction = candidate[0], candidate[1], trial
        np.savez_compressed(
            args.output / f"stage{sum(a['accepted'] for a in attempts):03d}.npz",
            body_displacement=body,
            joint_velocity=velocity,
            displacement=candidate[2],
            contact_force_N=-candidate[3],
            fraction=fraction,
        )
        increment = min(0.25, increment * 1.5)
        if fraction == 1:
            assert system.fraction == 1 and system.dt == config["dt"]
            np.savez_compressed(
                args.output / "recovered.npz",
                displacement=candidate[2],
                joint_position=old["previous_joint_position"] + np.longdouble(config["dt"]) * velocity,
                joint_velocity=velocity,
                body_velocity=(body - old_body) / config["dt"],
                body_acceleration=((body - old_body) / config["dt"] - old["previous_body_velocity"]) / config["dt"],
                contact_force_N=-candidate[3],
                applied_force_N=system.gravity,
                controller_target=system.target,
                controller_caps=system.caps,
            )
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "final_fraction": fraction,
        "original_dt_s": config["dt"],
        "recovered_original_equation": fraction == 1,
        "physical_previous_state_fixed": True,
        "attempts": attempts,
        "initial_body_defect_N": float(np.linalg.norm(system.defect[0])),
        "initial_joint_defect_effort": float(np.linalg.norm(system.defect[1])),
        "goal_completion": False,
        "limitations": "The artificial residual defect is only a Newton globalization device. Intermediate stages are not physical-history states and do not pass an unmodified robot-step gate. Only fraction1 removes the entire defect and requires all original budgets and a fresh robot step. Independent verification, complete original history and physical convergence remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({"final_fraction": fraction, "recovered_original_equation": fraction == 1}))


if __name__ == "__main__":
    main()
