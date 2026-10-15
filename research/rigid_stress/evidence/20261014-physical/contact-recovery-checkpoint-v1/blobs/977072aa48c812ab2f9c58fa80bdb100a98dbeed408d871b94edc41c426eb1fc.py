"""Globalize one rejected step through pseudo-timesteps, accepting only the original final equation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import GroundedOriginalContact, build_original, contact_groups
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WideArticulatedContact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-fraction", type=float, default=0.0625)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, report, _ = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    with np.load(args.run / "results/failure.npz") as saved:
        values = {key: saved[key].copy() for key in saved.files}
    with np.load(args.run / "results/reference_geometry.npz") as saved:
        np.testing.assert_array_equal(saved["rest_positions"], static.rest)
        np.testing.assert_array_equal(saved["faces"], static.faces)
    old_body = np.asarray(values["previous_body_displacement"], dtype=np.longdouble)
    old_q, old_velocity = values["previous_joint_position"], values["previous_joint_velocity"]
    old_body_velocity, previous = values["previous_body_velocity"], values["previous_full_displacement"]
    body_u, velocity = old_body.copy(), np.zeros(9)
    fraction, increment = 0.0, args.initial_fraction
    assert 0 < increment <= 1
    attempts, accepted = [], []
    while fraction < 1:
        trial = min(1.0, fraction + increment)
        dt = config["dt"] * trial
        system = GroundedOriginalContact(static, frame, report, args.mjcf, dt)
        system.target[:] = values["target"]
        system.caps[:] = values["caps"]
        system.begin_dynamic_step(old_q, old_velocity, old_body + dt * old_body_velocity)
        guess = velocity * (fraction / trial) if fraction else np.zeros(9)
        attempt = {"fraction": trial, "dt_s": dt, "previous_fraction": fraction}
        print(json.dumps({"continuation_attempt": attempt}), flush=True)
        try:
            candidate_body, candidate_velocity, full, contact = WideArticulatedContact.solve(
                system, body_u.copy(), guess, previous, False, f"pseudo_dt_{trial:.8f}"
            )
            fresh = system.verify_robot_step(candidate_velocity, contact)
            groups, _ = contact_groups(system, full)
            attempt.update(
                accepted=True, residuals=system.history[-1], fresh_robot_effort_residual=fresh, contact_groups=groups
            )
        except (RuntimeError, AssertionError, ValueError) as error:
            attempt.update(accepted=False, error=str(error))
            if hasattr(system, "failure_state"):
                np.savez_compressed(args.output / f"attempt{len(attempts):03d}-failure.npz", **system.failure_state)
            attempts.append(attempt)
            increment *= 0.5
            if increment < 1 / 1024:
                break
            continue
        attempts.append(attempt)
        body_u, velocity, fraction = candidate_body, candidate_velocity, trial
        accepted.append(attempt)
        np.savez_compressed(
            args.output / f"stage{len(accepted):03d}.npz",
            body_displacement=body_u,
            joint_velocity=velocity,
            displacement=full,
            contact_force_N=-contact,
            fraction=fraction,
        )
        increment = min(0.25, increment * 1.5)
        if fraction == 1:
            assert system.dt == config["dt"]
            q = old_q + np.longdouble(config["dt"]) * velocity
            np.savez_compressed(
                args.output / "recovered.npz",
                displacement=full,
                joint_position=q,
                joint_velocity=velocity,
                body_velocity=(body_u - old_body) / config["dt"],
                body_acceleration=((body_u - old_body) / config["dt"] - old_body_velocity) / config["dt"],
                contact_force_N=-contact,
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
        "goal_completion": False,
        "limitations": "Pseudo-timestep stages are solver iterates, not accepted physical-history states. Only the original final timestep and unchanged original force/fresh-step budgets can recover the rejected step. Independent final-state checks, the remaining complete original history and physical convergence remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"final_fraction": fraction, "recovered_original_equation": fraction == 1}))


if __name__ == "__main__":
    main()
