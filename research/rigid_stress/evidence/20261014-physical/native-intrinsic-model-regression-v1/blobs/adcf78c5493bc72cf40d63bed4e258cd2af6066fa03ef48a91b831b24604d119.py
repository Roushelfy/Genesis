"""Retry rejected original steps with pseudo-timestep Newton continuation at fixed physical history."""

import json
import os
from pathlib import Path

import numpy as np

from . import run_original_franka_prefix as runner
from .original_franka_ground import GroundedOriginalContact, contact_groups
from .wide_coordinate_franka import WideArticulatedContact


class GlobalizedGroundedContact(GroundedOriginalContact):
    def __init__(self, static, frame, capture_report, mjcf, dt):
        super().__init__(static, frame, capture_report, mjcf, dt)
        self.continuation_inputs = (static, frame, capture_report, mjcf)
        self.globalization_records = []

    def solve(self, body_u, variable, previous, hold_body, phase, max_iterations=100):
        try:
            return super().solve(body_u, variable, previous, hold_body, phase, max_iterations)
        except RuntimeError as error:
            if not self.dynamic or hold_body:
                raise
            direct_error = str(error)
        root = Path(os.environ["ORIGINAL_GLOBALIZATION_OUTPUT"]) / phase
        root.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(root / "direct_failure.npz", **self.failure_state)
        original_dt = self.dt
        old_body = np.asarray(previous[: self.n], dtype=np.longdouble)
        old_q, old_velocity = self.previous_q.copy(), self.previous_velocity.copy()
        predictor = self.contact.predictor.copy()
        target, caps = self.target.copy(), self.caps.copy()
        body, velocity = old_body.copy(), np.zeros(9)
        fraction, increment, attempts = 0.0, 0.0625, []
        while fraction < 1:
            trial = min(1.0, fraction + increment)
            system = GroundedOriginalContact(*self.continuation_inputs, original_dt * trial)
            system.target[:], system.caps[:] = target, caps
            pseudo_predictor = predictor.copy() if trial == 1 else old_body + trial * (predictor - old_body)
            system.begin_dynamic_step(old_q, old_velocity, pseudo_predictor)
            # Preserve the previously accepted endpoint joint pose when changing pseudo-dt.
            guess = velocity * (fraction / trial) if fraction else np.zeros(9)
            attempt = {"fraction": trial, "dt_s": system.dt, "previous_fraction": fraction}
            try:
                candidate = WideArticulatedContact.solve(
                    system, body.copy(), guess, previous, False, f"{phase}_pseudo_{trial:.8f}"
                )
                fresh = system.verify_robot_step(candidate[1], candidate[3])
                groups, _ = contact_groups(system, candidate[2])
                attempt.update(
                    accepted=True,
                    residuals=system.history[-1],
                    fresh_robot_effort_residual=fresh,
                    contact_groups=groups,
                )
            except (RuntimeError, AssertionError, ValueError) as error:
                attempt.update(accepted=False, error=str(error))
                if hasattr(system, "failure_state"):
                    np.savez_compressed(root / f"attempt{len(attempts):03d}-failure.npz", **system.failure_state)
                attempts.append(attempt)
                increment *= 0.5
                if increment < 1 / 1024:
                    break
                continue
            attempts.append(attempt)
            body, velocity, fraction = candidate[0], candidate[1], trial
            np.savez_compressed(
                root / f"stage{sum(a['accepted'] for a in attempts):03d}.npz",
                body_displacement=body,
                joint_velocity=velocity,
                displacement=candidate[2],
                contact_force_N=-candidate[3],
                fraction=fraction,
            )
            increment = min(0.25, increment * 1.5)
            if fraction == 1:
                assert system.dt == original_dt
                np.testing.assert_array_equal(system.contact.predictor, predictor)
                np.testing.assert_array_equal(system.previous_q, old_q)
                np.testing.assert_array_equal(system.previous_velocity, old_velocity)
                np.testing.assert_array_equal(system.target, target)
                np.testing.assert_array_equal(system.caps, caps)
        record = {
            "phase": phase,
            "direct_error": direct_error,
            "final_fraction": fraction,
            "recovered_original_equation": fraction == 1,
            "physical_previous_state_fixed": True,
            "pseudo_stages_are_physical_history": False,
            "attempts": attempts,
        }
        (root / "summary.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
        print(json.dumps({"globalization": record}, default=float), flush=True)
        if fraction != 1:
            raise RuntimeError(f"Original-equation continuation failed: {phase} fraction={fraction}")
        records = self.globalization_records + [record]
        self.__dict__.update(system.__dict__)
        self.globalization_records = records
        return candidate


def main():
    runner.GroundedOriginalContact = GlobalizedGroundedContact
    runner.main()


if __name__ == "__main__":
    main()
