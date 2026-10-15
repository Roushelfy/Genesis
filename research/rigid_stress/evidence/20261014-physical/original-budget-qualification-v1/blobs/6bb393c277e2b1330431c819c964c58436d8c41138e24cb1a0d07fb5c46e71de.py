"""Continue original history with a vanishing residual defect after a rejected direct Newton solve."""

import json
import os
from pathlib import Path

import numpy as np

from . import run_original_franka_prefix as runner
from .continue_original_residual import ResidualContinuationContact
from .original_franka_ground import GroundedOriginalContact, contact_groups
from .original_residual_stage_budget import check_saved_stage
from .wide_coordinate_franka import WideArticulatedContact


class ResidualGlobalizedContact(GroundedOriginalContact):
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
        # Reuse the exact original state, predictor and dt. Only Newton's residual
        # merit is continued; fraction1 evaluates the unmodified original residual.
        system = ResidualContinuationContact.__new__(ResidualContinuationContact)
        system.__dict__.update(self.__dict__)
        system.fraction, system.defect = 1.0, None
        body, velocity = np.asarray(previous[: self.n], dtype=np.longdouble).copy(), np.zeros(9)
        baseline = system.evaluate(body, velocity, previous, False)
        system.defect = (baseline[0].copy(), baseline[1].copy())
        fraction, increment, attempts = 0.0, 0.0625, []
        while fraction < 1:
            trial = min(1.0, fraction + increment)
            system.fraction = trial
            attempt = {"fraction": trial, "dt_s": self.dt, "previous_fraction": fraction}
            try:
                candidate = WideArticulatedContact.solve(
                    system, body.copy(), velocity.copy(), previous, False, f"{phase}_residual_{trial:.8f}"
                )
                groups, _ = contact_groups(system, candidate[2])
                attempt.update(accepted=True, residuals=system.history[-1], contact_groups=groups)
                if trial == 1:
                    attempt["fresh_robot_effort_residual"] = system.verify_robot_step(candidate[1], candidate[3])
            except (RuntimeError, AssertionError, ValueError) as error:
                checked = check_saved_stage(system, previous, trial, str(error)) if isinstance(error, RuntimeError) else None
                if checked is None:
                    attempt.update(accepted=False, error=str(error))
                    np.savez_compressed(root / f"attempt{len(attempts):03d}-failure.npz", **system.failure_state)
                    attempts.append(attempt)
                    increment *= 0.5
                    if increment < 1 / 1024:
                        break
                    continue
                candidate, record = checked
                attempt.update(accepted=True, residuals=record, contact_groups=record["contact_groups"],
                               acceptance_route="separately_checked_continued_equation_budget")
                if trial == 1:
                    attempt["fresh_robot_effort_residual"] = record["fresh_robot_step_effort_residual"]
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
        record = {
            "phase": phase,
            "direct_error": direct_error,
            "final_fraction": fraction,
            "recovered_original_equation": fraction == 1,
            "original_dt_s": self.dt,
            "internal_newton_target_N": 1e-10,
            "pseudo_stage_residual_budget_N": 1e-9,
            "original_physical_force_budget_N": 1e-9,
            "physical_previous_state_fixed": True,
            "pseudo_stages_are_physical_history": False,
            "attempts": attempts,
        }
        (root / "summary.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
        print(json.dumps({"residual_globalization": record}, default=float), flush=True)
        if fraction != 1:
            raise RuntimeError(f"Original residual continuation failed: {phase} fraction={fraction}")
        self.history = system.history
        self.failure_state = system.failure_state
        return candidate


def main():
    runner.GroundedOriginalContact = ResidualGlobalizedContact
    runner.main()


if __name__ == "__main__":
    main()
