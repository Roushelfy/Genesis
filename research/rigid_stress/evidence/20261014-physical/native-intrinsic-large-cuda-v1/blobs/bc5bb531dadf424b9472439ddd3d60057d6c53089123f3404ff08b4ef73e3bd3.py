"""Retain the strict Newton target while separately checking the unchanged original physical budgets."""

import json
import os
from pathlib import Path

import numpy as np

from . import run_original_franka_prefix as runner
from .original_franka_ground import GroundedOriginalContact, contact_groups
from .residual_globalized_original import ResidualGlobalizedContact


class BudgetCheckedOriginalContact(ResidualGlobalizedContact):
    def solve(self, body_u, variable, previous, hold_body, phase, max_iterations=100):
        try:
            return GroundedOriginalContact.solve(self, body_u, variable, previous, hold_body, phase, max_iterations)
        except RuntimeError as error:
            if not self.dynamic or hold_body:
                raise
            direct_error = str(error)
        candidate = self.failure_state
        body, velocity = candidate["body_displacement"], candidate["robot_variable"]
        rb, rj, full, contact, _, stencils, _ = self.evaluate(body, velocity, previous, False)
        body_error, joint_error = float(np.linalg.norm(rb)), float(np.linalg.norm(rj))
        if max(body_error, joint_error) <= 1e-9:
            try:
                fresh = self.verify_robot_step(velocity, contact)
            except AssertionError:
                pass
            else:
                groups, _ = contact_groups(self, full)
                record = {
                    "phase": phase,
                    "iteration": None,
                    "body_residual_N": body_error,
                    "joint_residual_N_or_Nm": joint_error,
                    "normal_stencils": stencils,
                    "strict_newton_target_met": max(body_error, joint_error) <= 1e-10,
                    "original_physical_budget_N": 1e-9,
                    "fresh_robot_step_effort_residual": fresh,
                    "contact_groups": groups,
                    "internal_newton_failure": direct_error,
                }
                root = Path(os.environ["ORIGINAL_GLOBALIZATION_OUTPUT"]) / "physical_budget_acceptances"
                root.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(root / f"{phase}.npz", **candidate)
                (root / f"{phase}.json").write_text(json.dumps(record, indent=2) + "\n")
                self.history.append(record)
                print(json.dumps({"original_physical_budget_acceptance": record}), flush=True)
                return body, velocity, full, contact
        # An artificial defect is only a numerical globalization device. Its
        # final fraction-one state still uses the original physical equation.
        return super().solve(body_u, variable, previous, hold_body, phase, max_iterations)


def main():
    runner.GroundedOriginalContact = BudgetCheckedOriginalContact
    runner.main()


if __name__ == "__main__":
    main()
