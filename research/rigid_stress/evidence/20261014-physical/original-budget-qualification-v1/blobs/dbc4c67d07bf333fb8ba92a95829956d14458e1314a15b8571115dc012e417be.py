"""Check unchanged original physical budgets on each complete Newton candidate."""

import json
import os
from pathlib import Path

import numpy as np

from . import run_original_franka_prefix as runner
from .budget_checked_original import BudgetCheckedOriginalContact
from .original_franka_ground import contact_groups


class OriginalPhysicalBudgetReached(Exception):
    """Internal control flow after an unmodified physical equation passes all gates."""


class StagnationCheckedOriginalContact(BudgetCheckedOriginalContact):
    def __init__(self, *args):
        super().__init__(*args)
        self.check_stagnation = False
        self.last_body_norm = None
        self.repeated_body_norms = 0
        self.stagnation_candidate = None

    def evaluate(self, body_u, variable, previous, hessian):
        result = super().evaluate(body_u, variable, previous, hessian)
        if self.check_stagnation and hessian:
            body_norm, joint_norm = float(np.linalg.norm(result[0])), float(np.linalg.norm(result[1]))
            self.repeated_body_norms = self.repeated_body_norms + 1 if body_norm == self.last_body_norm else 0
            self.last_body_norm = body_norm
            if 1e-10 < max(body_norm, joint_norm) <= 1e-9:
                try:
                    fresh = self.verify_robot_step(variable, result[3])
                except AssertionError:
                    pass
                else:
                    self.stagnation_candidate = (body_u.copy(), variable.copy(), result, fresh)
                    raise OriginalPhysicalBudgetReached
        return result

    def solve(self, body_u, variable, previous, hold_body, phase, max_iterations=100):
        self.last_body_norm, self.repeated_body_norms, self.stagnation_candidate = None, 0, None
        self.check_stagnation = self.dynamic and not hold_body
        try:
            return super().solve(body_u, variable, previous, hold_body, phase, max_iterations)
        except OriginalPhysicalBudgetReached:
            body, velocity, result, fresh = self.stagnation_candidate
            rb, rj, full, contact, _, stencils, _ = result
            groups, _ = contact_groups(self, full)
            record = {
                "phase": phase,
                "iteration": None,
                "body_residual_N": float(np.linalg.norm(rb)),
                "joint_residual_N_or_Nm": float(np.linalg.norm(rj)),
                "normal_stencils": stencils,
                "original_physical_budget_N": 1e-9,
                "strict_newton_target_N": 1e-10,
                "strict_newton_target_met": False,
                "identical_body_norm_repetitions": self.repeated_body_norms,
                "acceptance_trigger": "original_physical_budgets_and_fresh_robot_step",
                "fresh_robot_step_effort_residual": fresh,
                "contact_groups": groups,
            }
            root = Path(os.environ["ORIGINAL_GLOBALIZATION_OUTPUT"]) / "physical_budget_checks"
            root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root / f"{phase}.npz", body_displacement=body, joint_velocity=velocity)
            (root / f"{phase}.json").write_text(json.dumps(record, indent=2) + "\n")
            self.history.append(record)
            print(json.dumps({"original_physical_budget_after_newton_candidate": record}), flush=True)
            return body, velocity, full, contact
        finally:
            self.check_stagnation = False


def main():
    runner.GroundedOriginalContact = StagnationCheckedOriginalContact
    runner.main()


if __name__ == "__main__":
    main()
