"""Check a rejected numerical continuation stage without accepting it as physical history."""

import numpy as np

from .original_franka_ground import GroundedOriginalContact, contact_groups


def check_saved_stage(system, previous, fraction, solver_error):
    assert system.dynamic and system.fraction == fraction and 0 < fraction <= 1
    saved = system.failure_state
    body, velocity = saved["body_displacement"].copy(), saved["robot_variable"].copy()
    continued = system.evaluate(body, velocity, previous, False)
    body_error, joint_error = float(np.linalg.norm(continued[0])), float(np.linalg.norm(continued[1]))
    if not (body_error <= 1e-9 and joint_error <= 1e-9):
        return None
    physical = GroundedOriginalContact.evaluate(system, body, velocity, previous, False)
    physical_body, physical_joint = float(np.linalg.norm(physical[0])), float(np.linalg.norm(physical[1]))
    fresh = None
    if fraction == 1:
        assert system.fraction == 1
        if not (physical_body <= 1e-9 and physical_joint <= 1e-9):
            return None
        try:
            fresh = system.verify_robot_step(velocity, physical[3])
        except AssertionError:
            return None
    groups, _ = contact_groups(system, physical[2])
    record = {
        "phase": "saved_residual_continuation_stage",
        "iteration": None,
        "fraction": fraction,
        "body_residual_N": body_error,
        "joint_residual_N_or_Nm": joint_error,
        "continued_equation_stage_budget_N": 1e-9,
        "internal_newton_target_N": 1e-10,
        "strict_newton_target_met": max(body_error, joint_error) <= 1e-10,
        "physical_equation_body_residual_N": physical_body,
        "physical_equation_joint_effort_residual": physical_joint,
        "unmodified_physical_equation": fraction == 1,
        "fresh_robot_step_effort_residual": fresh,
        "contact_groups": groups,
        "internal_solver_error": solver_error,
        "intermediate_stage_is_physical_history": False,
    }
    system.history.append(record)
    return (body, velocity, physical[2], physical[3]), record
