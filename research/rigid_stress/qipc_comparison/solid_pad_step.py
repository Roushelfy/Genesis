"""Converge the constitutive correction while retaining the full contact solve."""

import numpy as np


def advance(
    factor, jacobian, contact, mode, rhs, old_displacement, gap, previous_normal, args, command, reference, nonlinear
):
    stiffness = reference.stiffness
    direct = getattr(args, "constitutive_correction", "subtracted") == "direct"
    correction = np.zeros_like(old_displacement[:-1])
    if nonlinear is not None:
        correction = (
            nonlinear.force_correction(old_displacement[:-1])
            if direct
            else stiffness @ old_displacement[:-1] - nonlinear.force_energy(old_displacement[:-1])[0]
        )
    rounds = []
    for iteration in range(20 if nonlinear is not None else 1):
        corrected_rhs = rhs + np.r_[args.dt * correction, 0.0]
        free = factor.solve(corrected_rhs)
        result = contact.solve(
            (jacobian @ free).reshape(-1, 3) if mode == "condensed" else free,
            gap,
            previous_normal,
            args.dt,
            1e9,
            args.friction,
            command,
            reference.total_mass,
        )
        generalized_force = jacobian.T @ result["force"].reshape(-1)
        # The modal solver already evaluates A^-1 J^T impulse for its full
        # compliance check. Reuse that response by linear superposition;
        # the driver still checks the actual complete physical equation.
        velocity = (
            result["full_velocity"]
            if args.reuse_contact_velocity
            else factor.solve(corrected_rhs + args.dt * generalized_force)
        )
        displacement = old_displacement + args.dt * velocity
        if nonlinear is None:
            internal = stiffness @ displacement[:-1]
            error = 0.0
        else:
            if direct:
                updated = nonlinear.force_correction(displacement[:-1])
                internal = None
            else:
                internal, _ = nonlinear.force_energy(displacement[:-1])
                updated = stiffness @ displacement[:-1] - internal
            error = float(np.linalg.norm(updated - correction) / max(command, 1e-3))
            correction = updated
        rounds.append(
            {
                "iteration": iteration + 1,
                "force_correction_error": error,
                "contact_newton_iterations": result["iterations"],
            }
        )
        if error <= 1e-9:
            if internal is None:
                # Independently reassemble complete Piola force for the actual
                # equation and current-position moment admission in the driver.
                internal, _ = nonlinear.force_energy(displacement[:-1])
                rounds[-1]["direct_vs_subtracted_correction_relative"] = float(
                    np.linalg.norm(correction - (stiffness @ displacement[:-1] - internal)) / max(command, 1e-3)
                )
            return result, generalized_force, velocity, displacement, internal, rounds
    raise RuntimeError(f"P2 constitutive/contact coupling failed: {rounds[-1]}")
