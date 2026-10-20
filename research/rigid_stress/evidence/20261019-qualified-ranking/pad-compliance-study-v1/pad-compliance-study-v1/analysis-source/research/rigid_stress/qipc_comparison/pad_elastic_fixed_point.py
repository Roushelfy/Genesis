"""Solve a soft physical pad through full-body elastic applications and certify the final equations."""

import json
from pathlib import Path
import time

import numpy as np

from .contact_qp import ContactQPCertificate
from .pad_equilibrium import normal_pad_equilibrium


def solve_pad_fixed_point(model, nodes, sign, interpolation, gap, rigid, area, stiffness, load, journal):
    """Iterate the unchanged pad law without forming a dense contact compliance matrix."""
    started = time.perf_counter()
    scale = load / (stiffness * np.pi * 0.002**2)
    alpha = np.zeros(len(nodes))
    coordinates = None
    point_rigid = interpolation @ rigid
    rounds = []
    for iteration in range(80):
        force = np.zeros(3 * len(model.vertices))
        force[3 * nodes + 2] = sign * alpha
        correction = model.mass_modes @ np.linalg.solve(model.gram, model.rigid_modes.T @ force)
        displacement = np.zeros_like(force)
        displacement[model.free] = model.factor.solve((force - correction)[model.free] / model.young)
        displacement = model.reference.align_rigid_gauge(displacement)
        elastic_gap = sign * displacement[3 * nodes + 2]
        coordinates, multipliers, force_error, _ = normal_pad_equilibrium(
            interpolation @ (gap + elastic_gap), point_rigid, area, stiffness, load, coordinates
        )
        updated = interpolation.T @ multipliers
        change = float(np.linalg.norm(updated - alpha) / max(np.linalg.norm(updated), 1e-30))
        alpha = updated
        body_gap = interpolation @ (gap + elastic_gap + rigid @ coordinates)
        slack = (body_gap + multipliers / (stiffness * area)) / scale
        complementarity = float(np.max(np.abs(multipliers / load * slack)))
        converged = bool(change <= 1e-10 and force_error <= 1e-8 and complementarity <= 1e-8 and slack.min() >= -1e-10)
        rounds.append(
            {
                "iteration": iteration + 1,
                "nodal_force_change_relative": change,
                "force_moment_relative_error": force_error,
                "converged": converged,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        journal.write(json.dumps(rounds[-1], allow_nan=False) + "\n")
        if converged:
            break
    return ContactQPCertificate(
        interpolation=interpolation,
        multipliers=multipliers / load,
        coordinates=coordinates / scale,
        nodal_force=alpha / load,
        constraint_gap=slack,
        rounds=rounds,
        status="FixedPointConverged" if converged else "IterationLimit",
        iterations=len(rounds),
        solve_seconds=time.perf_counter() - started,
        force_error=force_error,
        stationarity=change,
        complementarity=complementarity,
        minimum_gap=float(slack.min()),
        fixed_passed=converged,
        solver_version="full-elastic-fixed-point-1",
        solver_module=str(Path(__file__).resolve()),
    )
