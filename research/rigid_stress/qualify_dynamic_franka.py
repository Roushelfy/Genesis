"""Qualify consistent inertia and exact capped jaw-PD/passive virtual work."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np

from .dynamic_franka_reference import InertialRigidFingerContactSystem
from .franka_contact_reference import build_from_capture
from .refine_franka_reference import refine_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--body-refinement", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    static, _ = build_from_capture(args.capture, 2.5e-6)
    static = refine_body(static, args.body_refinement)
    system = InertialRigidFingerContactSystem(static, 0.0025)
    mapping, _held = system.coordinate_map(False)
    count = system.body.basis.N
    permutation = system.parts[0].permutation
    modes = system.body.rigid_modes[permutation]
    measured = modes.T @ (system.inertial_mass[:count, :count] @ modes)
    np.testing.assert_allclose(measured, system.body.gram, rtol=1e-12, atol=1e-14)
    reduced_mass = mapping.T @ system.inertial_mass @ mapping
    np.testing.assert_allclose(
        reduced_mass[-2:, -2:].toarray(), np.diag(system.masses + system.armature), rtol=1e-12, atol=1e-14
    )
    previous = np.zeros(system.stiffness.shape[0])
    external = system.gravity_force([0, 0, -9.81])
    tangential = ipctk.TangentialCollisions()
    system.q_target[:] = 0.018
    rng = np.random.default_rng(2026101431)
    reduced = rng.normal(size=mapping.shape[1]) * 1e-9
    reduced[-2:] = [-1e-5, -1.5e-5]
    u = mapping @ reduced
    _e, g, h, normal, _contact = system.evaluate(u, previous, tangential, True, external_force=external)
    assert len(normal) == 0
    direction = rng.normal(size=mapping.shape[1])
    direction /= np.linalg.norm(direction)
    full_direction = mapping @ direction
    checks = []
    for step in (1e-8, 3e-9):
        plus = system.evaluate(u + step * full_direction, previous, tangential, False, external_force=external)
        minus = system.evaluate(u - step * full_direction, previous, tangential, False, external_force=external)
        energy_error = abs((plus[0] - minus[0]) / (2 * step) - g @ full_direction) / abs(g @ full_direction)
        tangent_error = np.linalg.norm((plus[1] - minus[1]) / (2 * step) - h @ full_direction) / np.linalg.norm(
            h @ full_direction
        )
        assert energy_error < 1e-6 and tangent_error < 1e-6
        checks.append(
            {
                "step_m": step,
                "energy_gradient_relative_error": float(energy_error),
                "gradient_hessian_relative_error": float(tangent_error),
            }
        )
    virtual_work = []
    for q_increment in (-1e-5, -0.003):
        pure = np.zeros(mapping.shape[1])
        pure[-2:] = q_increment
        u = mapping @ pure
        _e, g, h, _normal, _contact = system.evaluate(u, previous, tangential, True, external_force=external)
        acceleration = q_increment / system.dt**2
        velocity = q_increment / system.dt
        raw = 100 * (0.04 + q_increment - 0.018) + 10 * velocity
        expected = (system.masses + system.armature) * acceleration + np.clip(raw, -4, 4) + velocity
        expected -= system.masses * (system.axes @ [0, 0, -9.81])
        observed = (mapping.T @ g)[-2:]
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
        derivative = (system.masses + system.armature) / system.dt**2 + 1 / system.dt
        if abs(raw) < 4:
            derivative += 100 + 10 / system.dt
        np.testing.assert_allclose((mapping.T @ h @ mapping)[-2:, -2:].diagonal(), derivative, rtol=1e-12, atol=1e-10)
        virtual_work.append(
            {
                "jaw_increment_m": q_increment,
                "raw_pd_force_N": raw,
                "capped_pd_force_N": float(np.clip(raw, -4, 4)),
                "complete_generalized_force_error_N": (observed - expected).tolist(),
            }
        )
    output = {
        "scope": __doc__,
        "body_refinement_levels": args.body_refinement,
        "body_tetrahedra": system.body.mesh.nelements,
        "fixed_initializer_points_m": static.fixed_initializer_points_m.tolist(),
        "fixed_initializer_components": static.fixed_initializer_components.tolist(),
        "consistent_body_mass_kg": system.body.total_mass,
        "physical_finger_mass_kg": system.masses.tolist(),
        "effective_finger_joint_mass_kg": (system.masses + system.armature).tolist(),
        "mass_mode_gram_max_absolute_error": float(np.max(np.abs(measured - system.body.gram))),
        "derivative_checks": checks,
        "exact_jaw_virtual_work": virtual_work,
        "goal_completion": False,
        "limitations": "Independent inertia/PD/passive-damping algebra and no-contact derivatives are qualified. Active implicit contact, dynamic balance, time/contact/spatial tails and matching the initializer/full arm require separate evidence. Rotor armature contributes generalized inertia but not physical gravity mass.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
