"""Check actual geometry, exact rigid jaw motion and nonlinear PD/bulk derivatives."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np

from .franka_contact_reference import build_from_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    system, row = build_from_capture(args.capture, 2.5e-6)
    assert len(system.finger_nodes) == 2
    zero = np.zeros(system.stiffness.shape[0])
    tangent = ipctk.TangentialCollisions()
    initial = system.mesh.vertices(system.rest)
    intersections = bool(ipctk.has_intersections(system.mesh, initial))
    normal, _ = system.normal_collisions(zero)
    assert len(normal) == 0
    mapping, held = system.coordinate_map(True)
    rng = np.random.default_rng(2026101417)
    reduced = rng.normal(size=mapping.shape[1]) * 1e-9
    reduced[-2:] = [-1e-6, -1.5e-6]
    u = mapping @ reduced
    for nodes, axis in zip(system.finger_nodes, system.axes, strict=True):
        translated = u.reshape(-1, 3)[nodes]
        np.testing.assert_allclose(translated, np.broadcast_to(translated[0], translated.shape), rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.cross(translated, axis), 0, rtol=0, atol=1e-15)
    direction = rng.normal(size=mapping.shape[1])
    direction /= np.linalg.norm(direction)
    system.q_target[:] = 0.018
    external = system.gravity_force([0, 0, -9.81])
    _energy, gradient, matrix, normal, _ = system.evaluate(u, u, tangent, True, external_force=external)
    assert len(normal) == 0
    red_gradient = mapping.T @ gradient
    red_matrix = mapping.T @ matrix @ mapping
    checks = []
    for step in (1e-8, 3e-9):
        plus = system.evaluate(u + step * (mapping @ direction), u, tangent, False, external_force=external)
        minus = system.evaluate(u - step * (mapping @ direction), u, tangent, False, external_force=external)
        e = abs((plus[0] - minus[0]) / (2 * step) - red_gradient @ direction) / abs(red_gradient @ direction)
        g = np.linalg.norm(mapping.T @ (plus[1] - minus[1]) / (2 * step) - red_matrix @ direction) / np.linalg.norm(
            red_matrix @ direction
        )
        assert e < 1e-6 and g < 1e-6
        checks.append(
            {"step_m": step, "energy_gradient_relative_error": float(e), "gradient_hessian_relative_error": float(g)}
        )
    # Isolate the jaw spring and rigid-mass virtual work at zero body displacement.
    pure = np.zeros(mapping.shape[1])
    pure[-2:] = [-0.001, -0.002]
    _, g, _, _, _ = system.evaluate(mapping @ pure, mapping @ pure, tangent, False, external_force=external)
    measured = (mapping.T @ g)[-2:]
    expected = 100 * (system.q_reference + pure[-2:] - system.q_target) - system.masses * (
        system.axes @ np.array([0, 0, -9.81])
    )
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-12)
    report = {
        "scope": __doc__,
        "capture": str(args.capture),
        "actual_source_snapshot": row["geometry_snapshot"],
        "body_mass_kg": system.body.total_mass,
        "finger_masses_kg": system.masses.tolist(),
        "rigid_finger_vertices": [len(nodes) for nodes in system.finger_nodes],
        "initial_normal_collisions": len(normal),
        "initial_intersections": intersections,
        "derivative_checks": checks,
        "jaw_gravity_spring_virtual_work_error_N": (measured - expected).tolist(),
        "held_body_gauge_rows": len(held),
        "goal_completion": False,
        "limitations": "Open geometry and no-contact derivative/virtual-work checks qualify the independent kinematic actuator model. Active contact, finite-body deformation, fixture release, solver tolerance/regularization/history and corresponding rigid fixed-hand comparison remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
