"""Test independently scaled two-face strip profiles against the stored corner control."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import betainc, digamma, roots_jacobi

from .artifacts import file_id, runtime_identity, write_json
from .compare_box_corner_fields import stress_errors
from .hertz_reference import halfspace_stress
from .hertz_tangential import cerruti_stress
from .study_box_corner_elasticity import galerkin_tensor


def face_scales(loads, modulus, normals, curvature, poisson):
    """Minimize each slender parabolic strip's leading elastic energy and gap work.

    U=3*F^2/(5*pi*E_star*b)*(log(2*b/w)+D), with gap work F*(s*w/5+k*b^2/10).
    The transverse square-root profile has mean log pair distance -log(4)-1/2.
    Finite parabolic line integration gives D=log(4)+1/4-(psi(3)-psi(7/2))/2,
    plus nu*n_x^2/(1-nu) for an oblique normal. Eliminating w leaves one scalar root.
    """
    widths, lengths = [], []
    for load, normal in zip(loads, normals):
        slope, curve = abs(normal[0]), normal[2] * curvature
        product = 3 * load / (np.pi * modulus * slope)
        coefficient = 3 * load / (np.pi * modulus * curve)
        constant = np.log(4) - 0.75 - (digamma(3) - digamma(3.5)) / 2
        constant += poisson * normal[0] ** 2 / (1 - poisson)

        def equation(length):
            return length**3 / coefficient - np.log(2 * length**2 / product) - constant

        lower = (2 * coefficient / 3) ** (1 / 3)
        assert equation(lower) < 0
        upper = lower * 2
        for _ in range(16):
            if equation(upper) > 0:
                break
            upper *= 2
        length = brentq(equation, lower, upper, xtol=1e-16, rtol=1e-13)
        assert abs(equation(length)) < 1e-8
        widths.append(product / length)
        lengths.append(length)
    widths, lengths = np.array(widths), np.array(lengths)
    assert np.max(widths / lengths) < 0.05
    return widths, lengths


def pressure_sources(loads, normals, widths, lengths, limits, order):
    """Integrate square-root transverse profiles and optionally truncated parabolic lines."""
    x, wx = roots_jacobi(12, 0.5, -0.5)
    x, wx = (x + 1) / 2, wx / wx.sum()
    y, wy = np.polynomial.legendre.leggauss(order)
    points, forces = [], []
    for face in range(2):
        along = y * limits[face]
        profile = 1 - (along / lengths[face]) ** 2
        integral = 2 * limits[face] * (1 - limits[face] ** 2 / (3 * lengths[face] ** 2))
        across = (1 if face == 0 else -1) * widths[face] * profile[:, None] * x
        point = np.column_stack([across.reshape(-1), np.repeat(along, len(x))])
        normal_force = (loads[face] * wy * limits[face] * profile / integral)[:, None] * wx
        np.testing.assert_allclose(normal_force.sum(), loads[face], rtol=1e-12)
        points.append(point)
        forces.append(normal_force.reshape(-1, 1) * normals[face])
    return np.concatenate(points), np.concatenate(forces)


def cell_forces(points, cells, loads, widths, lengths, limits):
    """Integrate the proposed density over independent reference-grid rectangles."""
    knots, weights = np.polynomial.legendre.leggauss(32)
    labels = (points[:, 0] < 0).astype(int)
    result = np.zeros(len(points))
    for face in range(2):
        selected = labels == face
        x = np.abs(points[selected, 0])
        dx, dy = cells[selected].T
        low = np.maximum(points[selected, 1] - dy / 2, -limits[face])
        high = np.minimum(points[selected, 1] + dy / 2, limits[face])
        span = np.maximum(high - low, 0)
        y = (high + low)[:, None] / 2 + span[:, None] * knots / 2
        profile = np.maximum(1 - (y / lengths[face]) ** 2, np.finfo(float).tiny)
        across = widths[face] * profile
        upper = np.clip((x + dx / 2)[:, None] / across, 0, 1)
        lower = np.clip((x - dx / 2)[:, None] / across, 0, 1)
        fraction = betainc(0.5, 1.5, upper) - betainc(0.5, 1.5, lower)
        integral = 2 * limits[face] * (1 - limits[face] ** 2 / (3 * lengths[face] ** 2))
        result[selected] = loads[face] * span / (2 * integral) * ((fraction * profile) @ weights)
    return result, labels


def compatibility(points, cells, force, labels, normals, young, poisson, curvature):
    """Measure loaded-cell normal mismatch and finite-face geometry without fitting pressures."""
    active = force > 1e-12 * force.sum()
    points, cells, force, labels = points[active], cells[active], force[active], labels[active]
    tensor, reciprocity = galerkin_tensor(points, cells, young, poisson, order=4)
    displacement = np.einsum("ijab,jb->ia", tensor, force[:, None] * normals[labels])
    rest = np.column_stack([points, curvature * (points[:, 1] ** 2 + cells[:, 1] ** 2 / 12) / 2])
    gap = (rest + displacement) @ normals.T
    approach = np.array(
        [force[labels == face] @ gap[labels == face, face] / force[labels == face].sum() for face in range(2)]
    )
    residual = gap - approach
    normal_error = residual[np.arange(len(force)), labels]
    other = residual[np.arange(len(force)), 1 - labels]
    return {
        "active_cells": len(force),
        "raw_reciprocity_relative_error": reciprocity,
        "rigid_normal_approach_m": approach.tolist(),
        "normal_compatibility_force_weighted_rms_m": float(np.sqrt(force @ normal_error**2 / force.sum())),
        "maximum_loaded_normal_mismatch_m": float(abs(normal_error).max()),
        "invalid_other_face_force_fraction": float(force[other > 1e-16].sum() / force.sum()),
        "maximum_other_face_gap_m": float(other.max()),
        "scope": "Loaded-cell diagnostic with load-weighted rigid approaches. Pressure parameters remain independently predicted. Cell projection and target quadrature affect these residuals. This supplies neither a complementary contact solution nor an admission threshold.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--frame-check", type=Path, required=True)
    cli.add_argument("--reference-fields", type=Path, required=True)
    cli.add_argument("--reference-grid", type=Path, required=True)
    cli.add_argument("--source-order", type=int, default=128)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.hard, args.frame_check, args.reference_fields, args.reference_grid]],
            "source_order": args.source_order,
        },
    )
    with np.load(args.hard) as saved:
        slope, curvature, poisson, modulus = [
            saved[key][0] for key in ["edge_slope", "edge_curvature", "edge_poisson", "edge_modulus"]
        ]
    c = 1 / np.sqrt(1 + slope**2)
    normals = np.array([[slope, 0, 1], [-1, 0, slope]]) * c
    curvature *= c
    frame = json.loads(args.frame_check.read_text())["rows"][0]
    loads = np.array([frame["opposing_face_force_n"][2], -frame["opposing_face_force_n"][0]])
    widths, lengths = face_scales(loads, modulus, normals, curvature, poisson)
    scaled = face_scales(loads * 9, modulus, normals, curvature / 3, poisson)
    np.testing.assert_allclose(scaled, np.array([widths, lengths]) * 3, rtol=1e-11)
    with np.load(args.reference_fields) as saved:
        probes, reference = saved["probes_m"], saved["reference_tensor_pa"]
    with np.load(args.reference_grid) as saved:
        points, cells = saved["points_m"], saved["widths_m"]
    rows = []
    for name in ["independent_strips", "finite_face_cap"]:
        limits = lengths.copy()
        if name == "finite_face_cap":
            limits[1] = min(lengths)
        integral = 2 * limits * (1 - limits**2 / (3 * lengths**2))
        current_widths = 4 * loads / (np.pi * modulus * abs(normals[:, 0]) * integral)
        source, traction = pressure_sources(loads, normals, current_widths, lengths, limits, args.source_order)
        tensor = halfspace_stress(probes, source, traction[:, 2], poisson) + cerruti_stress(
            probes, source, traction[:, :2], poisson
        )
        force, labels = cell_forces(points, cells, loads, current_widths, lengths, limits)
        closure = np.array([force[labels == face].sum() for face in range(2)]) / loads - 1
        assert np.max(abs(closure)) < 1e-3
        checks = compatibility(points, cells, force, labels, normals, modulus * (1 - poisson**2), poisson, curvature)
        depths = []
        for depth in np.unique(probes[:, 2]):
            selected = probes[:, 2] == depth
            depths.append(
                {"depth_m": float(depth), **stress_errors(probes[selected], reference[selected], tensor[selected])}
            )
        rows.append(
            {
                "model": name,
                "widths_m": current_widths.tolist(),
                "unconstrained_half_lengths_m": lengths.tolist(),
                "support_half_lengths_m": limits.tolist(),
                "projected_force_relative_error": closure.tolist(),
                "compatibility": checks,
                "depths": depths,
            }
        )
        np.savez_compressed(args.output / f"{name}.npz", probes_m=probes, tensor_pa=tensor, cell_force_n=force)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "normal_loads_n": loads.tolist(),
            "scope": "Independent leading strip-energy lengths and transverse equilibrium widths use geometry, material and force only. Cross-face elastic coupling is omitted in these scale equations. The second candidate geometrically caps side support at the primary longitudinal extent and renormalizes its load. This heuristic cap requires separate contact compatibility and physical stress evidence. The comparison is the previous frictionless local control; original frictional wrench/history and complete curved-shell accuracy remain open.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
