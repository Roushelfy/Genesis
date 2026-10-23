"""Bound one-face and two-face sharp-box reactions using continuous friction cones."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .artifacts import file_id, runtime_identity, write_json


def reaction_certificate(points, normals, friction, force, moment, length, facets):
    """Find admissible corner forces or a separating continuous-cone wrench.

    Nonnegative bilinear coordinates map every rectangular-face traction to
    its corners while preserving force, moment and the convex friction cone.
    Inner polygon generators provide feasible reactions. An outer polygon
    proposes a dual separator, checked directly against circular cones at all
    corners with a positive roundoff allowance.
    """
    tangent = np.cross(normals, np.eye(3)[np.argmin(np.abs(normals), axis=1)])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    other = np.cross(normals, tangent)
    angle = 2 * np.pi * np.arange(facets) / facets
    directions = np.cos(angle)[None, :, None] * tangent[:, None] + np.sin(angle)[None, :, None] * other[:, None]
    scale = np.linalg.norm(force)
    target = np.r_[force, moment / length] / scale
    generators = normals[:, None] + friction * directions
    matrix = np.concatenate([generators, np.cross(points[:, None] / length, generators)], axis=2).reshape(-1, 6).T
    primal = linprog(np.ones(matrix.shape[1]), A_eq=matrix, b_eq=target, bounds=(0, None), method="highs")
    if primal.success:
        forces = np.sum(primal.x.reshape(len(points), facets, 1) * generators, axis=1)
        compression = np.einsum("ij,ij->i", forces, normals)
        margins = friction * compression - np.linalg.norm(forces - compression[:, None] * normals, axis=1)
        residual = np.linalg.norm(matrix @ primal.x - target)
        assert residual < 1e-8 and margins.min() > -1e-9
        return {
            "classification": "feasible",
            "forces_n": (forces * scale).tolist(),
            "minimum_circular_cone_margin_n": float(margins.min() * scale),
            "normalized_wrench_residual": float(residual),
        }
    assert primal.status == 2, primal.message
    outer = normals[:, None] + friction / np.cos(np.pi / facets) * directions
    matrix = np.concatenate([outer, np.cross(points[:, None] / length, outer)], axis=2).reshape(-1, 6).T
    dual = linprog(target, A_ub=-matrix.T, b_ub=np.zeros(matrix.shape[1]), bounds=(-1, 1), method="highs")
    assert dual.success, dual.message
    separator = dual.x.copy()
    direction = separator[:3] + np.cross(separator[3:], points / length)
    axial = np.einsum("ij,ij->i", direction, normals)
    margins = axial - friction * np.linalg.norm(direction - axial[:, None] * normals, axis=1)
    bisector = normals.sum(axis=0)
    bisector = bisector / np.linalg.norm(bisector)
    component = normals @ bisector
    positive = np.min(component - friction * np.sqrt(np.maximum(1 - component**2, 0)))
    assert positive > 0
    separator[:3] += bisector * (max(0, -margins.min()) + 1e-10) / positive
    direction = separator[:3] + np.cross(separator[3:], points / length)
    axial = np.einsum("ij,ij->i", direction, normals)
    margins = axial - friction * np.linalg.norm(direction - axial[:, None] * normals, axis=1)
    projection = separator @ target
    assert margins.min() > 1e-12
    return {
        "classification": "excluded" if projection < -1e-8 else "unresolved_polygon_gap",
        "separator_force_coordinates": separator[:3].tolist(),
        "separator_scaled_moment_coordinates": separator[3:].tolist(),
        "minimum_continuous_cone_projection": float(margins.min()),
        "target_normalized_projection": float(projection),
    }


def rectangles(width, along, depth):
    """Place corners on a box's main face and its adjacent perpendicular face."""
    main = np.array([[x, y, 0] for x in [0, width] for y in [-along, along]])
    side = np.array([[0, y, z] for z in [-depth, 0] for y in [-along, along]])
    return np.concatenate([main, side]), np.repeat([[0, 0, 1], [-1, 0, 0]], 4, axis=0)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame-check", type=Path, required=True)
    cli.add_argument("--moments", type=Path, required=True)
    cli.add_argument("--facets", type=int, default=64)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.frame_check, args.moments]],
            "facets": args.facets,
        },
    )
    data = json.loads(args.frame_check.read_text())
    pressure = json.loads(args.moments.read_text())
    assert data["passed"] and pressure["passed"] and len(data["rows"]) == len(pressure["rows"]) == 1
    row, shape = data["rows"][0], pressure["rows"][0]
    force, moment, contacts, positions, witnesses = [
        np.array(row[key])
        for key in [
            "opposing_face_force_n",
            "opposing_face_moment_nm",
            "collision_forces_in_opposing_frame_n",
            "collision_positions_from_opposing_edge_m",
            "opponent_witnesses_from_opposing_edge_m",
        ]
    ]
    np.testing.assert_allclose(np.cross(positions, contacts).sum(axis=0), moment, atol=1e-15, rtol=0)
    np.testing.assert_allclose(witnesses[:, [0, 2]], 0, atol=1e-12, rtol=0)
    witness_moment = np.cross(witnesses, contacts).sum(axis=0)
    width, along, friction = shape["width_m"], shape["half_length_m"], row["friction"]
    assert 0 <= friction < 1 and force[0] < 0 and force[2] > 0
    # Fx >= -mu P - S and Fz >= P - mu S bound the total side compression.
    side_compression_upper = (friction * force[2] - force[0]) / (1 - friction**2)
    side_depth_lower = max(moment[1], 0) / side_compression_upper
    points, normals = rectangles(width, along, width)
    controls = {}
    for label, selected_force, selected_moment, expected in [
        ("main", np.array([0, 0, 1]), np.array([0, -width / 2, 0]), "feasible"),
        ("side", np.array([-1, 0, 0]), np.array([0, width / 2, 0]), "feasible"),
        ("tension", np.array([1, 0, -1]), np.zeros(3), "excluded"),
    ]:
        controls[label] = reaction_certificate(
            points, normals, friction, selected_force, selected_moment, along, args.facets
        )
        assert controls[label]["classification"] == expected
    force_only = reaction_certificate(np.zeros_like(points), normals, friction, force, np.zeros(3), along, args.facets)
    assert force_only["classification"] == "feasible"
    rows = []
    for along_scale in [1, 2]:
        for depth_scale in [1, 16, 64, 256]:
            points, normals = rectangles(width, along_scale * along, depth_scale * width)
            for label, target_moment in [("original", moment), ("opponent_witness", witness_moment)]:
                certificate = reaction_certificate(points, normals, friction, force, target_moment, along, args.facets)
                rows.append(
                    {
                        "load": label,
                        "half_length_m": along_scale * along,
                        "side_depth_m": depth_scale * width,
                        "primary_width_m": width,
                        "points_m": points.tolist(),
                        "certificate": certificate,
                    }
                )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "controls": controls,
            "force_only": force_only,
            "rows": rows,
            "original_moment_nm": moment.tolist(),
            "witness_moment_nm": witness_moment.tolist(),
            "witness_moment_change_nm": (witness_moment - moment).tolist(),
            "side_normal_force_upper_bound_n": float(side_compression_upper),
            "original_minimum_side_depth_bound_m": float(side_depth_lower),
            "scope": "Necessary force/wrench checks on two declared rectangular box-face supports. Continuous circular-cone exclusion is checked at rectangle corners. Feasibility supplies atomic corner reactions in the inner cone. Actual opponent witnesses independently identify the captured shared edge. Replacing midpoint moments by witness moments is a separately labelled input approximation. Pressure, material gap/deformation compatibility, friction history and physical stresses require independent checks.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
