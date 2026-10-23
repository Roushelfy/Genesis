"""Bound continuous quadratic-patch normals and certify a Coulomb force exclusion."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def bernstein(bary):
    """Evaluate the six quadratic triangular Bernstein weights."""
    return np.column_stack(
        [bary**2, 2 * bary[:, 0] * bary[:, 1], 2 * bary[:, 0] * bary[:, 2], 2 * bary[:, 1] * bary[:, 2]]
    )


def normal_controls(control):
    """Return Bernstein coefficients of the polynomial surface area vector."""
    first = 2 * (control[:, [3, 1, 5]] - control[:, [0, 3, 4]])
    second = 2 * (control[:, [4, 5, 2]] - control[:, [0, 3, 4]])
    diagonal = np.cross(first, second)
    mixed = (
        np.cross(first[:, [0, 0, 1]], second[:, [1, 2, 2]])
        + np.cross(first[:, [1, 2, 2]], second[:, [0, 0, 1]])
    ) / 2
    return np.concatenate([diagonal, mixed], axis=1), first, second


def bound_normals(nodes, face_normal, lower, upper, levels, coordinate_scale):
    """Enclose every quadratic face portion intersecting a supplied local box.

    Convex-hull exclusion and positive subdivision weights retain all possible
    intersections. Area-vector Bernstein coefficients enclose continuous normals.
    Rounded position and derivative allowances keep boundary decisions conservative.
    """
    control = nodes.copy()
    control[:, 3:] = 2 * nodes[:, 3:] - (nodes[:, [0, 0, 1]] + nodes[:, [1, 2, 2]]) / 2
    normals, first, second = normal_controls(control)
    area = np.cross(first.mean(axis=1), second.mean(axis=1))
    orientation = -np.sign(np.einsum("ij,ij->i", area, face_normal))
    assert np.all(orientation != 0)
    face = np.arange(len(nodes))
    vertices = np.eye(3)
    middle = (vertices[[0, 0, 1]] + vertices[[1, 2, 2]]) / 2
    children = np.array(
        [
            [vertices[0], middle[0], middle[1]],
            [middle[0], vertices[1], middle[2]],
            [middle[1], middle[2], vertices[2]],
            [middle[0], middle[2], middle[1]],
        ]
    )
    transform = []
    for triangle in children:
        corner = bernstein(triangle)
        mids = bernstein((triangle[[0, 0, 1]] + triangle[[1, 2, 2]]) / 2)
        transform.append(np.concatenate([corner, 2 * mids - (corner[[0, 0, 1]] + corner[[1, 2, 2]]) / 2]))
    transform = np.array(transform)
    assert transform.min() >= 0
    np.testing.assert_array_equal(transform.sum(axis=2), 1)
    counts = []
    for level in range(levels + 1):
        allowance = 2048 * np.finfo(float).eps * coordinate_scale * (level + 2)
        low, high = control.min(axis=1) - allowance, control.max(axis=1) + allowance
        selected = np.all(high >= lower, axis=1) & np.all(low <= upper, axis=1)
        control, orientation, face = control[selected], orientation[selected], face[selected]
        counts.append(len(control))
        assert len(control) > 0
        if level < levels:
            control = np.einsum("cij,fjk->fcik", transform, control).reshape(-1, 6, 3)
            orientation, face = np.repeat(orientation, 4), np.repeat(face, 4)
    low, high = control.min(axis=1) - allowance, control.max(axis=1) + allowance
    height_boundary = (low[:, 2] <= lower[2]) | (high[:, 2] >= upper[2])
    normals, first, second = normal_controls(control)
    normals *= orientation[:, None, None]
    derivative_scale = np.linalg.norm(first, axis=2).max(axis=1) + np.linalg.norm(second, axis=2).max(axis=1)
    error = 64 * allowance * derivative_scale + 512 * allowance**2
    axial_lower = normals[:, :, 2] - error[:, None]
    transverse_upper = np.linalg.norm(normals[:, :, :2], axis=2) + error[:, None]
    assert np.all(axial_lower > 0)
    slope = float(np.max(transverse_upper / axial_lower))
    return {
        "levels": levels,
        "retained_cells_by_level": counts,
        "original_faces": np.unique(face).tolist(),
        "maximum_normal_tangent_bound": slope,
        "maximum_normal_angle_degrees": float(np.degrees(np.arctan(slope))),
        "height_boundary_boxes": int(np.count_nonzero(height_boundary)),
        "position_roundoff_allowance_m": float(allowance),
        "minimum_rounded_axial_coefficient": float(axial_lower.min()),
    }


def force_certificate(force, friction, normal_slope):
    """Separate a force from the enlarged cone containing every local traction cone."""
    tangent = np.linalg.norm(force[:2])
    force_angle = np.arctan2(tangent, force[2])
    cone_angle = np.arctan(friction) + np.arctan(normal_slope)
    is_excluded = cone_angle < np.pi / 2 and force_angle > cone_angle + 1e-10
    result = {
        "force_angle_degrees": float(np.degrees(force_angle)),
        "enlarged_cone_angle_degrees": float(np.degrees(cone_angle)),
        "continuous_force_excluded": bool(is_excluded),
    }
    if is_excluded:
        middle = (force_angle + cone_angle) / 2
        ray = np.r_[np.cos(middle) * force[:2] / tangent, -np.sin(middle)]
        upper = np.sin(cone_angle - middle)
        projection = ray @ force
        assert upper < -1e-12 and projection > 1e-12 * np.linalg.norm(force)
        result.update(
            separating_ray=ray.tolist(),
            maximum_unit_traction_projection=float(upper),
            target_force_projection_n=float(projection),
        )
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--moments", type=Path, required=True)
    cli.add_argument("--levels", nargs="+", type=int, default=[8, 10])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.frame, args.hard, args.moments]],
        },
    )
    vertices = np.array([[-1.0, -1, 0], [1.0, -1, 0], [1.0, 1, 0], [-1.0, 1, 0]])
    corners = vertices[np.array([[0, 1, 2], [0, 2, 3]])]
    nodes = np.concatenate([corners, (corners[:, [0, 0, 1]] + corners[:, [1, 2, 2]]) / 2], axis=1)
    lower, upper = np.array([-0.1, -0.2, -0.1]), np.array([0.1, 0.2, 0.1])
    face_normal = np.tile([0.0, 0, -1], (2, 1))
    flat = bound_normals(nodes, face_normal, lower, upper, 6, 1)
    assert flat["height_boundary_boxes"] == 0 and flat["maximum_normal_tangent_bound"] < 1e-6
    assert force_certificate(np.array([0.4, 0, 1]), 0.3, flat["maximum_normal_tangent_bound"])["continuous_force_excluded"]
    assert not force_certificate(np.array([0.2, 0, 1]), 0.3, flat["maximum_normal_tangent_bound"])["continuous_force_excluded"]
    nodes[:, :, 2] = (0.2 * nodes[:, :, 0] ** 2 + 0.3 * nodes[:, :, 1] ** 2) / 2
    curved = bound_normals(nodes, face_normal, lower, upper, 9, 1)
    exact_slope = np.hypot(0.2 * upper[0], 0.3 * upper[1])
    assert exact_slope <= curved["maximum_normal_tangent_bound"] < 1.03 * exact_slope
    assert curved["height_boundary_boxes"] == 0
    moments = json.loads(args.moments.read_text())
    assert moments["passed"] and len(moments["rows"]) == 1
    model = moments["rows"][0]
    with np.load(args.hard) as saved:
        patch = model["region"] * saved["profile"].shape[0] + moments["environment"]
        center, basis = saved["center_m"][patch], saved["basis"][patch]
    with np.load(args.frame) as saved:
        points = saved["material_nodes_m"][saved["material_surface_nodes"].T]
        face_normal = saved["material_face_normal"] @ basis
    coordinate_scale = max(np.abs(points).max(), np.abs(center).max())
    nodes = (points - center) @ basis
    width, along, mean = model["width_m"], model["half_length_m"], model["pressure_mean_x_m"]
    lower, upper = np.array([-mean, -along, -2 * along]), np.array([width - mean, along, 2 * along])
    rows = []
    for level in args.levels:
        bounds = bound_normals(nodes, face_normal, lower, upper, level, coordinate_scale)
        assert bounds["height_boundary_boxes"] == 0
        proof = force_certificate(np.array(model["local_force_n"]), model["friction"], bounds["maximum_normal_tangent_bound"])
        rows.append({"bounds": bounds, "force_certificate": proof})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "controls": {"flat": flat, "paraboloid": curved, "paraboloid_exact_tangent_bound": float(exact_slope)},
            "box_lower_m": lower.tolist(),
            "box_upper_m": upper.tolist(),
            "rows": rows,
            "scope": "A continuous rest-surface normal bound on the rectangular superset of the predicted sharp-edge support. No retained surface box touches the height boundary, enclosing the local connected sheet. The enlarged circular cone contains arbitrary nonnegative Coulomb tractions over that sheet. Exclusion concerns this fixed support and undeformed-normal model; other support, deformed geometry or approximate loads require separate physical validation.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
