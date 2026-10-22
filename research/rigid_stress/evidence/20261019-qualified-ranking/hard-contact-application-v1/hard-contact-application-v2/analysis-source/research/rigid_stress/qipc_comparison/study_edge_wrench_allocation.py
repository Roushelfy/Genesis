"""Test constrained force redistribution at fixed captured physical pressure centers."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, minimize

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse
from .study_far_edge_contact import projected_surface
from .study_quadratic_rest_surface import quadratic_basis


def ground_pressure_geometry(frame):
    """Construct the ground Hertz patch directly on the captured quadratic material."""
    force, point = frame["contact_force_n"][0, 0], frame["contact_position_m"][0, 0]
    normal = -frame["contact_normal"][0, 0]
    axis = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    axis /= np.linalg.norm(axis)
    plane = np.stack([axis, np.cross(normal, axis)])
    nodes = frame["material_nodes_m"][frame["material_surface_nodes"].T]
    centers, normals, anchors = projected_surface(
        nodes,
        frame["material_face_normal"],
        point[None],
        plane,
        normal,
        frame["source_witness_m"][0, 0],
        return_geometry=True,
    )
    center, normal, anchor = centers[0], normals[0], anchors[0]
    local_nodes = nodes[anchor]
    coordinate = np.linalg.lstsq((local_nodes[1:3] - local_nodes[0]).T, center - local_nodes[0], rcond=None)[0]
    for _ in range(8):
        values, gradient, hessian = quadratic_basis(np.array([[1 - coordinate.sum(), *coordinate]]))
        tangent = local_nodes.T @ gradient[0]
        coordinate -= np.linalg.lstsq(tangent, values[0] @ local_nodes - center, rcond=None)[0]
    np.testing.assert_allclose(values[0] @ local_nodes, center, rtol=0, atol=1e-12)
    second = np.einsum("nab,ni->iab", hessian, local_nodes)
    inverse = np.linalg.inv(tangent.T @ tangent)
    tensor = -tangent @ inverse @ np.einsum("i,iab->ab", normal, second) @ inverse @ tangent.T
    axis -= normal * (axis @ normal)
    axis /= np.linalg.norm(axis)
    basis = np.column_stack([axis, np.cross(axis, normal), -normal])
    curvature = basis[:, :2].T @ tensor @ basis[:, :2]
    axes, direction, _, peak = ellipse(-force @ normal, 1e10 / 0.91, curvature)
    basis[:, :2] = basis[:, :2] @ np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    np.testing.assert_allclose(np.linalg.det(basis), 1, rtol=0, atol=1e-12)
    assert -normal @ frame["contact_normal"][0, 0] > 0.999
    return {
        "contact": 0,
        "kind": "pressure",
        "profile": "hertz",
        "center": center.tolist(),
        "basis": basis.tolist(),
        "anchor": int(anchor),
        "force": (force @ basis).tolist(),
        "along": float(axes[0]),
        "width": float(axes[1]),
        "exponent": 0.5,
        "amplitude": float(peak),
        "curvature_per_m": curvature.tolist(),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--normal-frame", choices=["collider", "material"], default="material")
    cli.add_argument("--ground-hertz", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.frame, args.conditions, args.geometry]],
            "normal_frame": args.normal_frame,
            "ground_hertz": args.ground_hertz,
            "scope": "Fixed pressure centers for the two edge contacts, unchanged captured locations for every other active source. Find the minimum weighted change in all vector forces subject to the original total force/moment and source-local compression/Coulomb cones. This is an optimistic necessary allocation check: other centers are not independently qualified here and changed forces have not yet been propagated through the pressure law or complete stress field. No runtime input gate changes.",
        },
    )
    with np.load(args.frame) as f:
        active = np.flatnonzero(f["contact_valid"][0] & (np.linalg.norm(f["contact_force_n"][0], axis=1) > 0))
        force = f["contact_force_n"][0, active]
        original = f["contact_position_m"][0, active]
        normal = f["contact_normal"][0, active]
        nodes = f["material_nodes_m"]
        pair = f["source_pair"][0, active]
        ground = ground_pressure_geometry(f) if args.ground_hertz else None
    with np.load(args.conditions) as f:
        friction = max(0.6 * float(f["friction"][0, 0]), 0.1, 0.01)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    magnitude = np.linalg.norm(force, axis=1)
    collision_compression = np.sum(force * normal, axis=1)
    collision_tangent = force - collision_compression[:, None] * normal
    assert np.all(collision_compression >= 0)
    assert np.all(np.linalg.norm(collision_tangent, axis=1) <= friction * collision_compression + 1e-10 * magnitude)
    centers = original.copy()
    if args.ground_hertz:
        assert args.normal_frame == "material"
        index = np.flatnonzero(active == 0)[0]
        centers[index], normal[index] = ground["center"], np.array(ground["basis"])[:, 2]
    geometry = json.loads(args.geometry.read_text())["cases"]
    for row in geometry:
        if row["kind"] == "pressure":
            index = np.flatnonzero(active == row["contact"])[0]
            centers[index] = row["center"]
            if args.normal_frame == "material":
                normal[index] = np.array(row["basis"])[:, 2]
    compression = np.sum(force * normal, axis=1)
    tangent = force - compression[:, None] * normal
    assert np.all(compression >= 0)
    origin = centers.mean(axis=0)
    length = np.linalg.norm(np.ptp(nodes, axis=0))
    matrix = np.concatenate(
        [np.vstack([np.eye(3), np.cross(p - origin, np.eye(3)).T / length]) for p in centers], axis=1
    )
    delta = np.r_[np.zeros(3), np.cross(original - centers, force).sum(axis=0) / length]
    weighted = matrix * np.repeat(magnitude, 3)
    left, singular, _ = np.linalg.svd(weighted, full_matrices=True)
    rank = np.count_nonzero(singular > singular[0] * 1e-12)
    equality = left[:, :rank].T @ weighted
    rhs = left[:, :rank].T @ delta
    minimum = np.linalg.lstsq(weighted, delta, rcond=1e-12)[0]
    residual = delta - weighted @ minimum
    force_scale = magnitude.sum()
    consistent = np.linalg.norm(residual) < 1e-10 * force_scale

    def constraints(value):
        candidate = force + magnitude[:, None] * value.reshape(-1, 3)
        axial = np.sum(candidate * normal, axis=1)
        radial = np.linalg.norm(candidate - axial[:, None] * normal, axis=1)
        return np.r_[axial / magnitude, (friction * axial - radial) / magnitude]

    def jacobian(value):
        candidate = force + magnitude[:, None] * value.reshape(-1, 3)
        axial = np.sum(candidate * normal, axis=1)
        radial = candidate - axial[:, None] * normal
        direction = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-30)
        result = np.zeros((2 * len(force), 3 * len(force)))
        for i in range(len(force)):
            result[i, 3 * i : 3 * i + 3] = normal[i]
            result[i + len(force), 3 * i : 3 * i + 3] = friction * normal[i] - direction[i]
        return result

    candidate = None
    if consistent:
        solved = minimize(
            lambda u: 0.5 * u @ u,
            minimum,
            jac=lambda u: u,
            method="SLSQP",
            constraints=[
                {"type": "eq", "fun": lambda u: equality @ u - rhs, "jac": lambda u: equality},
                {"type": "ineq", "fun": constraints, "jac": jacobian},
            ],
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        value = solved.x
        constraint = constraints(value)
        selected = constraint < 1e-7
        gradient = jacobian(value)[selected]
        multiplier_matrix = np.column_stack([equality.T, -gradient.T])
        multiplier = lsq_linear(
            multiplier_matrix,
            -value,
            bounds=(np.r_[np.full(rank, -np.inf), np.zeros(len(gradient))], np.full(rank + len(gradient), np.inf)),
            tol=1e-13,
        )
        stationarity = np.linalg.norm(multiplier_matrix @ multiplier.x + value)
        final_force = force + magnitude[:, None] * value.reshape(-1, 3)
        candidate = {
            "optimizer_success": bool(solved.success),
            "message": str(solved.message),
            "iterations": int(solved.nit),
            "forces_n": final_force.tolist(),
            "relative_changes": np.linalg.norm(value.reshape(-1, 3), axis=1).tolist(),
            "wrench_residual_over_force": float(np.linalg.norm(weighted @ value - delta) / force_scale),
            "minimum_normalized_cone_margin": float(constraint.min()),
            "stationarity": float(stationarity),
            "within_five_percent_force_change": bool(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1)) <= 0.05),
        }
        candidate["qualified_constrained_minimum"] = bool(
            solved.success
            and candidate["wrench_residual_over_force"] < 1e-10
            and constraint.min() > -1e-9
            and stationarity < 1e-6
        )
    rows = [
        {
            "contact": int(index),
            "pair": int(pair[i]),
            "original_position_m": original[i].tolist(),
            "pressure_position_m": centers[i].tolist(),
            "force_n": force[i].tolist(),
            "normal": normal[i].tolist(),
            "compression_n": float(compression[i]),
            "tangential_ratio": float(np.linalg.norm(tangent[i]) / compression[i]),
            "position_change_m": float(np.linalg.norm(centers[i] - original[i])),
        }
        for i, index in enumerate(active)
    ]
    np.savez_compressed(
        args.output / "allocation.npz",
        contact_ids=active,
        original_position_m=original,
        pressure_position_m=centers,
        original_force_n=force,
        normal=normal,
        matrix=matrix,
        weighted_matrix=weighted,
        delta_wrench_scaled=delta,
        unconstrained_relative_change=minimum.reshape(-1, 3),
        unreachable_wrench=residual,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "contacts": rows,
            "friction": friction,
            "normal_frame": args.normal_frame,
            "ground_hertz": args.ground_hertz,
            "ground_geometry": ground,
            "length_scale_m": float(length),
            "wrench_rank": int(rank),
            "singular_values": singular.tolist(),
            "linear_constraints_consistent": bool(consistent),
            "unreachable_wrench_over_force": float(np.linalg.norm(residual) / force_scale),
            "unconstrained_relative_changes": np.linalg.norm(minimum.reshape(-1, 3), axis=1).tolist(),
            "lower_bound_maximum_relative_change": float(np.linalg.norm(minimum) / np.sqrt(len(force))),
            "constrained": candidate,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
