"""Solve a small two-face corner contact with vector half-space compliance."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from .artifacts import file_id, runtime_identity, write_json
from .inspect_edge_shell_tilt import point_displacement


def rectangle_tensor(targets, centers, widths, young, poisson):
    """Integrate every surface Boussinesq/Cerruti component over rectangular sources."""
    delta = targets[:, None] - centers[None]
    normal = np.zeros(delta.shape[:2])
    xx, yy, xy = np.zeros_like(normal), np.zeros_like(normal), np.zeros_like(normal)
    cross = np.zeros_like(delta)
    for sx, sy in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        position = delta + np.array([sx, sy]) * widths[None] / 2
        x, y = position[..., 0], position[..., 1]
        radius2 = x * x + y * y
        first = x * np.arcsinh(np.divide(y, np.abs(x), out=np.zeros_like(x), where=x != 0))
        second = y * np.arcsinh(np.divide(x, np.abs(y), out=np.zeros_like(y), where=y != 0))
        normal += sx * sy * (first + second)
        xx += sx * sy * second
        yy += sx * sy * first
        xy -= sx * sy * np.sqrt(radius2)
        logarithm = np.log(radius2, out=np.zeros_like(radius2), where=radius2 > 0)
        for axis in [0, 1]:
            a, b = position[..., axis], position[..., 1 - axis]
            cross[..., axis] += sx * sy * (0.5 * b * logarithm - b + np.abs(a) * np.arctan2(b, np.abs(a)))
    tensor = np.zeros((*normal.shape, 3, 3))
    tensor[..., 0, 0] = (1 - poisson) * normal + poisson * xx
    tensor[..., 1, 1] = (1 - poisson) * normal + poisson * yy
    tensor[..., 0, 1] = tensor[..., 1, 0] = poisson * xy
    tensor[..., 2, 2] = (1 - poisson) * normal
    tensor[..., 2, :2] = (1 - 2 * poisson) * cross / 2
    tensor[..., :2, 2] = -(1 - 2 * poisson) * cross / 2
    return tensor * ((1 + poisson) / (np.pi * young * widths.prod(axis=1)))[None, :, None, None]


def galerkin_tensor(points, widths, young, poisson, order):
    """Average source-integrated compliance over target cells and retain reciprocity error."""
    knots, weights = np.polynomial.legendre.leggauss(order)
    offsets = np.array(np.meshgrid(knots / 2, knots / 2, indexing="ij")).reshape(2, -1).T
    weights = np.outer(weights, weights).reshape(-1) / 4
    tensor = np.empty((len(points), len(points), 3, 3))
    for start in range(0, len(points), 24):
        targets = points[start : start + 24, None] + widths[start : start + 24, None] * offsets
        value = rectangle_tensor(targets.reshape(-1, 2), points, widths, young, poisson)
        tensor[start : start + 24] = np.einsum(
            "iqjab,q->ijab", value.reshape(len(targets), len(weights), len(points), 3, 3), weights
        )
    reciprocal = tensor.transpose(1, 0, 3, 2)
    error = np.linalg.norm(tensor - reciprocal) / np.linalg.norm(tensor)
    return (tensor + reciprocal) / 2, float(error)


def normal_contact(compliance, gap, labels, loads, symmetric=True):
    """Solve load-controlled normal compatibility with unilateral face reactions.

    Symmetric normal compliance gives the complementary-energy minimum. Prescribed sliding directions give a
    nonsymmetric normal-response matrix and require direct complementarity checks at the returned active set.
    """
    total = loads.sum()
    scale = np.median(np.diag(compliance)) * total
    matrix, height, target = compliance * total / scale, gap / scale, loads / total
    equality = np.eye(len(loads))[labels].T
    force = np.zeros(len(gap))
    for face in range(len(loads)):
        indices = np.flatnonzero(labels == face)
        force[indices[gap[indices].argmin()]] = target[face]
    active = force > 0
    for iteration in range(5 * len(gap)):
        current = matrix[np.ix_(active, active)]
        constraint = equality[:, active]
        response = solve(
            current, np.column_stack([constraint.T, height[active]]), assume_a="pos" if symmetric else "gen"
        )
        rigid = solve(
            constraint @ response[:, :-1],
            target + constraint @ response[:, -1],
            assume_a="pos" if symmetric else "gen",
        )
        candidate = np.zeros_like(force)
        candidate[active] = response[:, :-1] @ rigid - response[:, -1]
        negative = np.flatnonzero(active & (candidate < -1e-12))
        if len(negative):
            fractions = force[negative] / (force[negative] - candidate[negative])
            leaving = negative[fractions.argmin()]
            force += fractions.min() * (candidate - force)
            force[leaving], active[leaving] = 0, False
            continue
        force = candidate
        dual = matrix @ force + height - equality.T @ rigid
        inactive = np.flatnonzero(~active)
        if not len(inactive) or dual[inactive].min() >= -1e-10:
            assert force.min() >= -1e-12 and np.linalg.norm(equality @ force - target) < 1e-10
            assert np.max(np.abs(force * dual)) < 1e-10 and dual.min() > -1e-10
            return force * total, rigid * scale, dual * scale, iteration + 1
        active[inactive[dual[inactive].argmin()]] = True
    raise RuntimeError("Two-face normal contact did not converge.")


def kernel_control():
    """Compare integrated vector kernels to independent point-displacement quadrature."""
    centers, widths = np.array([[0.0, 0.0]]), np.array([[0.4, 0.7]])
    targets = np.array([[1.2, 0.9], [-0.8, 1.3], [0.4, -1.7]])
    actual = rectangle_tensor(targets, centers, widths, 1.0, 0.3)[:, 0]
    knots, weights = np.polynomial.legendre.leggauss(32)
    offsets = np.array(np.meshgrid(knots / 2, knots / 2, indexing="ij")).reshape(2, -1).T * widths[0]
    weights = np.outer(weights, weights).reshape(-1) / 4
    expected = np.empty_like(actual)
    for axis in range(3):
        points = np.column_stack([(targets[:, None] - offsets).reshape(-1, 2), np.zeros(len(targets) * len(offsets))])
        value = point_displacement(points, np.zeros(3), np.eye(3), np.eye(3)[axis], poisson=0.3, young=1.0)
        expected[:, :, axis] = np.einsum("iqj,q->ij", value.reshape(len(targets), len(offsets), 3), weights)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--frame-check", type=Path, required=True)
    cli.add_argument("--moments", type=Path, required=True)
    cli.add_argument("--transverse", type=int, default=10)
    cli.add_argument("--transverse-power", type=int, default=2)
    cli.add_argument("--longitudinal", type=int, default=20)
    cli.add_argument("--quadrature", type=int, default=4)
    cli.add_argument("--along-extent", type=float, default=2)
    cli.add_argument("--face-updates", type=int, default=1)
    cli.add_argument("--motion", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert args.transverse_power >= 1
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.hard, args.frame_check, args.moments]]
            + ([file_id(args.motion)] if args.motion else []),
            "grid": [args.transverse, args.longitudinal],
            "transverse_power": args.transverse_power,
            "target_quadrature": args.quadrature,
            "along_extent": args.along_extent,
            "face_updates": args.face_updates,
            "prescribed_sliding": args.motion is not None,
            "scope": "Two-face local quadratic-gap contact control with specified face resultants. Default tractions are frictionless. Optional measured post-solve directions prescribe saturated Coulomb sliding on each face. Surface vector elasticity uses an undeformed flat material chart. Normal complementarity and deformed finite-face membership are checked. Original moments and elastic stick/slip history are separate requirements.",
        },
    )
    control = kernel_control()
    model = json.loads(args.moments.read_text())["rows"][0]
    frame = json.loads(args.frame_check.read_text())["rows"][0]
    with np.load(args.hard) as saved:
        slope, curvature = saved["edge_slope"][0], saved["edge_curvature"][0]
        poisson = saved["edge_poisson"][0]
        young = saved["edge_modulus"][0] * (1 - poisson**2)
    width, along = model["width_m"], model["half_length_m"]
    c = 1 / np.sqrt(1 + slope**2)
    normals = np.array([[slope * c, 0, c], [-c, 0, slope * c]])
    loads = np.array([frame["opposing_face_force_n"][2], -frame["opposing_face_force_n"][0]])
    directions = normals.copy()
    sliding = None
    if args.motion:
        motion = json.loads(args.motion.read_text())["rows"][-1]
        assert motion["step"] == 428 and len(motion["motion"]) == 1
        region = motion["motion"][0]
        np.testing.assert_allclose(
            region["force_in_face_frame_n"], frame["opposing_face_force_n"], rtol=1e-12, atol=1e-15
        )
        sliding = next(
            row
            for row in region["motions"]
            if row["points"] == "opponent_point" and row["velocity_phase"] == "post_solve_frozen_geometry"
        )
        face_to_material = np.column_stack([-normals[1], [0, 1, 0], normals[0]])
        directions = np.array(sliding["saturated_sliding_force_directions"]) @ face_to_material.T
        loads = np.array(sliding["best_nonnegative_sliding_normal_loads_n"])
        np.testing.assert_allclose(np.einsum("ij,ij->i", directions, normals), 1, atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(directions - normals, axis=1), region["friction"], atol=1e-13)
    assert loads.min() > 0
    edge = 4 * width * np.linspace(0, 1, args.transverse + 1) ** args.transverse_power
    edges = np.r_[-edge[:0:-1], edge]
    centers, xwidth = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    yedges = np.linspace(-args.along_extent * along, args.along_extent * along, args.longitudinal + 1)
    ycenters, ywidth = (yedges[:-1] + yedges[1:]) / 2, np.diff(yedges)
    points = np.array(np.meshgrid(centers, ycenters, indexing="ij")).reshape(2, -1).T
    widths = np.column_stack([np.repeat(xwidth, len(ycenters)), np.tile(ywidth, len(centers))])
    labels = np.repeat((centers < 0).astype(int), len(ycenters))
    rest = np.column_stack([points, c * curvature * (points[:, 1] ** 2 + widths[:, 1] ** 2 / 12) / 2])
    tensor, reciprocity = galerkin_tensor(points, widths, young, poisson, args.quadrature)
    rows = []
    for update in range(args.face_updates):
        current_normals = normals[labels]
        current_directions = directions[labels]
        compliance = np.einsum("ia,ijab,jb->ij", current_normals, tensor, current_directions)
        if args.motion is None:
            np.linalg.cholesky(compliance)
        gap = np.einsum("ij,ij->i", current_normals, rest)
        force, rigid, dual, iterations = normal_contact(compliance, gap, labels, loads, symmetric=args.motion is None)
        vector_force = force[:, None] * current_directions
        displacement = np.einsum("ijab,jb->ia", tensor, vector_force)
        translation = rigid @ normals
        position = rest + displacement - translation
        face_gaps = position @ normals.T
        np.testing.assert_allclose(face_gaps[np.arange(len(points)), labels], dual, rtol=1e-9, atol=1e-15)
        active = force > 1e-10 * loads.sum()
        other = face_gaps[np.arange(len(points)), 1 - labels]
        is_face_valid = other[active] <= 1e-10 * width
        rows.append(
            {
                "update": update,
                "iterations": iterations,
                "active_cells": int(active.sum()),
                "invalid_face_cells": int(np.count_nonzero(~is_face_valid)),
                "invalid_face_force_fraction": float(force[active][~is_face_valid].sum() / force.sum()),
                "maximum_active_other_face_gap_m": float(other[active].max()),
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
        if is_face_valid.all():
            break
        if update + 1 < args.face_updates:
            labels = face_gaps.argmax(axis=1)
    boundary = (np.abs(points[:, 0]) > 0.99 * np.max(np.abs(points[:, 0]))) | (
        np.abs(points[:, 1]) > 0.99 * np.max(np.abs(points[:, 1]))
    )
    np.savez_compressed(
        args.output / "fields.npz",
        points_m=points,
        widths_m=widths,
        rest_m=rest,
        labels=labels,
        normal_force_n=force,
        displacement_m=displacement,
        deformed_relative_m=position,
        compliance_m_per_n=compliance,
        vector_force_n=vector_force,
        gaps_m=face_gaps,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "kernel_relative_error": control,
            "raw_reciprocity_relative_error": reciprocity,
            "active_cells": int(active.sum()),
            "iterations": iterations,
            "face_updates": rows,
            "normal_loads_n": loads.tolist(),
            "transverse_power": args.transverse_power,
            "traction_directions_material": directions.tolist(),
            "measured_sliding_control": sliding,
            "rigid_translation_material_m": translation.tolist(),
            "rigid_normal_indentation_m": rigid.tolist(),
            "resultant_force_material_n": vector_force.sum(axis=0).tolist(),
            "moment_about_current_edge_nm": np.cross(position, vector_force).sum(axis=0).tolist(),
            "maximum_material_displacement_m": float(np.linalg.norm(displacement, axis=1).max()),
            "maximum_active_other_face_gap_m": float(other[active].max()),
            "invalid_face_cells": int(np.count_nonzero(~is_face_valid)),
            "invalid_face_force_fraction": float(force[active][~is_face_valid].sum() / force.sum()),
            "remote_boundary_force_fraction": float(force[boundary].sum() / force.sum()),
            "minimum_outside_box_gap_m": float(face_gaps.max(axis=1).min()),
            "finite_face_membership_passed": bool(is_face_valid.all()),
            "force_centroids_m": [
                (points[labels == side].T @ force[labels == side] / loads[side]).tolist() for side in range(2)
            ],
            "pressure_peak_cell_mean_pa": float((force / widths.prod(axis=1)).max()),
            "slope": float(slope),
            "curvature_per_m": float(curvature),
            "poisson": float(poisson),
            "young_pa": float(young),
            "scope": "Discrete load-controlled two-face contact on a local quadratic material gap. Optional measured directions prescribe saturated sliding with an explicitly recorded nearby force. Default tractions are frictionless. Elastic stick/slip history, the original moment, spatial convergence and physical large-gradient validity remain separate. Finite-face checks concern deformed cell averages. Numerical collision penetration is absent from this elastic solve.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
