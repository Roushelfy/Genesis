"""Validate an inexpensive edge-pressure lookup against independent contact pressures."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, halfspace_stress, solve_hard_contact, square_compliance
from .study_hard_edge import metrics, pressure_shapes
from .study_franka_contact_gap import von_mises


def elliptic_shapes(points, width, inverse, scale, offset, order=20):
    """Integrate the elliptical regular/edge trial with x=u^2 at the finite edge."""
    knots, weights = np.polynomial.legendre.leggauss(order)
    bounds = np.sqrt(np.maximum(points[:, 0, None] + np.array([-0.5, 0.5]) * width, 0))
    half = (bounds[:, 1] - bounds[:, 0]) / 2
    u = bounds.mean(axis=1)[:, None] + half[:, None] * knots
    x = u[:, :, None] ** 2 + offset
    y = (points[:, 1, None] + width / 2 * knots)[:, None, :]
    squared = inverse[0, 0] * x**2 + 2 * inverse[0, 1] * x * y + inverse[1, 1] * y**2
    numerator = np.sqrt(np.maximum(scale**2 - squared, 0))
    measure = half[:, None, None] * width * weights[None, :, None] * weights[None, None, :]
    return np.sum(numerator * u[:, :, None] * measure, axis=(1, 2)), np.sum(numerator * measure, axis=(1, 2))


def sampled_stress(observations, points, forces):
    is_nonzero = forces != 0
    return halfspace_stress(observations, points[is_nonzero], forces[is_nonzero])


def fit_elliptic(points, width, inverse, offset, offset_fraction, matrix, gap, load):
    """Minimize trial complementary energy independently of reference pressures."""

    def trial(scale):
        regular, edge = elliptic_shapes(points, width, inverse, scale, offset)
        regular *= load / regular.sum()
        edge *= load / edge.sum()
        difference = edge - regular
        fraction = np.clip(-difference @ (matrix @ regular + gap) / (difference @ matrix @ difference), 0, 1)
        forces = regular + fraction * difference
        energy = (0.5 * forces @ matrix @ forces + gap @ forces) / (load * np.max(np.abs(gap)))
        return energy, fraction, forces

    begun = perf_counter()
    lower, upper = abs(offset_fraction) + 0.03, np.sqrt(offset_fraction**2 + 2.2**2)
    fit = minimize_scalar(
        lambda scale: trial(scale)[0], bounds=(lower, upper), method="bounded", options={"xatol": 1e-6}
    )
    assert fit.success and min(fit.x - lower, upper - fit.x) > 1e-3
    energy, fraction, forces = trial(fit.x)
    return forces, fit.x, fraction, perf_counter() - begun


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prior", type=Path, required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--lookup", type=Path)
    cli.add_argument("--only", nargs="+", choices=["captured-franka", "anisotropic-0", "anisotropic-0.785398"])
    cli.add_argument("--fit-anisotropic", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads(args.prior.read_text())
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(path) for path in [args.prior, args.frame, args.geometry, args.boundaries]],
            "lookup_input": file_id(args.lookup) if args.lookup is not None else None,
            "selected_cases": args.only,
            "fit_anisotropic": args.fit_anisotropic,
            "targets": {
                "pressure_cell_l2_relative": 0.15,
                "sample_tensor_relative": 0.10,
                "sample_vm_peak_relative": 0.15,
                "reference_tensor_change": 0.02,
                "reference_vm_peak_change": 0.05,
            },
            "scope": "Reuse circular variational trials to build a nine-node one-parameter lookup. Test held-out offsets and anisotropic/rotated gaps, including the exact captured Franka finite-face boundary. Independent unilateral BEM chooses its own pressure/support. Surface singular and continuous global peaks, shell fields, friction and native scene throughput require separate checks.",
        },
    )
    if args.lookup is None:
        count = 72
        width = 4.8 / count
        axis = (np.arange(count) + 0.5) * width
        points = np.array(np.meshgrid(axis, axis - 2.4, indexing="ij")).reshape(2, -1).T
        matrix = square_compliance(points, width, 1.0)
        load = 4 / 3
        table = [{"offset": -1.0, "scale": 1.0, "fraction": 0.0, "source": "unclipped Hertz limit"}]
        for row in prior["rows"]:
            if row["grid"] == count:
                fit = row["fitting"]["mixed_ritz"]
                table.append(
                    {
                        "offset": row["offset_over_hertz_radius"],
                        "scale": fit["radius_m"] / prior["hertz_radius_m"],
                        "fraction": fit["fraction_edge"],
                        "source": "retained circular variational fit",
                    }
                )
        training = []
        for offset in [-0.75, -0.25, 0.25, 1.0]:
            gap = ((points[:, 0] + offset) ** 2 + points[:, 1] ** 2) / 2

            def trial(scale):
                regular, edge = pressure_shapes(points, width, scale, offset, 20)
                regular *= load / regular.sum()
                edge *= load / edge.sum()
                difference = edge - regular
                fraction = np.clip(-difference @ (matrix @ regular + gap) / (difference @ matrix @ difference), 0, 1)
                forces = regular + fraction * difference
                energy = (0.5 * forces @ matrix @ forces + gap @ forces) / load
                return energy, fraction, forces

            begun = perf_counter()
            lower, upper = abs(offset) + 0.03, np.sqrt(offset**2 + 2.2**2)
            fit = minimize_scalar(
                lambda scale: trial(scale)[0], bounds=(lower, upper), method="bounded", options={"xatol": 1e-6}
            )
            assert fit.success and min(fit.x - lower, upper - fit.x) > 1e-3
            energy, fraction, forces = trial(fit.x)
            table.append(
                {
                    "offset": offset,
                    "scale": fit.x,
                    "fraction": float(fraction),
                    "source": "new circular variational fit",
                }
            )
            training.append(
                {"offset": offset, "seconds": perf_counter() - begun, "evaluations": fit.nfev, "energy": float(energy)}
            )
    else:
        reused = json.loads(args.lookup.read_text())
        table, training = reused["nodes"], reused["training"]
    table.sort(key=lambda row: row["offset"])
    offsets = np.array([row["offset"] for row in table])
    scales = np.array([row["scale"] for row in table])
    fractions = np.array([row["fraction"] for row in table])
    write_json(
        args.output / "lookup.json",
        {
            "nodes": table,
            "training": training,
            "prior": file_id(args.prior),
            "interpolation": "piecewise linear",
            "supported_offset_interval": [-1, 2],
        },
    )
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.boundaries) as saved:
        boundaries = {key: saved[key] for key in saved.files}
    is_loaded = frame["contact_valid"] & (np.linalg.norm(frame["contact_force_n"], axis=2) > 0)
    selected = np.argwhere(is_loaded & (frame["region_status"] != 0))
    assert selected.shape == (1, 2)
    i_b, i_c = selected[0]
    assert boundaries["count"][i_b, i_c] == 4
    planes = boundaries["planes"][i_b, i_c]
    distances = planes[:, 2] / np.linalg.norm(planes[:, :2], axis=1)
    i_edge = int(np.argmin(distances))
    inward = -planes[i_edge, :2] / np.linalg.norm(planes[i_edge, :2])
    tangent = np.array([-inward[1], inward[0]])
    rotation = np.column_stack((inward, tangent))
    curvature = rotation.T @ frame["region_curvature_per_m"][i_b, i_c] @ rotation
    normal = boundaries["normal"][i_b, i_c]
    tangent_world = boundaries["tangent"][i_b, i_c]
    world_frame = np.column_stack((tangent_world, np.cross(normal, tangent_world))) @ rotation
    quaternion = frame["frame_quaternion"][i_b]
    material_rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    contact_position = material_rotation.apply(frame["contact_position_m"][i_b, i_c]) + frame["frame_position_m"][i_b]
    contact_force = material_rotation.apply(frame["contact_force_n"][i_b, i_c])
    observed_center = (contact_position - frame["region_center_m"][i_b, i_c]) @ world_frame
    normal_load = -contact_force @ normal
    modulus = prior["reduced_modulus_pa"]
    axes, direction, depth, pressure = ellipse(normal_load, modulus, curvature)
    principal = np.column_stack((direction, [-direction[1], direction[0]]))
    covariance = principal @ np.diag(axes**2) @ principal.T
    edge_scale = np.sqrt(covariance[0, 0])
    actual_offset = -distances[i_edge] / edge_scale
    assert -1 <= actual_offset <= 2
    assert np.partition(distances, 1)[1] > 3 * axes[0]
    captured = {
        "environment": int(i_b),
        "contact": int(i_c),
        "same_substep_face_distances_m": distances.tolist(),
        "offset_over_hertz_edge_width": float(actual_offset),
        "curvature_in_edge_coordinates_per_m": curvature.tolist(),
        "normal_force_n": float(normal_load),
        "hertz_axes_m": axes.tolist(),
        "observed_tangent_lever_m": observed_center.tolist(),
    }
    write_json(args.output / "captured.json", captured)
    cases = [(f"circular-{d:g}", np.eye(2) * 50, d, 0.1, [72]) for d in [-0.625, 0.125, 0.75]]
    cases.append(("captured-franka", curvature, actual_offset, normal_load, [48, 72]))
    for angle in [0, np.pi / 4]:
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        cases.append((f"anisotropic-{angle:g}", turn @ np.diag([25.0, 100.0]) @ turn.T, 0.125, 0.1, [72]))
    if args.only is not None:
        cases = [case for case in cases if case[0] in args.only]
    rows, arrays = [], {}
    observations_unit = (
        np.array(
            np.meshgrid(np.linspace(-1.2, 1.8, 17), np.linspace(-1.8, 1.8, 17), np.geomspace(0.25, 2, 9), indexing="ij")
        )
        .reshape(3, -1)
        .T
    )
    for name, curvature, offset_fraction, load, grids in cases:
        axes, direction, depth, pressure = ellipse(load, modulus, curvature)
        principal = np.column_stack((direction, [-direction[1], direction[0]]))
        covariance = principal @ np.diag(axes**2) @ principal.T
        inverse = np.linalg.inv(covariance)
        offset = offset_fraction * np.sqrt(covariance[0, 0])
        scale = np.interp(offset_fraction, offsets, scales)
        fraction = np.interp(offset_fraction, offsets, fractions)
        observations = observations_unit * axes[0]
        for count in grids:
            width = 4.8 * axes[0] / count
            axis = (np.arange(count) + 0.5) * width
            points = np.array(np.meshgrid(axis, axis - 2.4 * axes[0], indexing="ij")).reshape(2, -1).T
            matrix = square_compliance(points, width, modulus)
            relative = points + [offset, 0]
            gap = np.einsum("ni,ij,nj->n", relative, curvature, relative) / 2
            reference = solve_hard_contact(matrix, gap, load)
            grid_force = reference.force.reshape(count, count)
            assert max(grid_force[-1].max(), grid_force[:, 0].max(), grid_force[:, -1].max()) < load * 1e-10
            regular, edge = elliptic_shapes(points, width, inverse, scale, offset)
            forces = load * ((1 - fraction) * regular / regular.sum() + fraction * edge / edge.sum())
            grid_force = forces.reshape(count, count)
            assert max(grid_force[-1].max(), grid_force[:, 0].max(), grid_force[:, -1].max()) < load * 1e-10
            stress = sampled_stress(observations, points, forces)
            reference_stress = sampled_stress(observations, points, reference.force)
            report = metrics(forces, reference.force, stress, reference_stress, points, load)
            if name == "captured-franka":
                centroid = points.T @ forces / load + [offset, 0]
                report["centroid_relative_to_root_m"] = centroid.tolist()
                report["normal_moment_change_over_Fa"] = float(np.linalg.norm(centroid - observed_center) / axes[0])
            regular40, edge40 = elliptic_shapes(points, width, inverse, scale, offset, order=40)
            refined = load * ((1 - fraction) * regular40 / regular40.sum() + fraction * edge40 / edge40.sum())
            report["quadrature20_to40_force_relative"] = float(
                np.linalg.norm(forces - refined) / np.linalg.norm(refined)
            )
            fitted_force, fitted_stress = np.empty(0), np.empty(0)
            if args.fit_anisotropic:
                fitted_force, fitted_scale, fitted_fraction, fitted_seconds = fit_elliptic(
                    points, width, inverse, offset, offset_fraction, matrix, gap, load
                )
                fitted_stress = sampled_stress(observations, points, fitted_force)
                report["variational"] = metrics(
                    fitted_force, reference.force, fitted_stress, reference_stress, points, load
                )
                report["variational"].update(
                    scale=float(fitted_scale), fraction=float(fitted_fraction), seconds=fitted_seconds
                )
            path = args.output / f"{name}-{count}.npz"
            np.savez_compressed(
                path,
                points_m=points,
                gap_m=gap,
                reference_force_n=reference.force,
                force_n=forces,
                observation_points_m=observations,
                tensor_pa=stress,
                reference_tensor_pa=reference_stress,
                variational_force_n=fitted_force,
                variational_tensor_pa=fitted_stress,
            )
            rows.append(
                {
                    "case": name,
                    "grid": count,
                    "offset": float(offset_fraction),
                    "scale": float(scale),
                    "fraction": float(fraction),
                    "axes_ratio": float(axes[1] / axes[0]),
                    **report,
                    "fields": file_id(path),
                }
            )
            if name == "captured-franka":
                arrays[count] = reference_stress
            write_json(args.output / "progress.json", {"rows": rows})
    difference = np.linalg.norm(arrays[48] - arrays[72]) / np.linalg.norm(arrays[72])
    peak_change = abs(von_mises(arrays[48]).max() / von_mises(arrays[72]).max() - 1)
    passed = all(row["engineering_targets_met"] and row["quadrature20_to40_force_relative"] < 1e-3 for row in rows)
    passed = passed and difference < 0.02 and peak_change < 0.05
    write_json(
        args.output / "result.json",
        {
            "passed": bool(passed),
            "rows": rows,
            "captured": captured,
            "captured_reference_tensor_change": float(difference),
            "captured_reference_vm_peak_change": float(peak_change),
            "whole_goal_complete": False,
        },
    )
    assert passed


if __name__ == "__main__":
    main()
