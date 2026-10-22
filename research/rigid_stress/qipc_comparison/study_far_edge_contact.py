"""Resolve sharp-edge contacts using captured finite faces and actual curved material gaps."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, halfspace_stress, rectangular_compliance, solve_hard_contact
from .study_franka_contact_gap import von_mises
from .study_quadratic_rest_surface import quadratic_basis


def projected_surface(nodes, face_normals, targets, plane, normal, witness, *, return_geometry=False):
    """Intersect parallel lines with facing quadratic faces in bounded vectorized blocks."""
    chord = (nodes[:, 1:3] - nodes[:, :1]).transpose(0, 2, 1)
    chart = plane @ chord
    selected = np.abs(np.linalg.det(chart)) > 1e-10 * np.linalg.norm(chart, axis=(1, 2)) ** 2
    face_indices = np.flatnonzero(selected)
    nodes, face_normals, chart = nodes[selected], face_normals[selected], chart[selected]
    results, normals, indices = [], [], []
    for start in range(0, len(targets), 128):
        target = targets[start : start + 128]
        local = np.linalg.solve(chart[:, None], ((target[None] - nodes[:, None, 0]) @ plane.T)[..., None])[..., 0]
        is_regular = np.ones(local.shape[:2], dtype=bool)
        for iteration in range(10):
            bary = np.concatenate((1 - local.sum(axis=2, keepdims=True), local), axis=2)
            values, gradients, _ = quadratic_basis(bary.reshape(-1, 3))
            values = values.reshape(*local.shape[:2], 6)
            gradients = gradients.reshape(*local.shape[:2], 6, 2)
            positions = np.einsum("fqn,fni->fqi", values, nodes)
            tangents = np.einsum("fqna,fni->fqia", gradients, nodes)
            jacobian = plane @ tangents
            is_regular &= np.abs(np.linalg.det(jacobian)) > 1e-12 * np.linalg.norm(jacobian, axis=(2, 3)) ** 2
            jacobian[~is_regular] = np.eye(2)
            residual = (positions - target[None]) @ plane.T
            if iteration < 9:
                update = np.linalg.solve(jacobian, residual[..., None])[..., 0]
                update[~is_regular] = 0
                local -= update
        area = np.cross(tangents[..., 0], tangents[..., 1])
        area *= np.sign(np.einsum("fqi,fi->fq", area, face_normals))[..., None]
        is_inside = is_regular & (bary.min(axis=2) >= -1e-10) & (bary.max(axis=2) <= 1 + 1e-10)
        is_inside &= np.linalg.norm(residual, axis=2) < 1e-11
        is_inside &= area @ normal > 0
        assert np.all(is_inside.any(axis=0)), np.flatnonzero(~is_inside.any(axis=0)).tolist()
        distance = np.linalg.norm(positions - witness, axis=2)
        distance[~is_inside] = np.inf
        index = distance.argmin(axis=0)
        results.append(positions[index, np.arange(len(target))])
        if return_geometry:
            selected_area = area[index, np.arange(len(target))]
            normals.append(selected_area / np.linalg.norm(selected_area, axis=1, keepdims=True))
            indices.append(face_indices[index])
    if return_geometry:
        return np.concatenate(results), np.concatenate(normals), np.concatenate(indices)
    return np.concatenate(results)


def edge_trial(points, widths, extent, along, shift, order, *, edge_exponent=0.5):
    """Integrate a parabolic footprint with an integrable sharp-edge power law."""
    assert 0 < edge_exponent < 1
    knots, weights = np.polynomial.legendre.leggauss(order)
    bounds = np.maximum(points[:, 0, None] + np.array([-0.5, 0.5]) * widths[0], 0) ** (1 - edge_exponent)
    half = (bounds[:, 1] - bounds[:, 0]) / 2
    u = bounds.mean(axis=1)[:, None] + half[:, None] * knots
    y = points[:, 1, None] + widths[1] * knots / 2 - shift
    profile = (
        np.maximum(1 - u[:, :, None] ** (1 / (1 - edge_exponent)) / extent - y[:, None, :] ** 2 / along**2, 0)
        ** edge_exponent
    )
    measure = (
        half[:, None, None] * widths[1] * weights[None, :, None] * weights[None, None, :] / (2 * (1 - edge_exponent))
    )
    return np.sum(profile * measure, axis=(1, 2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--contact", type=int, required=True)
    cli.add_argument("--grids", nargs="+", type=int, default=[24, 36, 48])
    cli.add_argument("--prior", type=Path)
    cli.add_argument("--extent-m", nargs=2, type=float)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.frame), file_id(args.boundaries)]
            + ([file_id(args.prior)] if args.prior is not None else []),
            "trial_parameters_reused": args.prior is not None,
            "domain_extent_m": args.extent_m,
            "targets": {
                "pressure_cell_l2": 0.15,
                "sample_tensor_relative": 0.10,
                "sample_vm_peak_relative": 0.15,
                "reference_tensor_change": 0.02,
                "reference_vm_peak_change": 0.05,
            },
            "scope": "Offline unilateral normal-contact BEM on a finite half-plane, using the captured actual P2 boundary gap and constant rectangular cells. Independent active-set KKT solve determines support and pressure. A two-scale parabolic edge trial minimizes complementary energy without fitting reference pressure. Box face, load, force-line moment and original refusal remain visible. Elastic half-space and fixed friction direction remain approximations. Positive-depth samples do not qualify surface/global peaks or hollow-shell fields. No native acceptance or runtime default changes.",
        },
    )
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.boundaries) as saved:
        bounds = {key: saved[key] for key in saved.files}
    env, contact = 0, args.contact
    assert bounds["count"][env, contact] == 4 and frame["contact_valid"][env, contact]
    quaternion = frame["frame_quaternion"][env]
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    root = frame["region_center_m"][env, contact]
    normal, tangent = bounds["normal"][env, contact], bounds["tangent"][env, contact]
    basis = np.column_stack((tangent, np.cross(normal, tangent)))
    planes = bounds["planes"][env, contact]
    distances = planes[:, 2] / np.linalg.norm(planes[:, :2], axis=1)
    edge = distances.argmin()
    inward = -planes[edge, :2] / np.linalg.norm(planes[edge, :2])
    edge_rotation = np.column_stack((inward, [-inward[1], inward[0]]))
    world_basis = basis @ edge_rotation
    curvature = edge_rotation.T @ frame["region_curvature_per_m"][env, contact] @ edge_rotation
    offset = -distances[edge]
    assert offset > 0
    force = rotation.apply(frame["contact_force_n"][env, contact])
    load = -force @ normal
    modulus = 1e10 / 0.91
    hertz_axes, _, _, _ = ellipse(load, modulus, curvature)
    quadratic_shift = -curvature[0, 1] * offset / curvature[1, 1]
    nodes = frame["material_nodes_m"][frame["material_surface_nodes"].T]
    material_normal, material_plane = rotation.inv().apply(normal), rotation.inv().apply(world_basis.T)
    witness = frame["source_witness_m"][env, contact]
    world_origin = frame["frame_position_m"][env]

    def material_points(xy):
        world = root + (xy + [offset, 0]) @ world_basis.T
        targets = rotation.inv().apply(world - world_origin)
        return projected_surface(
            nodes, frame["material_face_normal"], targets, material_plane, material_normal, witness
        )

    def edge_height(value):
        return float(-material_points(np.array([[0.0, value]]))[0] @ material_normal)

    minimum = minimize_scalar(
        edge_height,
        bounds=(quadratic_shift - 0.0015, quadratic_shift + 0.0015),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert minimum.success
    shift = minimum.x
    center = material_points(np.array([[0.0, shift]]))[0]
    delta = 1e-7
    sample = material_points(np.array([[delta, shift], [0.0, shift + delta], [0.0, shift - delta]]))
    height = -(sample - center) @ material_normal
    slope = height[0] / delta
    along_curvature = (height[1] + height[2]) / delta**2
    assert slope > 0 and along_curvature > 0
    transverse = hertz_axes[0] ** 2 / offset
    along = hertz_axes[0]
    extent = np.array([10 * transverse, 8 * along] if args.extent_m is None else args.extent_m)
    prior = json.loads(args.prior.read_text()) if args.prior is not None else None
    if prior is not None:
        assert prior["contact"] == contact
    pressure_center_world = rotation.apply(center) + world_origin
    point = rotation.apply(frame["contact_position_m"][env, contact]) + world_origin
    observed_xy = (point - root) @ world_basis - [offset, shift]
    observed_material = material_points((observed_xy + [0, shift])[None])[0]
    observed_gap = -(observed_material - center) @ material_normal
    rows, previous_tensor, previous_peak = [], None, None
    unit_observations = (
        np.array(np.meshgrid(np.linspace(-0.5, 2, 13), np.linspace(-2, 2, 17), np.geomspace(0.25, 2, 7), indexing="ij"))
        .reshape(3, -1)
        .T
    )
    observations = unit_observations * along
    for count in args.grids:
        widths = extent / count
        points = (
            np.array(
                np.meshgrid(
                    (np.arange(count) + 0.5) * widths[0],
                    (np.arange(count) + 0.5) * widths[1] - extent[1] / 2,
                    indexing="ij",
                )
            )
            .reshape(2, -1)
            .T
        )
        physical = material_points(points + [0, shift])
        gap = -(physical - center) @ material_normal
        begun = perf_counter()
        matrix = rectangular_compliance(points, points, widths, modulus)
        np.testing.assert_allclose(matrix, matrix.T, rtol=3e-11, atol=1e-18)
        reference = solve_hard_contact(matrix, gap, load)
        reference_seconds = perf_counter() - begun
        grid = reference.force.reshape(count, count)
        assert max(grid[-1].max(), grid[:, 0].max(), grid[:, -1].max()) < 1e-10 * load
        normalizer = load * max(np.max(np.abs(gap)), np.max(np.diag(matrix)) * load)

        def energy(parameters):
            trial = edge_trial(
                points,
                widths,
                transverse * np.exp(parameters[0]),
                along * np.exp(parameters[1]),
                along * parameters[2],
                16,
            )
            if trial.sum() == 0:
                return 1e20
            trial *= load / trial.sum()
            return (0.5 * trial @ matrix @ trial + gap @ trial) / normalizer

        begun = perf_counter()
        evaluations = 0
        if prior is None:
            fit = minimize(
                energy,
                [0.0, 0.0, 0.0],
                method="Powell",
                bounds=[(-2.5, 1.5), (-1.5, 1), (-0.5, 0.5)],
                options={"xtol": 1e-4, "ftol": 1e-10, "maxiter": 80},
            )
            assert fit.success
            x_extent, y_extent, y_shift = transverse * np.exp(fit.x[0]), along * np.exp(fit.x[1]), along * fit.x[2]
            evaluations = fit.nfev
        else:
            x_extent, y_extent = prior["rows"][-1]["trial_extent_m"]
            y_shift = prior["rows"][-1]["trial_shift_m"]
        trial = edge_trial(points, widths, x_extent, y_extent, y_shift, 32)
        trial *= load / trial.sum()
        fitted_seconds = perf_counter() - begun
        is_nonzero = reference.force > 0
        reference_tensor = halfspace_stress(observations, points[is_nonzero], reference.force[is_nonzero])
        is_nonzero = trial > 0
        tensor = halfspace_stress(observations, points[is_nonzero], trial[is_nonzero])
        vm, reference_vm = von_mises(tensor), von_mises(reference_tensor)
        centroid = physical.T @ reference.force / load
        centroid_world = rotation.apply(centroid) + world_origin
        moment = np.cross(centroid_world - point, force)
        pressure_error = np.linalg.norm(trial - reference.force) / np.linalg.norm(reference.force)
        tensor_error = np.linalg.norm(tensor - reference_tensor) / np.linalg.norm(reference_tensor)
        peak_error = abs(vm.max() - reference_vm.max()) / reference_vm.max()
        converged = None
        if previous_tensor is not None:
            converged = {
                "tensor_change": float(
                    np.linalg.norm(reference_tensor - previous_tensor) / np.linalg.norm(reference_tensor)
                ),
                "vm_peak_change": float(abs(reference_vm.max() - previous_peak) / reference_vm.max()),
            }
        rows.append(
            {
                "grid": count,
                "cell_widths_m": widths.tolist(),
                "active_cells": int(np.count_nonzero(reference.force)),
                "reference_iterations": reference.iterations,
                "reference_seconds": reference_seconds,
                "reference_cell_average_pressure_max_pa": float(reference.force.max() / widths.prod()),
                "reference_centroid_edge_xy_m": (points.T @ reference.force / load).tolist(),
                "reference_moment_difference_nm": moment.tolist(),
                "reference_moment_difference_over_Fa": float(np.linalg.norm(moment) / (load * hertz_axes[0])),
                "trial_extent_m": [float(x_extent), float(y_extent)],
                "trial_shift_m": float(y_shift),
                "trial_offline_seconds": fitted_seconds,
                "trial_objective_evaluations": evaluations,
                "reference_indentation_m": float(reference.indentation),
                "observed_gap_above_minimum_m": float(observed_gap),
                "observed_gap_over_reference_indentation": float(observed_gap / reference.indentation),
                "observed_edge_xy_m": observed_xy.tolist(),
                "pressure_cell_l2_relative": float(pressure_error),
                "sample_tensor_relative": float(tensor_error),
                "sample_vm_peak_relative": float(peak_error),
                "reference_vm_peak_pa": float(reference_vm.max()),
                "trial_targets_met": bool(pressure_error < 0.15 and tensor_error < 0.10 and peak_error < 0.15),
                "reference_refinement": converged,
            }
        )
        np.savez_compressed(
            args.output / f"grid{count}.npz",
            points_m=points,
            material_points_m=physical,
            gap_m=gap,
            reference_force_n=reference.force,
            trial_force_n=trial,
            observations_m=observations,
            reference_tensor_pa=reference_tensor,
            trial_tensor_pa=tensor,
        )
        previous_tensor, previous_peak = reference_tensor, reference_vm.max()
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "contact": contact,
            "root_offset_m": float(offset),
            "root_offset_over_hertz_major": float(offset / hertz_axes[0]),
            "hertz_axes_m": hertz_axes.tolist(),
            "edge_stationary_shift_m": float(shift),
            "quadratic_root_shift_m": float(quadratic_shift),
            "edge_slope": float(slope),
            "edge_along_curvature_per_m": float(along_curvature),
            "edge_minimum_world_m": pressure_center_world.tolist(),
            "observed_contact_world_m": point.tolist(),
            "normal_load_n": float(load),
            "tangential_over_normal": float(np.linalg.norm(force + load * normal) / load),
            "rows": rows,
            "global_surface_peak_qualified": False,
            "native_scene_admitted": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
