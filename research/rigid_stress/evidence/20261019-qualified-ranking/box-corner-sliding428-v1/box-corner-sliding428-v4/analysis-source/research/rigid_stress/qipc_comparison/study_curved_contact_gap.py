"""Test captured Hertz inputs against independent smooth and reconstructed-gap contact."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, ellipse_pressure, halfspace_stress, solve_hard_contact, square_compliance
from .study_franka_contact_gap import von_mises
from .study_quadratic_rest_surface import exact_surface, implicit, quadratic_basis


def curved_depth(points, nodes, interior):
    triangles = nodes[:, :3]
    area = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    center = triangles.mean(axis=1)
    # The first intersection is on the incoming side of this convex control.
    normals = area * np.sign(np.einsum("fi,fi->f", area, center-interior))[:, None]
    nodes = nodes[normals[:, 2] < 0]
    triangles = nodes[:, :3]
    edges = triangles[:, 1:, :2] - triangles[:, :1, :2]
    inverse = np.linalg.inv(edges.transpose(0, 2, 1))
    coordinates = np.einsum("fij,fnj->fni", inverse, points[None]-triangles[:, None, 0, :2])
    for _ in range(8):
        bary = np.concatenate([1-coordinates.sum(axis=2, keepdims=True), coordinates], axis=2)
        values, gradient, _ = quadratic_basis(bary.reshape(-1, 3))
        values, gradient = values.reshape(*bary.shape[:2], 6), gradient.reshape(*bary.shape[:2], 6, 2)
        location = np.einsum("fqn,fni->fqi", values, nodes)
        jacobian = np.einsum("fqna,fni->fqia", gradient, nodes[:, :, :2])
        residual = location[:, :, :2]-points[None]
        coordinates -= np.linalg.solve(jacobian, residual[..., None])[..., 0]
    bary = np.concatenate([1-coordinates.sum(axis=2, keepdims=True), coordinates], axis=2)
    values, _, _ = quadratic_basis(bary.reshape(-1, 3))
    location = np.einsum("fqn,fni->fqi", values.reshape(*bary.shape[:2], 6), nodes)
    inside = (bary.min(axis=2) >= -1e-10) & (bary.max(axis=2) <= 1+1e-10)
    assert np.max(np.linalg.norm(location[:, :, :2]-points[None], axis=2)[inside]) < 1e-12
    depth = np.min(np.where(inside, location[:, :, 2], np.inf), axis=0)
    assert np.isfinite(depth).all() and depth.min() >= -1e-12
    return np.maximum(depth, 0)


def analytic_depth(points, center, basis):
    parameter = np.zeros(len(points))
    base = center + points @ basis[:, :2].T
    for _ in range(8):
        value, gradient, _ = implicit(base + parameter[:, None]*basis[:, 2], "egg")
        parameter -= value / (gradient @ basis[:, 2])
    value, _, _ = implicit(base + parameter[:, None]*basis[:, 2], "egg")
    assert np.max(np.abs(value)) < 1e-12 and parameter.min() >= -1e-12
    return np.maximum(parameter, 0)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"reference_tensor_change": 0.02, "reference_moment_change": 0.02,
               "pressure_relative": 0.05, "sample_tensor_relative": 0.05, "sample_peak_relative": 0.10}
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "frame": file_id(args.frame), "geometry": file_id(args.geometry), "targets": targets,
        "scope": "Captured normal load and local frame; independent force-controlled contact with unchanged half-space compliance. Compare the analytic egg gap and the normal-constrained quadratic rest geometry, then test native versus analytic-curvature Hertz pressure. This is a local pressure/gap control, not a finite-shell elasticity or global-peak qualification. No engine geometry or production gate is changed."})
    with np.load(args.frame) as saved:
        center, basis = saved["hertz_centers_m"][0], saved["hertz_frames"][0]
        direction = saved["law_direction"][0]
        rotation = np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
        basis[:, :2] = basis[:, :2] @ rotation
        load, native_axes, native_peak = saved["law_force_n"][0], saved["law_axes_m"][0], saved["law_pressure_pa"][0]
    with np.load(args.geometry) as saved:
        # The retained control has80 outer triangles, two wall layers and
        # three tetrahedra per triangular prism, in the producer's declared order.
        surface = {kind: (saved[kind+"_volume_nodes_m"][:240:3, [0, 1, 2, 4, 5, 7]]-center) @ basis
                   for kind in ["analytic", "estimated"]}
    _, normal, tensor = exact_surface(center[None], "egg")
    np.testing.assert_allclose(normal[0], -basis[:, 2], rtol=0, atol=1e-12)
    curvature = basis[:, :2].T @ tensor[0] @ basis[:, :2]
    modulus = 1e10 / 0.91
    axes, direction, _, peak = ellipse(load, modulus, curvature)
    extent = 1.6 * max(native_axes.max(), axes.max())
    observation_axis = np.linspace(-1.5, 1.5, 13) * axes.max()
    observations = np.array(np.meshgrid(observation_axis, observation_axis,
                           np.geomspace(0.4, 2, 7)*axes.max(), indexing="ij")).reshape(3, -1).T
    rows, fields, moments = [], {}, {}
    for count in [32, 48]:
        width = 2*extent/count
        axis = (np.arange(count)+0.5)*width-extent
        points = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
        matrix = square_compliance(points, width, modulus)
        gaps = {"analytic_egg": analytic_depth(points, center, basis)}
        gaps.update({kind+"_quadratic": curved_depth(points, nodes, -center@basis) for kind, nodes in surface.items()})
        gauss, weights = np.polynomial.legendre.leggauss(16)
        offsets = np.array(np.meshgrid(gauss, gauss, indexing="ij")).reshape(2, -1).T * width/2
        weights = np.outer(weights, weights).reshape(-1)/4
        samples = (points[:, None]+offsets[None]).reshape(-1, 2)
        candidates = {}
        for kind, candidate_axes, candidate_direction, candidate_peak in [
            ("native_hertz", native_axes, np.array([1.0, 0]), native_peak),
            ("analytic_hertz", axes, direction, peak),
        ]:
            force = ellipse_pressure(samples, candidate_axes, candidate_direction, candidate_peak).reshape(len(points), -1) @ weights * width**2
            mass_error = abs(force.sum()/load-1)
            assert mass_error < 1e-3
            force *= load/force.sum()
            candidates[kind] = (force, halfspace_stress(observations, points, force), mass_error)
        for kind, gap in gaps.items():
            solved = solve_hard_contact(matrix, gap, load)
            grid = solved.force.reshape(count, count)
            assert max(grid[[0, -1]].max(), grid[:, [0, -1]].max()) < load*1e-10
            active = solved.force > load*1e-12
            field = halfspace_stress(observations, points[active], solved.force[active])
            second = (points*solved.force[:, None]).T @ points/load
            fields[kind, count], moments[kind, count] = field, second
            errors = {}
            for candidate, (force, candidate_field, mass_error) in candidates.items():
                tensor_error = np.linalg.norm(candidate_field-field)/np.linalg.norm(field)
                pressure_error = np.linalg.norm(force-solved.force)/np.linalg.norm(solved.force)
                vm_error = abs(von_mises(candidate_field).max()/von_mises(field).max()-1)
                tensile_error = abs(np.linalg.eigvalsh(candidate_field)[:, -1].max()/np.linalg.eigvalsh(field)[:, -1].max()-1)
                errors[candidate] = {"pressure_cell_relative": float(pressure_error),
                                     "sample_tensor_relative": float(tensor_error),
                                     "sample_vm_peak_relative": float(vm_error),
                                     "sample_tensile_peak_relative": float(tensile_error),
                                     "raw_pressure_quadrature_force_error": float(mass_error),
                                     "pressure_and_tensor_targets_met": bool(pressure_error < targets["pressure_relative"]
                                                                             and tensor_error < targets["sample_tensor_relative"]
                                                                             and max(vm_error, tensile_error) < targets["sample_peak_relative"])}
            path = args.output / f"{kind}-{count}.npz"
            np.savez_compressed(path, positions_m=points, gap_m=gap, force_n=solved.force,
                                residual_gap_m=solved.gap, tensor_pa=field, observation_points_m=observations,
                                **{name+"_force_n": value[0] for name, value in candidates.items()})
            rows.append({"gap": kind, "grid": count, "active_cells": int(active.sum()),
                         "indentation_m": solved.indentation, "iterations": solved.iterations,
                         "second_moment_m2": second.tolist(), "candidates": errors, "fields": file_id(path)})
            write_json(args.output / "progress.json", {"rows": rows})
    changes = {kind: {"tensor_relative": float(np.linalg.norm(fields[kind, 32]-fields[kind, 48])/np.linalg.norm(fields[kind, 48])),
                     "second_moment_relative": float(np.linalg.norm(moments[kind, 32]-moments[kind, 48])/np.linalg.norm(moments[kind, 48]))}
               for kind in gaps}
    differences = {kind: float(np.linalg.norm(fields[kind, 48]-fields["analytic_egg", 48])/np.linalg.norm(fields["analytic_egg", 48]))
                   for kind in gaps}
    report = {"passed": False, "rows": rows, "targets": targets, "reference_changes": changes,
              "gap_model_tensor_difference": differences, "native_axes_m": native_axes.tolist(),
              "analytic_axes_m": axes.tolist(), "native_pressure_pa": native_peak, "analytic_pressure_pa": peak,
              "whole_goal_complete": False}
    write_json(args.output / "result.json", report)
    assert all(row["tensor_relative"] < targets["reference_tensor_change"] and row["second_moment_relative"] < targets["reference_moment_change"]
               for row in changes.values()), changes
    report["passed"] = True
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
