"""Measure the captured vertex-gap discrepancy with a fixed-compliance contact control."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, halfspace_stress, solve_hard_contact, square_compliance
from .hertz_stress import elliptic_stress


def surface_depth(points, triangles, normals):
    """First material intersection of source-normal rays with actual outer faces."""
    facing = normals[:, 2] < -1e-10
    triangles = triangles[facing]
    edges = triangles[:, 1:, :2] - triangles[:, :1, :2]
    inverse = np.linalg.inv(edges.transpose(0, 2, 1))
    local = np.einsum("fij,fnj->fni", inverse, points[None] - triangles[:, None, 0, :2])
    bary = np.concatenate([1 - local.sum(axis=2, keepdims=True), local], axis=2)
    inside = (bary.min(axis=2) >= -1e-10) & (bary.max(axis=2) <= 1 + 1e-10)
    depth = np.einsum("fni,fi->fn", bary, triangles[:, :, 2])
    result = np.min(np.where(inside, depth, np.inf), axis=0)
    assert np.isfinite(result).all() and result.min() > -1e-12
    return np.maximum(result, 0)


def von_mises(tensor):
    deviation = tensor - np.trace(tensor, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * (deviation * deviation).sum(axis=(1, 2)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"reference_tensor_change": 0.02, "reference_vm_peak_change": 0.05,
               "reference_second_moment_change": 0.05}
    write_json(args.output / "start.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "frame": file_id(args.frame), "targets": targets,
        "scope": "Captured first loaded Franka vertex: exact finite-facet gap versus the smoothed quadratic gap. The independent contact solve holds Boussinesq half-space compliance fixed and uses the faceted gap as a rigid obstacle. This isolates geometry sensitivity, not the elasticity of the faceted hollow shell. Positive-depth sampled stresses exclude surface/global peak qualification. No production gate is changed.",
    })
    with np.load(args.frame) as saved:
        data = {key: saved[key] for key in saved.files}
    batch = len(data["input_status"])
    patch = int(np.flatnonzero(data["law_force_n"] > 0)[0])
    center, frame = data["hertz_centers_m"][patch], data["hertz_frames"][patch]
    direction = data["law_direction"][patch]
    rotation = np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    basis = frame.copy()
    basis[:, :2] = basis[:, :2] @ rotation
    curvature = rotation.T @ data["region_curvature_per_m"][patch % batch, patch // batch] @ rotation
    load, axes = data["law_force_n"][patch], data["law_axes_m"][patch]
    peak = data["law_pressure_pa"][patch]
    modulus = 1e10 / (1 - 0.3**2)
    exact_axes, _, indentation, exact_peak = ellipse(load, modulus, curvature)
    np.testing.assert_allclose(axes, exact_axes, rtol=5e-4)
    np.testing.assert_allclose(peak, exact_peak, rtol=5e-4)
    nodes = (data["material_nodes_m"] - center) @ basis
    triangles = nodes[data["material_surface_nodes"][:3].T]
    normals = data["material_face_normal"] @ basis
    offsets = data["probe_offsets"] * np.r_[axes, axes[0]]
    depth = surface_depth(offsets[:, :2], triangles, normals)
    quadratic_depth = 0.5 * np.einsum("ni,ij,nj->n", offsets[:, :2], curvature, offsets[:, :2])
    angle = 2 * np.pi * (np.arange(128) + 0.5) / 128
    unit_circle = np.column_stack([np.cos(angle), np.sin(angle)])
    directional_slope = surface_depth(unit_circle * axes[0], triangles, normals) / axes[0]
    assert directional_slope.min() > 0
    # A homogeneous linear gap balances E* m a^2 with F. This is a mesh
    # extent estimate only: the unilateral solve determines its own support.
    scale = np.sqrt(load / (modulus * np.mean(directional_slope)))
    observations = np.array(np.meshgrid(np.linspace(-2, 2, 13), np.linspace(-2, 2, 13),
                                        np.geomspace(0.5, 4, 7), indexing="ij")).reshape(3, -1).T * scale
    hertz_tensor = elliptic_stress(observations, axes, peak=peak, order=96)
    rows, fields = [], []
    for count in (32, 48):
        extent = 3 * scale
        width = 2 * extent / count
        axis = (np.arange(count) + 0.5) * width - extent
        points = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
        gap = surface_depth(points, triangles, normals)
        matrix = square_compliance(points, width, modulus)
        solved = solve_hard_contact(matrix, gap, load)
        grid_force = solved.force.reshape(count, count)
        assert max(grid_force[[0, -1]].max(), grid_force[:, [0, -1]].max()) < load * 1e-10
        active = solved.force > load * 1e-12
        tensor = halfspace_stress(observations, points[active], solved.force[active])
        fields.append(tensor)
        covariance = (points * solved.force[:, None]).T @ points / load
        centroid = points.T @ solved.force / load
        row = {"grid": count, "width_m": width, "active_cells": int(active.sum()),
               "indentation_m": solved.indentation, "iterations": solved.iterations,
               "force_relative_error": float(abs(solved.force.sum() / load - 1)),
               "centroid_m": centroid.tolist(), "second_moment_m2": covariance.tolist(),
               "support_area_m2": float(active.sum() * width**2),
               "hertz_area_ratio": float(np.pi * axes.prod() / (active.sum() * width**2)),
               "hertz_tensor_relative_to_faceted_gap": float(np.linalg.norm(hertz_tensor - tensor) / np.linalg.norm(tensor)),
               "hertz_vm_peak_relative_to_faceted_gap": float(abs(von_mises(hertz_tensor).max() / von_mises(tensor).max() - 1)),
               "sample_vm_peak_pa": float(von_mises(tensor).max())}
        path = args.output / f"faceted-gap-{count}.npz"
        np.savez_compressed(path, points_m=points, geometric_gap_m=gap, force_n=solved.force,
                            residual_gap_m=solved.gap, observation_points_m=observations,
                            tensor_pa=tensor, hertz_tensor_pa=hertz_tensor)
        rows.append({**row, "fields": file_id(path)})
        write_json(args.output / "progress.json", {"rows": rows})
    reference_change = {
        "tensor": float(np.linalg.norm(fields[0] - fields[1]) / np.linalg.norm(fields[1])),
        "vm_peak": float(abs(von_mises(fields[0]).max() / von_mises(fields[1]).max() - 1)),
        "second_moment": float(np.linalg.norm(np.array(rows[0]["second_moment_m2"]) - rows[1]["second_moment_m2"])
                               / np.linalg.norm(rows[1]["second_moment_m2"])),
    }
    path = args.output / "geometry.npz"
    np.savez_compressed(path, probe_offsets_m=offsets, facet_depth_m=depth, quadratic_depth_m=quadratic_depth,
                        directional_slope=directional_slope, curvature_per_m=curvature, axes_m=axes)
    report = {"passed": False, "targets": targets, "rows": rows, "reference_change": reference_change,
              "patch": patch, "load_n": load, "native_axes_m": axes.tolist(),
              "hertz_indentation_m": indentation, "facet_gap_scale_m": scale,
              "outside_faceted_material_probes": int(np.count_nonzero(offsets[:, 2] < depth - 1e-12)),
              "probe_count": len(offsets), "maximum_facet_depth_m": float(depth.max()),
              "maximum_quadratic_depth_m": float(quadratic_depth.max()),
              "maximum_facet_depth_over_hertz_indentation": float(depth.max() / indentation),
              "geometry": file_id(path), "whole_goal_complete": False}
    write_json(args.output / "result.json", report)
    assert reference_change["tensor"] < targets["reference_tensor_change"], reference_change
    assert reference_change["vm_peak"] < targets["reference_vm_peak_change"], reference_change
    assert reference_change["second_moment"] < targets["reference_second_moment_change"], reference_change
    report["passed"] = True
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
