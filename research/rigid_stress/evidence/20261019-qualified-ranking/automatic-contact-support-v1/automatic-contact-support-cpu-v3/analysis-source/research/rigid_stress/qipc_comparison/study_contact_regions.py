"""CPU support/pressure controls and measured double spreading of saved pad loads.

This is a contact-model study, not a stress qualification. The force-controlled
column solution uses physical gaps and area, without a prescribed support radius.
Saved deformed gaps are a constitutive check, not a prediction of shell compliance.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json


@dataclass
class FoundationSolution:
    indentation: float
    forces: np.ndarray


def solve_foundation(gap, area, stiffness, load):
    """Solve sum(k*A*[delta-gap]+)=load exactly over the active sorted prefix."""
    assert gap.ndim == area.ndim == 1 and gap.shape == area.shape and len(gap)
    assert np.isfinite(gap).all() and np.isfinite(area).all()
    assert np.all(area > 0) and stiffness > 0 and load > 0
    origin = gap.min()
    shifted = gap - origin
    order = np.argsort(shifted)
    cumulative_area = np.cumsum(area[order])
    trial = (load / stiffness + np.cumsum(area[order] * shifted[order])) / cumulative_area
    last = np.flatnonzero(trial > shifted[order])[-1]
    indentation = trial[last]
    # The prefix determines the active set; use pairwise reductions for its
    # final balance. A long cumsum otherwise accumulates avoidable load error.
    for _ in range(2):
        force = stiffness * area * np.maximum(indentation - shifted, 0)
        active = indentation > shifted
        indentation += (load - force.sum()) / (stiffness * area[active].sum())
    force = stiffness * area * np.maximum(indentation - shifted, 0)
    np.testing.assert_allclose(force.sum(), load, rtol=1e-12, atol=0)
    return FoundationSolution(indentation + origin, force)


def quadratic_column_axes(curvatures, stiffness, load):
    """Positive quadratic gap only; no shell/half-space equivalence is assumed."""
    if not np.isfinite(curvatures).all() or np.any(curvatures <= 0):
        return None
    indentation = np.sqrt(load * np.sqrt(np.prod(curvatures)) / (np.pi * stiffness))
    return indentation, np.sqrt(2 * indentation / curvatures)


def distribution(points, weights, area=None):
    assert np.all(weights >= 0) and np.isfinite(weights).all()
    total = weights.sum()
    if total == 0:
        return None
    center = weights @ points / total
    relative = points - center
    covariance = (relative * weights[:, None]).T @ relative / total
    result = {
        "unsigned_load_n": total.item(),
        "centroid_material_m": center.tolist(),
        "in_plane_rms_axes_m": np.sqrt(np.maximum(np.linalg.eigvalsh(covariance[:2, :2]), 0)).tolist(),
        "covariance_material_m2": covariance.tolist(),
    }
    if area is not None:
        active = weights > 0
        assert np.all(area >= 0) and not np.any(active & (area == 0))
        pressure = np.divide(weights, area, out=np.zeros_like(weights), where=area > 0)
        result.update(
            support_projected_area_m2=area[active].sum().item(),
            support_area_equivalent_radius_m=np.sqrt(area[active].sum() / np.pi).item(),
            maximum_cell_average_pressure_pa=pressure.max().item(),
            active_samples=int(active.sum()),
        )
    return result


def analytic_controls():
    # Positive midpoint quadrature on a square; neither numerical search extent
    # nor the grid spacing is an imposed physical footprint.
    count, extent = 480, 0.006
    axis = (np.arange(count) + 0.5) * (2 * extent / count) - extent
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.reshape(-1), y.reshape(-1), np.zeros(x.size)))
    area = np.full(x.size, (2 * extent / count) ** 2)
    rows = []
    for curvatures, angle, load, stiffness in (
        ((50.0, 50.0), 0.0, 0.1, 1e9),
        ((50.0, 50.0), 0.0, 1.0, 1e9),
        ((25.0, 100.0), 0.43, 0.3, 1e9),
        ((25.0, 100.0), 0.43, 0.3, 2e9),
    ):
        curvature = np.array(curvatures)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = points[:, :2] @ rotation
        gap = np.sum(local**2 * curvature, axis=1) / 2
        expected_depth, expected_axes = quadratic_column_axes(curvature, stiffness, load)
        actual = solve_foundation(gap, area, stiffness, load)
        stats = distribution(points, actual.forces, area)
        depth_error = abs(actual.indentation / expected_depth - 1)
        area_error = abs(stats["support_projected_area_m2"] / (np.pi * np.prod(expected_axes)) - 1)
        axes_error = np.max(np.abs(np.sqrt(6) * np.array(stats["in_plane_rms_axes_m"]) / np.sort(expected_axes) - 1))
        assert depth_error < 1e-3 and area_error < 0.02 and axes_error < 0.002
        # Refining each source's area partition cannot change the physical patch.
        split = solve_foundation(np.repeat(gap, 2), np.repeat(area / 2, 2), stiffness, load)
        split_error = np.linalg.norm(split.forces.reshape(-1, 2).sum(axis=1) - actual.forces) / np.linalg.norm(actual.forces)
        assert split_error < 1e-11
        rows.append({
            "curvatures_per_m": list(curvatures), "orientation_rad": angle, "load_n": load,
            "stiffness_n_per_m3": stiffness, "analytic_axes_m": expected_axes.tolist(),
            "indentation_relative_error": depth_error.item(), "area_relative_error": area_error,
            "axes_relative_error": axes_error.item(), "split_source_force_relative_error": split_error.item(),
        })
    # Two physical regions must share a total load before their supports are
    # determined. They are not one independently predicted patch per sample.
    gap = np.minimum((x - 0.002) ** 2 + y**2, (x + 0.002) ** 2 + y**2).reshape(-1) / 0.04
    nonconvex = []
    for load in (0.1, 10.0, 0.1):
        solution = solve_foundation(gap, area, 1e9, load)
        components = label((solution.forces > 0).reshape(x.shape))[1]
        nonconvex.append({"load_n": load, "connected_regions": components, "indentation_m": solution.indentation.item()})
    assert [r["connected_regions"] for r in nonconvex] == [2, 1, 2]
    assert quadratic_column_axes(np.array([0.0, 50.0]), 1e9, 0.1) is None
    assert quadratic_column_axes(np.array([-50.0, 50.0]), 1e9, 0.1) is None
    return {"passed": True, "quadratic_cases": rows, "nonconvex_loading_unloading": nonconvex,
            "nonpositive_curvature_refused": True,
            "scope": "Rigid-link independent-column pad only. Hertz is inapplicable to this declared pad law; shell compliance and tangential history are not supplied by these formulas."}


def reference_support(frame, stiffness, output):
    with np.load(frame) as saved:
        points, current, force, area = (saved[k] for k in (
            "source_points_m", "source_points_current_m", "source_forces_n", "projected_area_m2"))
    rows, arrays = [], {}
    for name, mask, sign in (("upper", points[:, 2] > 0, -1), ("lower", points[:, 2] <= 0, 1)):
        p, q, a = points[mask], current[mask], area[mask]
        measured = sign * force[mask, 2]
        assert measured.min() >= 0 and measured.sum() > 0
        variants = {}
        for variant, geometry in (("rigid_gap_prediction", p), ("saved_deformed_gap_check", q)):
            height = sign * geometry[:, 2]
            gap = height - height.min()
            solution = solve_foundation(gap, a, stiffness, measured.sum())
            error = np.sqrt(np.sum((solution.forces - measured) ** 2 / a) / np.sum(measured**2 / a))
            if variant == "saved_deformed_gap_check":
                assert error < 1e-8, "Saved source pressure does not obey the declared instantaneous normal pad law"
            variants[variant] = {
                "pressure_area_l2_relative_error": error.item(),
                "indentation_from_geometry_extreme_m": solution.indentation.item(),
                "distribution": distribution(p, solution.forces, a),
                "load_outside_measured_support_fraction": (solution.forces[measured == 0].sum() / measured.sum()).item(),
            }
            arrays[f"{name}_{variant}_normal_force_n"] = solution.forces
        arrays[f"{name}_points_m"], arrays[f"{name}_area_m2"], arrays[f"{name}_normal_force_n"] = p, a, measured
        rows.append({"side": name, "measured": distribution(p, measured, a), "variants": variants})
    path = output / "reference-support.npz"
    np.savez_compressed(path, **arrays)
    return {"frame": file_id(frame), "maps": file_id(path), "sides": rows,
            "scope": "Same measured side loads and quadrature. Rigid-gap pressure is a prediction with no shell compliance. Saved deformed gaps only check the constitutive law; they cannot validate a compliance predictor. This is a coarse admitted physical history, not the pending final spatial qualification."}


def double_spreading(run, capture, level, mesh, output):
    receipt = json.loads((capture / "result.json").read_text())
    assert receipt["passed"]
    case = next(c for c in receipt["cases"] if c["level"] == level)
    original = run / f"frames/stress-level{level}.npz"
    physical = run / f"frames/level{level}.npz"
    quadrature = Path(case["quadrature"]["path"])
    assert file_id(original) == case["original"] and file_id(quadrature) == case["quadrature"]
    assert file_id(physical) == case["physical_state"]
    assert case["capture_checks"]["passed"]
    start = json.loads((run / "start.json").read_text())
    terminal = json.loads((run / "result.json").read_text())
    assert terminal["completed"] and terminal["mechanical_history_passed"]
    assert start["args"]["source_reduction"] == "none" and start["args"]["radius"] == 0.002
    assert file_id(mesh) == start["mesh"]
    with np.load(mesh) as saved:
        triangle = saved["outer_vertices"][saved["mid_faces"]]
    with np.load(original) as saved:
        points, forces, coordinates = (saved[k] for k in (
            "original_contact_points_m", "original_contact_forces_n", "vertices_m"))
    with np.load(physical) as saved:
        rotation = saved["rotation"]
        np.testing.assert_allclose(saved["contact_force_n"] @ rotation, forces, rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(points, triangle.mean(axis=1), rtol=0, atol=1e-14)
    projected = np.abs(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])[:, 2]) / 2
    with np.load(quadrature) as saved:
        sample_points, sample_forces, nodes, ids = (saved[k] for k in (
            "points_material_m", "quadrature_forces_n", "p2_nodes", "source_sample_face"))
    # Identify complete parent facets by their actual corner coordinates. A
    # surface that cannot be matched is reported, never folded into an outer cell.
    _, first, inverse = np.unique(ids[:, 2], return_index=True, return_inverse=True)
    distance, face = cKDTree(points).query(coordinates[nodes[first, :3]].mean(axis=1))
    matched = distance < 1e-13
    assert np.all(matched), "Actual reconstruction includes unmatched/inner facets; retain and classify before comparing outer pressure"
    face = face[inverse]
    reconstructed = np.zeros_like(forces)
    np.add.at(reconstructed, face, sample_forces)
    unsigned = np.linalg.norm(forces, axis=1).sum()
    force_error = np.linalg.norm(sample_forces.sum(axis=0) - forces.sum(axis=0)) / unsigned
    moment_error = np.linalg.norm(np.cross(sample_points, sample_forces).sum(axis=0) - np.cross(points, forces).sum(axis=0)) / (0.03 * unsigned)
    assert max(force_error, moment_error) < 1e-8
    original_world, reconstructed_world, sample_world = forces @ rotation.T, reconstructed @ rotation.T, sample_forces @ rotation.T
    rows = []
    for name, mask in (("upper", points[:, 2] > 0), ("lower", points[:, 2] <= 0)):
        sample_mask = sample_points[:, 2] > 0 if name == "upper" else sample_points[:, 2] <= 0
        normal, actual = np.abs(original_world[:, 2]), np.abs(reconstructed_world[:, 2])
        active = mask & (normal > 0)
        valid_area = mask & (projected > 0)
        pressure_error = np.sqrt(np.sum((actual[valid_area] - normal[valid_area]) ** 2 / projected[valid_area]) / np.sum(normal[valid_area]**2 / projected[valid_area]))
        normal_sample = np.abs(sample_world[sample_mask, 2])
        tangent_sample = np.linalg.norm(sample_world[sample_mask, :2], axis=1)
        rows.append({
            "side": name,
            "original_normal": distribution(points[mask], normal[mask], projected[mask]),
            "reconstructed_normal_cell_averages": distribution(points[mask], actual[mask], projected[mask]),
            "reconstructed_normal_actual_quadrature": distribution(sample_points[sample_mask], normal_sample),
            "original_tangent": distribution(points[mask], np.linalg.norm(original_world[mask, :2], axis=1)),
            "reconstructed_tangent_actual_quadrature": distribution(sample_points[sample_mask], tangent_sample),
            "normal_pressure_area_l2_relative_difference": pressure_error.item(),
            "normal_load_outside_original_active_faces_fraction": (actual[mask & ~active].sum() / normal[mask].sum()).item(),
        })
    path = output / f"{start['args']['method']}-level{level}-face-maps.npz"
    np.savez_compressed(path, points_material_m=points, projected_area_m2=projected,
                        original_forces_world_n=original_world, reconstructed_forces_world_n=reconstructed_world)
    return {"method": start["args"]["method"], "level": level, "original": file_id(original),
            "physical_state": case["physical_state"],
            "capture_receipt": file_id(capture / "result.json"), "actual_quadrature": case["quadrature"],
            "mesh": file_id(mesh), "face_maps": file_id(path), "sides": rows,
            "force_relative_error": force_error.item(), "moment_relative_error": moment_error.item(),
            "scope": "Read-only actual production reconstruction, not a surrogate Gaussian convolution. Positive normal force uses the physical plate direction; tangential weights use unsigned force magnitude. Pressure differences compare common original face cell averages. Exact nonzero support retains every force residue; moments use actual quadrature positions."}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--solid-frame", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--native-run", type=Path, required=True)
    cli.add_argument("--capture", type=Path, required=True)
    cli.add_argument("--levels", type=int, nargs="+", default=[0, 1])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    controls = analytic_controls()
    write_json(args.output / "analytic-controls.json", controls)
    reference = reference_support(args.solid_frame, 1e9, args.output)
    write_json(args.output / "reference-support.json", reference)
    cases = []
    for level in args.levels:
        cases.append(double_spreading(args.native_run, args.capture, level, args.mesh, args.output))
        write_json(args.output / "progress.json", {"cases": cases})
    write_json(args.output / "result.json", {"passed": True, "analytic_controls": controls,
        "reference_support": reference, "double_spreading": cases, "elapsed_seconds": perf_counter() - started,
        "whole_goal_complete": False, "native_implementation_admitted": False,
        "remaining": "Coupled shell/pad compliance prediction, stress/peak comparison, changing convex/nonconvex/friction cases and measured Quadrants integration remain required."})


if __name__ == "__main__":
    main()
