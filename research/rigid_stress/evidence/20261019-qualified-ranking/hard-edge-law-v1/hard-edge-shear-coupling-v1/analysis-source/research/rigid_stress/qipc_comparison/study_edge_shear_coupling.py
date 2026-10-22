"""Check sharp-edge pressure sensitivity to proportional shear and normal displacement coupling."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import (
    halfspace_stress,
    rectangular_compliance,
    rectangular_shear_normal_compliance,
    solve_hard_contact,
)
from .hertz_tangential import cerruti_stress
from .study_franka_contact_gap import von_mises


def controls():
    widths = np.array([0.7, 0.3])
    source = np.array([[0.1, 0.15]])
    target = np.array([[-1.3, 1.5], [-1.2, -0.6], [1.1, 0.2]])
    modulus, poisson = 2.7, 0.3
    analytical = rectangular_shear_normal_compliance(target, source, widths, modulus, poisson)[:, 0]
    knots, weights = np.polynomial.legendre.leggauss(48)
    samples = np.stack(np.meshgrid(knots, knots, indexing="ij"), axis=-1).reshape(-1, 2) * widths / 2
    measure = np.outer(weights, weights).reshape(-1) / 4
    delta = target[:, None] - source - samples
    quadrature = np.sum(delta / np.sum(delta**2, axis=-1)[..., None] * measure[None, :, None], axis=1)
    quadrature *= (1 - 2 * poisson) / (2 * (1 - poisson) * np.pi * modulus)
    np.testing.assert_allclose(analytical, quadrature, rtol=2e-12, atol=1e-15)
    zero = rectangular_shear_normal_compliance(source, source, widths, modulus, poisson)
    np.testing.assert_allclose(zero, 0, rtol=0, atol=1e-15)
    scaled = rectangular_shear_normal_compliance(target * 3, source * 3, widths * 3, modulus * 5, poisson)
    np.testing.assert_allclose(scaled[:, 0], analytical / 15, rtol=3e-12, atol=1e-15)
    assert not rectangular_shear_normal_compliance(target, source, widths, modulus, 0.5).any()
    points = np.stack(np.meshgrid(np.arange(5) * 0.2, np.arange(7) * 0.17, indexing="ij"), axis=-1).reshape(-1, 2)
    widths = np.array([0.2, 0.17])
    normal = rectangular_compliance(points, points, widths, modulus)
    cross = rectangular_shear_normal_compliance(points, points, widths, modulus, poisson)
    np.testing.assert_allclose(cross, -cross.transpose(1, 0, 2), rtol=1e-11, atol=1e-14)
    matrix = normal + cross @ np.array([0.6, -0.4])
    np.testing.assert_allclose((matrix + matrix.T) / 2, normal, rtol=1e-11, atol=1e-14)
    assert np.linalg.eigvalsh(normal).min() > 0
    prescribed = np.zeros(len(points))
    prescribed[[2, 3, 9, 10, 11, 17, 18, 24, 25]] = np.linspace(0.07, 0.31, 9)
    dual = np.where(prescribed > 0, 0, np.linspace(0.01, 0.3, len(points)))
    gap = 2 + dual - matrix @ prescribed
    recovered = solve_hard_contact(matrix, gap, prescribed.sum(), is_symmetric=False)
    np.testing.assert_allclose(recovered.force, prescribed, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(recovered.gap, dual, rtol=1e-10, atol=1e-12)
    baseline = solve_hard_contact(normal, gap, prescribed.sum())
    general = solve_hard_contact(normal, gap, prescribed.sum(), is_symmetric=False)
    np.testing.assert_allclose(baseline.force, general.force, rtol=1e-10, atol=1e-12)
    return {
        "quadrature_relative_error": float(np.linalg.norm(analytical - quadrature) / np.linalg.norm(quadrature)),
        "manufactured_force_maximum_error": float(np.max(np.abs(recovered.force - prescribed))),
        "manufactured_iterations": recovered.iterations,
        "zero_self_integral_checked": True,
        "incompressible_limit_checked": True,
        "dimensional_scaling_checked": True,
        "skew_reciprocity_checked": True,
        "positive_symmetric_part_checked": True,
        "normal_solver_regression_checked": True,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path)
    cli.add_argument("--frame", type=Path)
    cli.add_argument("--boundaries", type=Path)
    cli.add_argument("--grids", nargs="+", type=int, default=[48])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [] if args.reference is None else [args.reference / "result.json", args.frame, args.boundaries]
    if args.reference is not None:
        inputs.extend(args.reference / f"grid{count}.npz" for count in args.grids)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(path) for path in inputs],
            "targets": {"pressure_cell_l2": 0.15, "sample_tensor_relative": 0.10, "sample_vm_peak_relative": 0.15},
            "scope": "Offline constant-rectangle half-space compliance at the saved actual P2 gap. Tangential traction is prescribed proportional to pressure using the observed resultant ratio, with a reversed-traction control. A nonsymmetric force-controlled complementarity solve includes Cerruti normal displacement. This is a declared proportional-traction sensitivity model, without stick/slip history or hollow-shell compliance. Original native refusal and all physical-pressure/peak qualifications remain open.",
            "surface_kernel_source": "https://doi.org/10.1680/jgere.22.00063 Eq.86 and Table4",
        },
    )
    checks = controls()
    write_json(args.output / "controls.json", checks)
    if args.reference is None:
        write_json(args.output / "result.json", {"passed": True, "controls": checks, "whole_goal_complete": False})
        return
    reference = json.loads((args.reference / "result.json").read_text())
    contact, load = reference["contact"], reference["normal_load_n"]
    with np.load(args.frame) as saved:
        rotation = Rotation.from_quat(saved["frame_quaternion"][0][[1, 2, 3, 0]])
        origin = saved["frame_position_m"][0]
        force = rotation.apply(saved["contact_force_n"][0, contact])
    with np.load(args.boundaries) as saved:
        normal, tangent = saved["normal"][0, contact], saved["tangent"][0, contact]
        planes = saved["planes"][0, contact]
    edge = (planes[:, 2] / np.linalg.norm(planes[:, :2], axis=1)).argmin()
    inward = -planes[edge, :2] / np.linalg.norm(planes[edge, :2])
    basis = np.column_stack((tangent, np.cross(normal, tangent))) @ np.column_stack((inward, [-inward[1], inward[0]]))
    ratio = force @ basis / load
    np.testing.assert_allclose(-force @ normal, load, rtol=1e-13)
    observed = np.array(reference["observed_contact_world_m"])
    modulus, poisson = 1e10 / 0.91, 0.3
    rows, previous = [], {}
    for count in args.grids:
        with np.load(args.reference / f"grid{count}.npz") as saved:
            points, gap, physical = saved["points_m"], saved["gap_m"], saved["material_points_m"]
            archived, observations = saved["reference_force_n"], saved["observations_m"]
        widths = np.array(next(row["cell_widths_m"] for row in reference["rows"] if row["grid"] == count))
        begun = perf_counter()
        normal_matrix = rectangular_compliance(points, points, widths, modulus)
        cross_matrix = rectangular_shear_normal_compliance(points, points, widths, modulus, poisson)
        assembly_seconds = perf_counter() - begun
        np.testing.assert_allclose(cross_matrix, -cross_matrix.transpose(1, 0, 2), rtol=1e-7, atol=1e-15)
        base = solve_hard_contact(normal_matrix, gap, load)
        np.testing.assert_array_equal(base.force, archived)
        positive = archived > 0
        base_normal_tensor = halfspace_stress(observations, points[positive], archived[positive])
        for direction in (1, -1):
            begun = perf_counter()
            matrix = normal_matrix + cross_matrix @ (direction * ratio)
            solution = solve_hard_contact(matrix, gap, load, is_symmetric=False)
            solve_seconds = perf_counter() - begun
            grid = solution.force.reshape(count, count)
            assert max(grid[-1].max(), grid[:, 0].max(), grid[:, -1].max()) < 1e-10 * load
            baseline_tensor = base_normal_tensor + cerruti_stress(
                observations, points[positive], archived[positive, None] * direction * ratio
            )
            active = solution.force > 0
            tensor = halfspace_stress(observations, points[active], solution.force[active]) + cerruti_stress(
                observations, points[active], solution.force[active, None] * direction * ratio
            )
            vm, baseline_vm = von_mises(tensor), von_mises(baseline_tensor)
            resultant = -load * normal + basis @ (direction * ratio * load)
            centroid = rotation.apply(physical.T @ solution.force / load) + origin
            baseline_centroid = rotation.apply(physical.T @ archived / load) + origin
            moment = np.cross(centroid - observed, resultant)
            pressure_change = np.linalg.norm(solution.force - archived) / np.linalg.norm(solution.force)
            tensor_change = np.linalg.norm(tensor - baseline_tensor) / np.linalg.norm(tensor)
            peak_change = abs(vm.max() - baseline_vm.max()) / vm.max()
            convergence = None
            if direction in previous:
                old_tensor, old_peak = previous[direction]
                convergence = {
                    "tensor_change": float(np.linalg.norm(tensor - old_tensor) / np.linalg.norm(tensor)),
                    "vm_peak_change": float(abs(vm.max() - old_peak) / vm.max()),
                }
            row = {
                "grid": count,
                "shear_direction": direction,
                "cell_widths_m": widths.tolist(),
                "assembly_seconds": assembly_seconds,
                "solve_seconds": solve_seconds,
                "iterations": solution.iterations,
                "active_cells": int(active.sum()),
                "indentation_m": solution.indentation,
                "centroid_edge_xy_m": (points.T @ solution.force / load).tolist(),
                "centroid_change_m": float(np.linalg.norm(centroid - baseline_centroid)),
                "moment_difference_over_Fa": float(np.linalg.norm(moment) / (load * reference["hertz_axes_m"][0])),
                "pressure_cell_l2_change": float(pressure_change),
                "sample_tensor_change": float(tensor_change),
                "sample_vm_peak_change": float(peak_change),
                "coupled_cell_mean_pressure_max_pa": float(solution.force.max() / widths.prod()),
                "coupled_sample_vm_peak_pa": float(vm.max()),
                "normal_only_approximation_targets_met": bool(
                    pressure_change < 0.15 and tensor_change < 0.10 and peak_change < 0.15
                ),
                "reference_refinement": convergence,
            }
            rows.append(row)
            previous[direction] = tensor, vm.max()
            np.savez_compressed(
                args.output / f"grid{count}-direction{direction}.npz",
                points_m=points,
                normal_only_force_n=archived,
                coupled_force_n=solution.force,
                complementarity_gap_m=solution.gap,
                observations_m=observations,
                coupled_tensor_pa=tensor,
                normal_only_tensor_pa=baseline_tensor,
            )
            write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "controls": checks,
            "contact": contact,
            "tangent_over_normal_components": ratio.tolist(),
            "rows": rows,
            "native_scene_admitted": False,
            "global_surface_peak_qualified": False,
            "friction_history_law_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
