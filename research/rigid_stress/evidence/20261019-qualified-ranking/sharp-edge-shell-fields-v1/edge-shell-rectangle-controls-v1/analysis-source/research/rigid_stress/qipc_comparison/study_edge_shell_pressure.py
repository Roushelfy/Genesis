"""Independent normal sharp-edge contact on the retained hollow polar land.

Reuse finite-body normal-source responses, but solve new unilateral pressures.
This offline reference does not prescribe the candidate support or add a runtime
contact solve. Finite-depth samples are not complete-volume or surface maxima.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from skfem import Basis, ElementTetP2, MeshTet

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import edge_scales
from .edge_fields import composite_tensor, continuous_tensor
from .hertz_reference import halfspace_stress, rectangular_compliance, solve_hard_contact
from .study_edge_fields import field_metrics
from .study_far_edge_contact import edge_trial
from .study_shell_contact_pressure import points_on_grid
from .study_shell_pressure_stress import P2Readout


def source_average(axes, points, widths):
    """Exactly average the piecewise bilinear source interpolant over each cell.

    Split at table knots before Gaussian integration, including the knot at the
    edge. These uniform-pressure sources differ from the earlier pressure hats.
    """
    interpolation = RegularGridInterpolator(axes, np.eye(len(axes[0]) * len(axes[1])).reshape(*map(len, axes), -1))
    knots, weights = np.polynomial.legendre.leggauss(2)
    result = []
    for point in points:
        rules = []
        for axis, center, width in zip(axes, point, widths):
            lo, hi = center - width / 2, center + width / 2
            limits = np.r_[lo, axis[(axis > lo) & (axis < hi)], hi]
            x, w = [], []
            for left, right in zip(limits[:-1], limits[1:]):
                x.extend((left + right) / 2 + (right - left) * knots / 2)
                w.extend((right - left) * weights / (2 * width))
            rules.append((np.array(x), np.array(w)))
        samples = points_on_grid([rule[0] for rule in rules])
        measure = np.outer(rules[0][1], rules[1][1]).reshape(-1)
        result.append(measure @ interpolation(samples))
    result = np.array(result).T
    np.testing.assert_allclose(result.sum(axis=0), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.T @ points_on_grid(axes), points, rtol=1e-11, atol=1e-16)
    assert result.min() >= -1e-14
    return result


def cell_tensor(queries, points, widths, force, order):
    """Integrate cell pressure with independent displacement-gradient kernels."""
    selected = force > 0
    knots, weights = np.polynomial.legendre.leggauss(order)
    offsets = points_on_grid([knots * width / 2 for width in widths])
    measures = np.outer(weights, weights).reshape(-1) / 4
    sources = (points[selected, None] + offsets).reshape(-1, 2)
    loads = (force[selected, None] * measures).reshape(-1)
    return halfspace_stress(queries, sources, loads)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--grids", type=int, nargs="+", default=[24, 36, 48])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.parent / "start.json").read_text())
    mesh_path = Path(parent["mesh"]["path"])
    assert file_id(mesh_path) == parent["mesh"]
    load, young, poisson, slope, curvature = 0.005, 1e10, 0.3, 0.02, 80.0
    model = edge_scales(load, young / (1 - poisson**2), slope, curvature, 0)
    targets = {
        "pressure_cell_l2": 0.15,
        "sample_tensor": 0.10,
        "sample_peak": 0.15,
        "reference_tensor_change": 0.02,
        "reference_peak_change": 0.05,
        "source_table_pressure_change": 0.02,
        "integration_tensor_change": 0.005,
        "trial_integrated_force_relative": 1e-4,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.parent / name) for name in ["start.json", "result.json", "normal-source-table.npz"]]
            + [file_id(mesh_path)],
            "targets": targets,
            "problem": {
                "load_n": load,
                "young_pa": young,
                "poisson": poisson,
                "slope": slope,
                "along_curvature_per_m": curvature,
                "wall_m": 0.0005,
                "land_m": 0.0003,
                "gap": "slope*x + curvature*y*y/2, x >= 0; fixed orientation, normal force control",
                "reaction_moments": "Outputs; no imposed zero-moment or free-tilt condition",
                "kinematics": "Linear elastic, small-slope frictionless normal contact on a flat land",
            },
            "candidate": asdict(model),
            "grids": args.grids,
            "scope": "Independent new sharp-edge pressure/support with retained hollow-shell regular compliance. "
            "No deformable pad, fitted pressure/support or default coupled runtime solve. Source-table, cell-grid "
            "and body-mesh variations stay separate. Positive-depth sampled combined stress is not complete-volume "
            "or surface/global peak qualification; the ideal loaded edge remains singular.",
        },
    )
    with np.load(mesh_path) as mesh:
        vertices, tets = mesh["vertices"], mesh["tetrahedra"]
    basis = Basis(MeshTet(vertices.T, tets.T, sort_t=False), ElementTetP2(), intorder=4)
    with np.load(args.parent / "normal-source-table.npz") as table:
        np.testing.assert_array_equal(table["coordinates_m"], basis.doflocs.T)
        sources, displacements = table["source_points_m"], table["displacement_m_per_n"]
    source_axes = [np.unique(sources[:, axis]) for axis in (0, 1)]
    np.testing.assert_array_equal(points_on_grid(source_axes), sources[:, :2])
    queries = np.stack(
        np.meshgrid(
            model.width * np.array([-1, 0, 0.25, 0.75, 1.5]),
            model.along * np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5]),
            model.along * np.array([0.25, 0.5, 1, 2]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    physical = queries.copy()
    physical[:, 2] = 0.03 - queries[:, 2]
    readout = P2Readout(vertices, tets, physical)
    exact_candidate = continuous_tensor(queries, model, load, np.zeros(2), 48, 32)
    candidate_particular = composite_tensor(queries, model, load, np.zeros(2), 32)
    # Transform inward-positive local depth into the mesh's global +z axis.
    for tensor in (exact_candidate, candidate_particular):
        tensor[:, 2, :2] *= -1
        tensor[:, :2, 2] *= -1
    centroid = np.array([[0.4 * (1 - model.exponent) * model.width, 0]])
    point_weights = RegularGridInterpolator(source_axes, np.eye(25).reshape(5, 5, 25))(centroid)[0]
    point_displacement = np.einsum("nig,g->ni", displacements, point_weights * load)
    candidate_regular, _ = readout.field(point_displacement)
    candidate = candidate_particular + candidate_regular
    np.savez_compressed(
        args.output / "candidate.npz",
        local_queries_m=queries,
        physical_queries_m=physical,
        exact_pressure_particular_pa=exact_candidate,
        composite_particular_pa=candidate_particular,
        regular_pa=candidate_regular,
        combined_pa=candidate,
    )
    rows, prior = [], None
    for count in args.grids:
        widths = np.array([2 * model.width, 2.8 * model.along]) / count
        axes = [
            (np.arange(count) + 0.5) * widths[0],
            (np.arange(count) + 0.5 - count / 2) * widths[1],
        ]
        points = points_on_grid(axes)
        gap = slope * points[:, 0] + curvature * points[:, 1] ** 2 / 2
        base = rectangular_compliance(points, points, widths, young / (1 - poisson**2))
        np.testing.assert_allclose(base, base.T, rtol=1e-10, atol=1e-18)
        probes = np.column_stack((points, np.full(len(points), 0.03)))
        probe = sparse.vstack([basis.probes(probes[i : i + 128].T) for i in range(0, len(points), 128)], format="csr")
        response = -(probe @ displacements[:, 2, :])
        trial = (
            edge_trial(points, widths, model.width, model.along, 0, 64) * model.amplitude * model.width**model.exponent
        )
        force_error = abs(trial.sum() / load - 1)
        assert force_error < targets["trial_integrated_force_relative"], force_error
        saved = {"points_m": points, "widths_m": widths, "gap_m": gap, "trial_force_n": trial}
        table_forces = {}
        for table in [0, 3, 5]:
            selected = np.arange(25) if table == 5 else np.array([i * 5 + j for i in [0, 2, 4] for j in [0, 2, 4]])
            averaged = source_average(
                source_axes if table == 5 else [axis[::2] for axis in source_axes], points, widths
            )
            correction = response[:, selected] @ averaged if table else np.zeros_like(base)
            begun = perf_counter()
            solution = solve_hard_contact(base + correction, gap, load, is_symmetric=table == 0)
            seconds = perf_counter() - begun
            force = solution.force
            support = force.reshape(count, count) > 1e-10 * load
            assert not (support[-1].any() or support[:, 0].any() or support[:, -1].any())
            reference = cell_tensor(queries, points, widths, force, 4)
            lower = cell_tensor(queries, points, widths, force, 2)
            for tensor in (reference, lower):
                tensor[:, 2, :2] *= -1
                tensor[:, :2, 2] *= -1
            regular = np.zeros_like(reference)
            if table:
                displacement = np.einsum("nig,g->ni", displacements[:, :, selected], averaged @ force)
                regular, _ = readout.field(displacement)
            combined = reference + regular
            second = np.einsum("n,ni,nj->ij", force / load, points, points) + np.diag(widths**2 / 12)
            row = {
                "grid": count,
                "source_table": table,
                "solve_seconds": seconds,
                "iterations": solution.iterations,
                "cell_widths_m": widths.tolist(),
                "active_cells": int(support.sum()),
                "force_error_relative": float(abs(force.sum() / load - 1)),
                "minimum_gap_m": float(solution.gap.min()),
                "complementarity_nm": float(np.max(np.abs(force * solution.gap))),
                "indentation_m": float(solution.indentation),
                "cell_pressure_max_pa": float(force.max() / widths.prod()),
                "centroid_m": (force @ points / load).tolist(),
                "second_moment_m2": second.tolist(),
                "support_bounds_m": [
                    points[support.ravel()].min(axis=0).tolist(),
                    points[support.ravel()].max(axis=0).tolist(),
                ],
                "trial_force_quadrature_relative": float(force_error),
                "trial_cell_force_l2_relative": float(np.linalg.norm(trial - force) / np.linalg.norm(force)),
                "particular_prediction": field_metrics(exact_candidate, reference),
                "combined_point_prediction": field_metrics(candidate, combined) if table else None,
                "stress_integration_change": field_metrics(lower + regular, combined),
                "compliance_asymmetry_relative": float(
                    np.linalg.norm(correction - correction.T) / np.linalg.norm(base + correction)
                ),
            }
            if table:
                row["pressure_change_from_halfspace"] = float(
                    np.linalg.norm(force - table_forces[0]) / np.linalg.norm(force)
                )
            if table == 5:
                row["source3_source5_pressure_change"] = float(
                    np.linalg.norm(force - table_forces[3]) / np.linalg.norm(force)
                )
                row["grid_stress_change"] = field_metrics(prior, combined) if prior is not None else None
                prior = combined.copy()
            table_forces[table] = force
            rows.append(row)
            saved.update(
                {
                    f"sources{table}-force_n": force,
                    f"sources{table}-gap_m": solution.gap,
                    f"sources{table}-particular_pa": reference,
                    f"sources{table}-regular_pa": regular,
                    f"sources{table}-combined_pa": combined,
                }
            )
            write_json(args.output / "progress.json", {"rows": rows})
        np.savez_compressed(args.output / f"grid{count}.npz", **saved)
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "targets": targets,
            "sampling_depth_m": [float(queries[:, 2].min()), float(queries[:, 2].max())],
            "sampling_points": len(queries),
            "candidate_composite_error": field_metrics(candidate_particular, exact_candidate),
            "independent_complete_stress_qualified": False,
            "surface_global_peak_finite": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
