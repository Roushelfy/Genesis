"""Compare physical pad pressure from rigid, diagonal and coupled free-link compliance.

The static normal study reuses one elastic factor and cached contact operators.
Numerical search expands after an exact omitted-face gap check. All pressure
variants retain their own force balance; only the coupled model satisfies the
full elastic pad law. Source quadrature and load-model differences stay separate.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm, peaks
from .compliant_plate_reference import condense, pad_quadrature
from .contact_qp import solve_contact_qp
from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum
from .pad_elastic_fixed_point import solve_pad_fixed_point
from .pad_equilibrium import normal_pad_equilibrium
from .study_contact_regions import distribution


@dataclass
class ComplianceCache:
    nodes: np.ndarray
    matrix: np.ndarray
    seconds: float
    reciprocity: float


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--seed-operator", type=Path, required=True)
    cli.add_argument("--baseline-frame", type=Path, required=True)
    cli.add_argument("--qp-solver-path", type=Path, required=True)
    cli.add_argument("--cases", type=Path, required=True)
    cli.add_argument("--search-radii", type=float, nargs="+", default=[0.003, 0.0045, 0.006])
    cli.add_argument("--quadrature-orders", type=int, nargs="+", default=[0, 6])
    cli.add_argument("--independent-controls", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    assert args.search_radii == sorted(set(args.search_radii)) and min(args.search_radii) > 0
    assert args.quadrature_orders and min(args.quadrature_orders) >= 0
    assert cases and all(c["young_pa"] > 0 and c["pad_stiffness_n_per_m3"] > 0 and c["load_n"] > 0 for c in cases)
    started = perf_counter()
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "mesh": file_id(args.mesh), "seed_operator": file_id(args.seed_operator),
        "baseline_frame": file_id(args.baseline_frame), "cases": cases,
        "search_radii_m": args.search_radii, "quadrature_orders": args.quadrature_orders,
        "targets": {"wrench": 1e-8, "own_pad_law": 1e-7, "elastic_equation": 1e-7,
                    "omitted_gap_scaled": -1e-8, "baseline_field_relative": 1e-6},
        "scope": "Static linear elasticity and physical normal spring pads; no friction, continuum qualification or production timing."})
    with np.load(args.mesh) as saved:
        vertices, tets, outer, faces = (saved[k] for k in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
    assert volumes.min() > 0
    center = vertices.mean(axis=0)
    model = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000, column_faces=faces,
                         through_wall_ordering=True, cvxopt_factor=True)
    factor_seconds = perf_counter() - started
    coordinates = model.vertices
    _, _, boundary = model.surface(faces)
    with np.load(args.seed_operator) as saved:
        seed_nodes, seed_matrix = saved["nodes"], saved["compliance_m_per_n"]
    caches = {0.003: ComplianceCache(seed_nodes, seed_matrix, 0.0, 0.0)}
    records, comparisons = [], []
    independent = None
    baseline_passed = False
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for case_index, case in enumerate(cases):
            young, stiffness, load = (case[k] for k in ("young_pa", "pad_stiffness_n_per_m3", "load_n"))
            scale = load / (stiffness * np.pi * 0.002**2)
            for order in args.quadrature_orders:
                for radius in args.search_radii:
                    nodes, interpolation, points, area, candidate = pad_quadrature(model, len(outer), radius, order)
                    top = coordinates[nodes, 2] > center[2]
                    point_top = points[:, 2] > center[2]
                    sign = np.where(top, -1.0, 1.0)
                    if radius not in caches:
                        before = perf_counter()
                        matrix, reciprocity = condense(model, nodes, sign, journal)
                        caches[radius] = ComplianceCache(nodes, matrix, perf_counter() - before, reciprocity)
                        np.savez_compressed(args.output / f"compliance-{radius:g}.npz", nodes=nodes, compliance_m_per_n=matrix)
                    cache = caches[radius]
                    np.testing.assert_array_equal(cache.nodes, nodes)
                    gap = np.where(top, outer[:, 2].max() - coordinates[nodes, 2],
                                   coordinates[nodes, 2] - outer[:, 2].min())
                    rigid = np.column_stack((sign, sign * (coordinates[nodes, 1] - center[1]) / 0.03,
                                             -sign * (coordinates[nodes, 0] - center[0]) / 0.03, top))
                    target = np.array([0.0, 0.0, 0.0, 1.0])
                    stiffness_factor = 1 + case["stiffness_gradient"] * np.tanh(points[:, 0] / 0.001)
                    assert (stiffness_factor > 0).all()
                    spring = stiffness * stiffness_factor * area
                    selected = {tuple(sorted(face)) for face in candidate}
                    omitted = np.array([tuple(sorted(face)) not in selected for face in boundary[:, :3]])
                    fields, pressures, attempt = {}, {}, []
                    for mode in ("coupled", "rigid", "diagonal"):
                        before = perf_counter()
                        matrix = cache.matrix * (1e10 / young)
                        if mode == "diagonal":
                            matrix = np.diag(np.diag(matrix))
                        if mode == "rigid":
                            pose, point_force, wrench_error, iterations = normal_pad_equilibrium(
                                interpolation @ gap, interpolation @ rigid, area * stiffness_factor, stiffness, load)
                            alpha = interpolation.T @ point_force
                            predicted_gap = interpolation @ (gap + rigid @ pose)
                        else:
                            qp = solve_contact_qp(matrix * load / scale, gap / scale, rigid, target,
                                np.empty((0, 6), dtype=int), args.qp_solver_path, interpolation=interpolation,
                                round_limit=1, journal=journal, constraint_compliance=load / (scale * spring))
                            assert qp.candidate_passed, (case_index, order, mode, qp.status)
                            pose, point_force, alpha = qp.coordinates * scale, qp.multipliers * load, qp.nodal_force * load
                            predicted_gap = interpolation @ (gap + matrix @ alpha + rigid @ pose)
                            wrench_error, iterations = qp.force_error, qp.iterations
                        contact_seconds = perf_counter() - before
                        law_force = spring * np.maximum(-predicted_gap, 0)
                        own_law_error = np.linalg.norm(point_force - law_force) / np.linalg.norm(point_force)
                        assert own_law_error <= 1e-7 and wrench_error <= 1e-8
                        force = np.zeros_like(coordinates)
                        force[nodes, 2] = sign * alpha
                        stress, elastic, correction, residual = model.solve(force)
                        elastic *= 1e10 / young
                        displacement = elastic.copy()
                        displacement[:, 2] += pose[0]
                        displacement += np.cross(np.array([pose[1] / 0.03, pose[2] / 0.03, 0.0]), coordinates - center)
                        bottom_gap = coordinates[:, 2] + displacement[:, 2] - outer[:, 2].min()
                        top_gap = outer[:, 2].max() + pose[3] - coordinates[:, 2] - displacement[:, 2]
                        actual_gap = interpolation @ np.where(top, top_gap[nodes], bottom_gap[nodes])
                        actual_law_error = np.linalg.norm(point_force - spring * np.maximum(-actual_gap, 0)) / np.linalg.norm(point_force)
                        outside = min(quadratic_triangle_minimum(bottom_gap[boundary[omitted]]).min(),
                                      quadratic_triangle_minimum(top_gap[boundary[omitted]]).min())
                        contained = bool(outside >= -1e-8 * scale)
                        equation_error = np.linalg.norm(residual) / np.linalg.norm(force)
                        balance_error = np.linalg.norm(correction) / np.linalg.norm(force)
                        assert equation_error <= 1e-7 and balance_error <= 1e-8
                        if mode == "coupled":
                            assert actual_law_error <= 1e-7
                        pressure = law_force / area
                        footprints = []
                        for upper in (False, True):
                            mask = point_top == upper
                            stats = distribution(points[mask], law_force[mask], area[mask])
                            stats.update(upper=upper,
                                certain_area_m2=area[mask & (predicted_gap < -1e-8 * scale)].sum().item(),
                                possible_area_m2=area[mask & (predicted_gap < 1e-8 * scale)].sum().item())
                            footprints.append(stats)
                        path = args.output / f"case{case_index}-q{order}-r{radius:g}-{mode}.npz"
                        np.savez_compressed(path, points_m=points, area_m2=area, pressure_pa=pressure,
                            point_force_n=point_force, predicted_gap_m=predicted_gap, actual_elastic_gap_m=actual_gap,
                            vertices_m=coordinates, tetrahedra=tets, stress_pa=stress, displacement_m=displacement,
                            forces_n=force, equation_residual_n=residual, rigid_coordinates_m=pose)
                        row = {"case": case_index, "parameters": case, "order": order, "mode": mode,
                            "search_radius_m": radius, "search_contains_actual_elastic_compression": contained,
                            "minimum_omitted_boundary_gap_m": outside.item(), "own_pad_law_relative_error": own_law_error.item(),
                            "full_elastic_pad_law_relative_error": actual_law_error.item(),
                            "equation_relative_error": equation_error.item(), "balance_correction_relative": balance_error.item(),
                            "wrench_relative_error": float(wrench_error), "contact_iterations": iterations,
                            "contact_seconds": contact_seconds, "elastic_and_output_seconds": perf_counter() - before - contact_seconds,
                            "plate_travel_m": -pose[3].item(), "footprints": footprints, "peaks": peaks(stress),
                            "maximum_displacement_gradient": np.linalg.norm(model.displacement_gradient(displacement), axis=(-2, -1)).max().item(),
                            "field": file_id(path)}
                        records.append(row)
                        attempt.append(row)
                        fields[mode], pressures[mode] = stress, pressure
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        if case_index == 0 and order == 0 and mode == "coupled" and radius == 0.003:
                            assert young == 1e10 and stiffness == 1e9 and load == 0.1 and case["stiffness_gradient"] == 0
                            with np.load(args.baseline_frame) as saved:
                                difference = field_norm(stress - saved["stress_pa"], volumes) / field_norm(saved["stress_pa"], volumes)
                            assert difference <= 1e-6
                            baseline_passed = True
                            row["cached_baseline_complete_stress_relative_error"] = difference.item()
                            if args.independent_controls:
                                fixed = solve_pad_fixed_point(model, nodes, sign, interpolation, gap, rigid,
                                                              area, stiffness, load, journal)
                                assert fixed.candidate_passed
                                split = solve_contact_qp(matrix * load / scale, gap / scale, rigid, target,
                                    np.empty((0, 6), dtype=int), args.qp_solver_path,
                                    interpolation=np.repeat(interpolation, 2, axis=0), round_limit=1, journal=journal,
                                    constraint_compliance=load / (scale * np.repeat(spring / 2, 2)))
                                assert split.candidate_passed
                                fixed_error = np.linalg.norm(fixed.nodal_force * load - alpha) / np.linalg.norm(alpha)
                                split_error = np.linalg.norm(split.nodal_force * load - alpha) / np.linalg.norm(alpha)
                                assert max(fixed_error, split_error) <= 1e-7
                                independent = {"full_elastic_fixed_point_force_relative_error": fixed_error.item(),
                                    "split_source_nodal_force_relative_error": split_error.item(),
                                    "fixed_point_iterations": fixed.iterations,
                                    "same_nominal_baseline_stress_relative_error": difference.item()}
                    if all(row["search_contains_actual_elastic_compression"] for row in attempt):
                        for mode in ("rigid", "diagonal"):
                            comparisons.append({"case": case_index, "order": order, "mode": mode,
                                "search_radius_m": radius,
                                "pressure_area_relative_l2": np.sqrt(np.sum(area * (pressures[mode] - pressures["coupled"])**2)
                                    / np.sum(area * pressures["coupled"]**2)).item(),
                                "complete_stress_volume_relative_l2": (field_norm(fields[mode] - fields["coupled"], volumes)
                                    / field_norm(fields["coupled"], volumes)).item()})
                        break
                else:
                    raise RuntimeError("Declared numerical search limits still exclude physical compression")
                write_json(args.output / "progress.json", {"records": records, "comparisons": comparisons})
    assert baseline_passed and len(comparisons) == len(cases) * len(args.quadrature_orders) * 2
    assert not args.independent_controls or independent is not None
    write_json(args.output / "result.json", {"passed": True, "cases": cases, "records": records,
        "comparisons": comparisons, "cached_baseline_complete_field_passed": baseline_passed,
        "independent_controls": independent,
        "factor_seconds": factor_seconds, "total_seconds": perf_counter() - started,
        "operators": [{"radius_m": radius, "nodes": len(cache.nodes), "bytes": cache.matrix.nbytes,
                       "new_condensation_seconds": cache.seconds, "reciprocity_error": cache.reciprocity}
                      for radius, cache in caches.items()],
        "physical_reference_qualified": False, "production_model_admitted": False, "whole_goal_complete": False,
        "scope": "Actual static normal pressure/support and complete-field contrasts on one fixed coarse linear elastic mesh. Rigid and diagonal models solve their declared approximate pad law; their residual against full elasticity is reported. Coupled compliance retains all free-link modes and cross-region effects. Exact omitted-face minima certify numerical search containment; source quadrature0/6 remains separate from model error. No friction, thickness sweep, continuum-error or production-speed claim."})


if __name__ == "__main__":
    main()
