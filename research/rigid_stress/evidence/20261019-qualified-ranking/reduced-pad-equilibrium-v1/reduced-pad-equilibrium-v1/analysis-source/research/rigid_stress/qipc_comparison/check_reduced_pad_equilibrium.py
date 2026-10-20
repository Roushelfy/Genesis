"""Compare eliminated-pad Newton solves with retained coupled physical fields."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm, peaks
from .compliant_plate_reference import pad_quadrature
from .contact_qp import solve_contact_qp
from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum
from .pad_equilibrium import normal_pad_equilibrium
from .reduced_pad_equilibrium import ReducedPadEquilibrium


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--qp-solver-path", type=Path, required=True)
    cli.add_argument("--paired-repeats", type=int, default=0)
    cli.add_argument("--changing-load-controls", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    study, original = (json.loads((args.study / name).read_text()) for name in ("result.json", "start.json"))
    assert study["passed"] and args.paired_repeats >= 0
    inputs = [file_id(args.study / name) for name in ("result.json", "start.json")]
    for key in ("mesh", "seed_operator"):
        assert file_id(Path(original[key]["path"])) == original[key]
        inputs.append(original[key])
    targets = {"complete_field_relative": 1e-6, "pressure_area_relative": 1e-6,
               "force_relative": 1e-6, "wrench": 1e-8, "pad_law": 1e-7,
               "elastic_equation": 1e-7, "omitted_gap_scaled": -1e-8}
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "inputs": inputs, "targets": targets, "paired_repeats": args.paired_repeats,
        "changing_load_controls": args.changing_load_controls,
        "scope": "Exact elimination of local pad springs, retaining full elastic and regional coupling. CPU prototype and same-discrete-model checks; no native throughput or continuum qualification."})
    before = perf_counter()
    with np.load(original["mesh"]["path"]) as saved:
        vertices, tets, outer, faces = (saved[k] for k in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
    center = vertices.mean(axis=0)
    model = FreeElasticP2(vertices, tets, 1e10, 0.3, 2000, column_faces=faces,
                         through_wall_ordering=True, cvxopt_factor=True)
    coordinates = model.vertices
    _, _, boundary = model.surface(faces)
    with np.load(original["seed_operator"]["path"]) as saved:
        seed_nodes, seed_matrix = saved["nodes"], saved["compliance_m_per_n"]
    factor_seconds = perf_counter() - before
    records, timings, changing = [], [], []
    for order in original["quadrature_orders"]:
        nodes, interpolation, points, area, candidate = pad_quadrature(model, len(outer), 0.003, order)
        np.testing.assert_array_equal(nodes, seed_nodes)
        top = coordinates[nodes, 2] > center[2]
        sign = np.where(top, -1.0, 1.0)
        gap = np.where(top, outer[:, 2].max() - coordinates[nodes, 2],
                       coordinates[nodes, 2] - outer[:, 2].min())
        rigid = np.column_stack((sign, sign * (coordinates[nodes, 1] - center[1]) / 0.03,
                                 -sign * (coordinates[nodes, 0] - center[0]) / 0.03, top))
        target = np.array([0.0, 0.0, 0.0, 1.0])
        selected = {tuple(sorted(face)) for face in candidate}
        omitted = np.array([tuple(sorted(face)) not in selected for face in boundary[:, :3]])
        solver = None
        previous_young = None
        setup_seconds = 0.0
        for case_index, case in enumerate(study["cases"]):
            young, stiffness, load = (case[k] for k in ("young_pa", "pad_stiffness_n_per_m3", "load_n"))
            matrix = seed_matrix * (1e10 / young)
            if young != previous_young:
                before = perf_counter()
                solver = ReducedPadEquilibrium(matrix, interpolation, rigid)
                setup_seconds = perf_counter() - before
                previous_young = young
            scale = load / (stiffness * np.pi * 0.002**2)
            stiffness_factor = 1 + case["stiffness_gradient"] * np.tanh(points[:, 0] / 0.001)
            spring = stiffness * stiffness_factor * area
            before = perf_counter()
            initial, _, _, _ = normal_pad_equilibrium(
                interpolation @ gap, interpolation @ rigid, area * stiffness_factor, stiffness, load)
            initial_seconds = perf_counter() - before
            solution = solver.solve(gap, spring, load, scale, initial)
            force = np.zeros_like(coordinates)
            force[nodes, 2] = sign * solution.nodal_force
            stress, elastic, correction, residual = model.solve(force)
            elastic *= 1e10 / young
            displacement = elastic.copy()
            displacement[:, 2] += solution.coordinates[0]
            displacement += np.cross(
                np.array([solution.coordinates[1] / 0.03, solution.coordinates[2] / 0.03, 0.0]),
                coordinates - center,
            )
            bottom_gap = coordinates[:, 2] + displacement[:, 2] - outer[:, 2].min()
            top_gap = outer[:, 2].max() + solution.coordinates[3] - coordinates[:, 2] - displacement[:, 2]
            actual_gap = interpolation @ np.where(top, top_gap[nodes], bottom_gap[nodes])
            law_error = np.linalg.norm(solution.point_force - spring * np.maximum(-actual_gap, 0))
            law_error /= np.linalg.norm(solution.point_force)
            outside = min(quadratic_triangle_minimum(bottom_gap[boundary[omitted]]).min(),
                          quadratic_triangle_minimum(top_gap[boundary[omitted]]).min())
            equation_error = np.linalg.norm(residual) / np.linalg.norm(force)
            correction_error = np.linalg.norm(correction) / np.linalg.norm(force)
            assert law_error <= targets["pad_law"] and equation_error <= targets["elastic_equation"]
            assert correction_error <= targets["wrench"]
            assert outside >= targets["omitted_gap_scaled"] * scale
            baseline = next(row for row in study["records"] if row["case"] == case_index
                            and row["mode"] == "coupled" and row["order"] == order
                            and row["search_contains_actual_elastic_compression"])
            assert baseline["search_radius_m"] == 0.003
            path = Path(baseline["field"]["path"])
            assert file_id(path) == baseline["field"]
            pressure = solution.point_force / area
            with np.load(path) as saved:
                np.testing.assert_array_equal(points, saved["points_m"])
                np.testing.assert_array_equal(area, saved["area_m2"])
                field_error = field_norm(stress - saved["stress_pa"], volumes) / field_norm(saved["stress_pa"], volumes)
                force_error = np.linalg.norm(force - saved["forces_n"]) / np.linalg.norm(saved["forces_n"])
                pressure_error = np.sqrt(np.sum(area * (pressure - saved["pressure_pa"])**2)
                                         / np.sum(area * saved["pressure_pa"]**2))
            assert field_error <= targets["complete_field_relative"]
            assert force_error <= targets["force_relative"] and pressure_error <= targets["pressure_area_relative"]
            path = args.output / f"case{case_index}-q{order}.npz"
            np.savez_compressed(path, points_m=points, area_m2=area, pressure_pa=pressure,
                point_force_n=solution.point_force, vertices_m=coordinates, tetrahedra=tets,
                stress_pa=stress, displacement_m=displacement, forces_n=force,
                actual_elastic_gap_m=actual_gap, rigid_coordinates_m=solution.coordinates)
            row = {"case": case_index, "order": order, "parameters": case,
                "complete_stress_relative_error": field_error.item(),
                "pressure_area_relative_error": pressure_error.item(),
                "nodal_force_relative_error": force_error.item(),
                "full_elastic_pad_law_relative_error": law_error.item(),
                "elastic_equation_relative_error": equation_error.item(),
                "balance_correction_relative": correction_error.item(),
                "minimum_omitted_boundary_gap_m": outside.item(),
                "wrench_relative_error": solution.wrench_error, "stationarity": solution.stationarity,
                "rounds": solution.rounds, "solve_seconds": solution.seconds,
                "rigid_initialization_seconds": initial_seconds,
                "operator_setup_seconds": setup_seconds,
                "operator_bytes": matrix.nbytes + solver.factor.nbytes + solver.point_factor.nbytes,
                "peaks": peaks(stress), "field": file_id(path), "baseline": baseline["field"]}
            records.append(row)
            write_json(args.output / "progress.json", {"records": records, "paired_timings": timings})
            if order == 6 and case_index in (0, 5, 7):
                for repeat in range(args.paired_repeats):
                    before = perf_counter()
                    original_solution = solve_contact_qp(
                        matrix * load / scale, gap / scale, rigid, target, np.empty((0, 6), dtype=int),
                        args.qp_solver_path, interpolation=interpolation, round_limit=1,
                        constraint_compliance=load / (scale * spring))
                    original_seconds = perf_counter() - before
                    assert original_solution.candidate_passed
                    before = perf_counter()
                    initial, _, _, _ = normal_pad_equilibrium(
                        interpolation @ gap, interpolation @ rigid, area * stiffness_factor, stiffness, load)
                    measured = solver.solve(gap, spring, load, scale, initial)
                    reduced_seconds = perf_counter() - before
                    error = np.linalg.norm(measured.point_force - original_solution.multipliers * load)
                    error /= np.linalg.norm(measured.point_force)
                    assert error <= targets["force_relative"]
                    timings.append({"case": case_index, "order": order, "repeat": repeat,
                                    "full_qp_seconds": original_seconds,
                                    "reduced_with_initialization_seconds": reduced_seconds,
                                    "point_force_relative_error": error.item()})
                    write_json(args.output / "progress.json", {"records": records, "paired_timings": timings})
        if order == 6 and args.changing_load_controls:
            # Prescribed tilt changes the upper physical gap. Recheck each new
            # state against the full QP; previous pressure is only an initial guess.
            stiffness, young = 1e10, 1e9
            matrix = seed_matrix * (1e10 / young)
            solver = ReducedPadEquilibrium(matrix, interpolation, rigid)
            spring = stiffness * area
            previous = None
            for step, (load, tilt) in enumerate(((0.1, 0.0), (0.2, 0.001), (0.05, -0.001), (0.1, 0.0))):
                scale = load / (stiffness * np.pi * 0.002**2)
                current_gap = gap + top * tilt * coordinates[nodes, 0]
                initial, _, _, _ = normal_pad_equilibrium(
                    interpolation @ current_gap, interpolation @ rigid, area, stiffness, load)
                cold = solver.solve(current_gap, spring, load, scale, initial)
                warm = solver.solve(current_gap, spring, load, scale,
                    initial if previous is None else previous.coordinates,
                    None if previous is None else previous.nodal_force)
                qp = solve_contact_qp(matrix * load / scale, current_gap / scale, rigid, target,
                    np.empty((0, 6), dtype=int), args.qp_solver_path, interpolation=interpolation,
                    round_limit=1, constraint_compliance=load / (scale * spring))
                assert qp.candidate_passed
                cold_error = np.linalg.norm(cold.point_force - qp.multipliers * load) / np.linalg.norm(cold.point_force)
                warm_error = np.linalg.norm(warm.point_force - cold.point_force) / np.linalg.norm(cold.point_force)
                assert max(cold_error, warm_error) <= targets["force_relative"]
                path = args.output / f"changing-{step}.npz"
                np.savez_compressed(path, points_m=points, area_m2=area, gap_m=current_gap,
                    cold_force_n=cold.point_force, warm_force_n=warm.point_force,
                    qp_force_n=qp.multipliers * load, coordinates_m=warm.coordinates)
                changing.append({"step": step, "load_n": load, "upper_tilt": tilt,
                    "qp_relative_force_error": cold_error.item(), "warm_relative_force_error": warm_error.item(),
                    "cold_rounds": cold.rounds, "warm_rounds": warm.rounds,
                    "cold_seconds": cold.seconds, "warm_seconds": warm.seconds, "arrays": file_id(path),
                    "scope": "Prescribed normal load/upper-gap tilt on the fixed candidate region. A temporal initialization control, not friction history or general moving-search admission."})
                previous = warm
                write_json(args.output / "progress.json", {"records": records,
                    "paired_timings": timings, "changing_load_controls": changing})
    assert len(records) == len(study["cases"]) * len(original["quadrature_orders"])
    write_json(args.output / "result.json", {"passed": True, "targets": targets, "records": records,
        "paired_timings": timings, "changing_load_controls": changing,
        "full_elastic_factor_seconds": factor_seconds,
        "production_model_admitted": False, "whole_goal_complete": False,
        "scope": "Sixteen coupled-field replays retain physical pad law, complete regional elastic coupling, force/moment balance and exact omitted-face search checks. CPU solve cost includes cold rigid initialization; fixed elastic/pad preprocessing is separate. Paired timings use the same process, hardware and physical cases, and are not independent-process repeats or native throughput. No new physical model, friction history, thickness validation or continuum-error rank is claimed."})


if __name__ == "__main__":
    main()
