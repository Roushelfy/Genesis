"""Remeasure fields, warm-start closure and costs of the three pad solvers."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm, peaks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--direct", type=Path, required=True)
    cli.add_argument("--fixed", type=Path, required=True)
    cli.add_argument("--adaptive", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parents, receipts = [], []
    for directory in (args.direct, args.fixed, args.adaptive):
        result = json.loads((directory / "result.json").read_text())
        assert result["passed"] and len(result["records"]) == 16
        assert len(result["changing_load_controls"]) == 4
        parents.append(result)
        receipts.extend(file_id(directory / name) for name in ("start.json", "result.json"))
        for row in result["records"]:
            assert file_id(Path(row["field"]["path"])) == row["field"]
            assert file_id(Path(row["baseline"]["path"])) == row["baseline"]
            assert row["rounds"][-1]["gradient_norm"] <= 1e-10
            assert row["full_elastic_pad_law_relative_error"] <= 1e-7
    assert parents[1]["linear_solver"] == "pcg-fixed" and parents[2]["linear_solver"] == "pcg-adaptive"
    assert parents[0]["targets"] == parents[1]["targets"] == parents[2]["targets"]
    rows, iterations, histories = [], [], []
    for mode, parent in zip(("direct", "pcg-fixed", "pcg-adaptive"), parents):
        for index, row in enumerate(parent["records"]):
            reference = parents[0]["records"][index]
            assert (row["case"], row["order"], row["parameters"]) == (
                reference["case"], reference["order"], reference["parameters"])
            assert row["baseline"] == reference["baseline"]
            with np.load(row["field"]["path"]) as current, np.load(reference["field"]["path"]) as first:
                for key in ("vertices_m", "tetrahedra", "points_m", "area_m2"):
                    np.testing.assert_array_equal(current[key], first[key])
                vertices, tets, area = (first[k] for k in ("vertices_m", "tetrahedra", "area_m2"))
                volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
                field_error = field_norm(current["stress_pa"] - first["stress_pa"], volumes)
                field_error /= field_norm(first["stress_pa"], volumes)
                pressure_error = np.sqrt(np.sum(area * (current["pressure_pa"] - first["pressure_pa"])**2)
                                         / np.sum(area * first["pressure_pa"]**2))
                force_error = np.linalg.norm(current["forces_n"] - first["forces_n"]) / np.linalg.norm(first["forces_n"])
                original_peak, current_peak = peaks(first["stress_pa"]), peaks(current["stress_pa"])
                peak_errors = {key: abs(current_peak[key] / original_peak[key] - 1)
                               for key in ("raw_vm_pa", "raw_tensile_pa")}
                assert max(field_error, pressure_error, force_error, *peak_errors.values()) <= 1e-6
            rows.append({"mode": mode, "case": row["case"], "order": row["order"],
                         "complete_stress_relative": field_error.item(),
                         "pressure_area_relative": pressure_error.item(),
                         "nodal_force_relative": force_error.item(), "peak_relative": peak_errors})
            iterations.append({"mode": mode, "case": row["case"], "order": row["order"],
                "newton_updates": len(row["rounds"]) - 1,
                "pcg_iterations": sum(r.get("linear_iterations", 0) for r in row["rounds"]),
                "final_gradient": row["rounds"][-1]["gradient_norm"],
                "solve_seconds": row["solve_seconds"]})
        states = []
        for step, row in enumerate(parent["changing_load_controls"]):
            assert row["step"] == step
            path = Path(row["arrays"]["path"])
            assert file_id(path) == row["arrays"]
            with np.load(path) as saved:
                cold, warm, qp = (saved[k] for k in ("cold_force_n", "warm_force_n", "qp_force_n"))
                assert np.isfinite(warm).all() and warm.min() >= 0
                cold_error = np.linalg.norm(cold - qp) / np.linalg.norm(qp)
                warm_error = np.linalg.norm(warm - qp) / np.linalg.norm(qp)
                assert max(cold_error, warm_error) <= 1e-6
                states.append(warm)
        closure_error = np.linalg.norm(states[-1] - states[0]) / np.linalg.norm(states[0])
        assert closure_error <= 1e-6
        histories.append({"mode": mode, "states": 4, "return_to_initial_load_pressure_relative": closure_error.item(),
            "scope": "Fixed candidate-region normal load/tilt change with full QP controls; no friction or general moving search."})
    timings = []
    assert len(parents[0]["paired_timings"]) == 9
    for case in (0, 5, 7):
        pair = [row for row in parents[0]["paired_timings"] if row["case"] == case]
        assert len(pair) == 3
        full = median(row["full_qp_seconds"] for row in pair)
        reduced = median(row["reduced_with_initialization_seconds"] for row in pair)
        timings.append({"case": case, "repeats": 3, "full_qp_median_seconds": full,
                        "reduced_with_initialization_median_seconds": reduced, "ratio": full / reduced})
    write_json(args.output / "result.json", {"passed": True, "identity": runtime_identity(require_gpu_inventory=False),
        "inputs": receipts, "field_comparisons": rows, "iterations": iterations,
        "changing_load_controls": histories, "paired_cpu_timings": timings,
        "fixed_outer_gradient_target": 1e-10, "fixed_pad_law_target": 1e-7,
        "fixed_force_moment_target": 1e-8,
        "adaptive_forcing": "min(0.1, sqrt(dimensionless_current_gradient_norm)); this is a tested heuristic, not the Eisenstat-Walker update.",
        "production_model_admitted": False, "whole_goal_complete": False,
        "scope": "Independent complete-field/pressure/peak comparisons on all16 matched physical cases per solver and four normal load/tilt states. Paired CPU timing compares full QP versus direct reduced Newton in one process including cold initialization; it excludes separately reported fixed operators and full FEM output. PCG iteration counts compare fixed1e-6 and explicitly labelled adaptive forcing with unchanged outer and physical checks. Cross-job PCG wall times are not a paired speed experiment. No new physical pressure law, continuum rank, native model or throughput is admitted."})


if __name__ == "__main__":
    main()
