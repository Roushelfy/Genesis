"""Check actual Quadrants outer-pressure transfer against the paired independent CPU solve."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import genesis as gs

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_recovery import NativePadRecovery
from .plate_native_pad import history_checks
from .plates import plate_xml


def compare(path, expected, target):
    with np.load(path) as saved:
        differences = {key: (np.linalg.norm(saved[key] - value) / np.linalg.norm(value)).item()
                       for key, value in expected.items()}
    assert max(differences.values()) <= target, differences
    return differences


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--cpu", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.run / "start.json").read_text())
    terminal = json.loads((args.run / "result.json").read_text())
    cpu = json.loads((args.cpu / "result.json").read_text())
    assert terminal["completed"] and terminal["mechanical_history_passed"] and cpu["passed"]
    assert all(terminal["history_checks"].values()) and all(w["mechanical_window_passed"] for w in terminal["windows"])
    params = SimpleNamespace(**start["args"])
    params.assets, params.output = Path(params.assets), args.output
    params.stress_output, params.primary_cost = "full", False
    assert params.method == "dpr" and params.source_reduction == "none"
    assert file_id(Path(params.mesh)) == start["mesh"]
    with np.load(params.mesh) as saved:
        outer, faces = saved["outer_vertices"], saved["mid_faces"]
    points = outer[faces].mean(axis=1)
    history = [json.loads(line) for line in (args.run / "steps.jsonl").read_text().splitlines()]
    targets = {"fixed_radius_replay": 1e-8, "resolved_cpu_fields": 1e-6, "equation": 1e-7,
               "zero_after_loading_absolute_stress_pa": 1e-3}
    write_json(args.output / "start.json", {"identity": runtime_identity(), "targets": targets,
        "original_start": file_id(args.run / "start.json"), "original_result": file_id(args.run / "result.json"),
        "original_journal": file_id(args.run / "steps.jsonl"), "cpu_result": file_id(args.cpu / "result.json"),
        "mesh": start["mesh"], "primary_timing": False})
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    xml = plate_xml(float(outer[:, 2].min() - params.clearance), float(outer[:, 2].max() + params.clearance))
    baseline = NativePadRecovery.allocate(params, outer, faces, points, xml)
    resolved = NativePadRecovery.allocate(params, outer, faces, points, xml, "resolved_outer_p2")
    max_params = SimpleNamespace(**vars(params))
    max_params.stress_output = "max"
    maximum = NativePadRecovery.allocate(max_params, outer, faces, points, xml, "resolved_outer_p2")
    rows = []
    for case in cpu["cases"]:
        level = case["level"]
        endpoint = next(row for row in reversed(history) if row["level"] == level)
        assert all(history_checks(endpoint).values()) and endpoint["stress"]["native_valid"]
        physical = args.run / f"frames/level{level}.npz"
        original = args.run / f"frames/stress-level{level}.npz"
        assert file_id(original) == case["original"] == endpoint["stress"]["field"]
        with np.load(physical) as saved:
            force, rotation, angular = saved["contact_force_n"], saved["rotation"], saved["angular_velocity_rad_per_s"]
        with np.load(original) as saved:
            load = saved["load_n"].item()
            expected = {key: saved[key] for key in ("stress_pa", "forces_n", "rhs_n")}
        old_path = args.output / f"fixed-radius-level{level}.npz"
        old = baseline.sample(force, rotation, angular, params.friction, load, old_path)
        assert old["native_valid"] and old["elastic_equation_relative_residual"] <= targets["equation"]
        original_difference = compare(old_path, expected, targets["fixed_radius_replay"])
        cpu_path = Path(case["variants"]["distributed_outer"]["field"]["path"])
        assert file_id(cpu_path) == case["variants"]["distributed_outer"]["field"]
        with np.load(cpu_path) as saved:
            # CPU node ordering is independent; align it to this native model.
            distance, permutation = cKDTree(saved["p2_coordinates_m"]).query(resolved.coordinates)
            assert distance.max() < 1e-14 and len(np.unique(permutation)) == len(resolved.coordinates)
            expected = {"stress_pa": saved["stress_pa"], "forces_n": saved["forces_n"][permutation],
                        "rhs_n": saved["rhs_n"][permutation]}
        new_path = args.output / f"resolved-outer-level{level}.npz"
        new = resolved.sample(force, rotation, angular, params.friction, load, new_path, physical_law_admitted=True)
        assert new["native_valid"] and new["load_transfer"] == "resolved_outer_p2"
        assert new["elastic_equation_relative_residual"] <= targets["equation"]
        cpu_difference = compare(new_path, expected, targets["resolved_cpu_fields"])
        max_path = args.output / f"resolved-max-level{level}.npz"
        max_row = maximum.sample(force, rotation, angular, params.friction, load, max_path, physical_law_admitted=True)
        peak_error = abs(max_row["raw_vm_peak_pa"] / case["variants"]["distributed_outer"]["raw_peaks"]["raw_vm_pa"] - 1)
        assert max_row["native_valid"] and peak_error <= targets["resolved_cpu_fields"]
        assert max_row["field_host_bytes"] == 0 and max_row["output_mode"] == "max"
        rows.append({"level": level, "physical_state": file_id(physical), "original": file_id(original),
                     "cpu_field": file_id(cpu_path), "baseline_replay": original_difference,
                     "resolved_cpu": cpu_difference, "fixed_radius": old, "resolved_outer": new,
                     "resolved_max": max_row, "resolved_max_cpu_peak_relative_error": peak_error})
        write_json(args.output / "progress.json", {"cases": rows})
    checks = {"fixed_radius_replay_preserved": True, "both_complete_resolved_cpu_fields_match": True}
    checks["both_native_max_outputs_match_cpu_peaks_without_full_host_fields"] = True
    try:
        resolved.sample(force, rotation, angular, params.friction, load)
    except ValueError as error:
        assert "physical law" in str(error)
        checks["missing_physical_admission_refused"] = True
    else:
        raise AssertionError("Missing physical source admission was accepted")
    zero = np.zeros_like(force)
    zero_path = args.output / "zero-after-loading.npz"
    empty = resolved.sample(zero, np.eye(3), np.zeros(3), params.friction, 0.0, zero_path, physical_law_admitted=True)
    assert empty["native_valid"] and empty["source_count"] == 0
    with np.load(zero_path) as saved:
        assert np.count_nonzero(saved["forces_n"]) == 0
        assert np.abs(saved["stress_pa"]).max() <= targets["zero_after_loading_absolute_stress_pa"]
    checks["empty_current_load_clears_loaded_stress"] = True
    empty_max = maximum.sample(zero, np.eye(3), np.zeros(3), params.friction, 0.0, physical_law_admitted=True)
    assert empty_max["native_valid"] and empty_max["raw_vm_peak_pa"] == 0
    checks["empty_current_load_clears_maximum_output"] = True
    damaged = zero.copy()
    damaged[0, 2] = np.nan
    invalid_path = args.output / "nonfinite-negative-control.npz"
    invalid = resolved.sample(damaged, np.eye(3), np.zeros(3), params.friction, 0.0, invalid_path,
                              physical_law_admitted=True)
    assert not invalid["native_valid"] and invalid["errno"] != 0
    checks["nonfinite_force_rejected_by_native_acceptance"] = True
    restored_path = args.output / "restored-after-rejection.npz"
    restored = resolved.sample(force, rotation, angular, params.friction, load, restored_path,
                               physical_law_admitted=True)
    assert restored["native_valid"]
    restored_difference = compare(restored_path, expected, targets["resolved_cpu_fields"])
    checks["current_valid_load_recovers_after_rejection"] = True
    write_json(args.output / "result.json", {"passed": all(checks.values()), "checks": checks, "cases": rows,
        "targets": targets, "zero": empty, "nonfinite_negative": invalid, "restored": restored_difference,
        "whole_goal_complete": False, "production_automatic_model_admitted": False,
        "scope": "Actual Quadrants work-conjugate transfer for known physical outer-face centroid forces, followed by unchanged native balance/PCG/acceptance/full-field kernels. Complete normal/friction fields match the independent CPU transfer; the manual fixed-radius replay remains a paired baseline. Empty current loading, missing physical-law admission and nonfinite source rejection are checked. The declared lagged pad law has its own complete mechanical source history; this is not instantaneous SCR cone admission, unknown collider-pressure prediction, a continuous new history or a production timing."})


if __name__ == "__main__":
    main()
