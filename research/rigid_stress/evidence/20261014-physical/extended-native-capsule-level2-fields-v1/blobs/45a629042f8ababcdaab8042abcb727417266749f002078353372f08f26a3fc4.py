"""Replay the same qualified576 material probes across every fixed-mesh contact-distance state."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_objective_frame import finite_frame
from .compare_covered_cad_fields import point_fields
from .compare_spatial_franka import qualify, weighted_change
from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--qualified-depth-probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest

    for run in [args.comparison, args.qualified_depth_probes]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        retain(run / "results/summary.json")
    comparison = json.loads((args.comparison / "results/summary.json").read_text())
    qualified = json.loads((args.qualified_depth_probes / "results/summary.json").read_text())
    assert comparison["all_geometry_and_contact_map_arrays_exactly_equal"]
    assert comparison["every_original_five_state_force_audit_and_raw_bounds_matched"]
    assert len(comparison["rows"]) == 15
    assert qualified["all_shared_points_strictly_inside_every_mesh_and_evaluator_qualified"]
    assert qualified["supplemental_probes"] == 576
    for module in ["compare_covered_cad_fields", "compare_spatial_franka", "audit_objective_frame", "skfem_reference"]:
        relative = Path(f"research/rigid_stress/{module}.py")
        original_source = args.qualified_depth_probes / "source" / relative
        assert hashlib.sha256(original_source.read_bytes()).digest() == hashlib.sha256(relative.read_bytes()).digest()
        retain(original_source)
    path = args.qualified_depth_probes / "results/case00.npz"
    retain(path)
    with np.load(path) as saved:
        points, depths, source_ids, original_tensor = [
            saved[key].copy()
            for key in ["shared_points_m", "facet_normal_depth_m", "source_ids", "objective_tensor_Pa"]
        ]
    baseline = Path(comparison["rows"][0]["run"])
    static, config = restore_static(baseline)
    assert static.body.mesh.nelements == 31668 and len(points) == 576
    qualification = qualify(static.body, points)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, changes, previous = [], [], {}
    producers = list(dict.fromkeys(row["run"] for row in comparison["rows"]))
    assert len(producers) == 3
    for index, run_string in enumerate(producers):
        run = Path(run_string)
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        retain(run / "results/configuration.json")
        for filename in ["reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
            path = run / "results" / filename
            retain(path)
            with np.load(path) as actual, np.load(baseline / "results" / filename) as original:
                assert set(actual.files) == set(original.files)
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], original[key])
        selected_rows = [row for row in comparison["rows"] if row["run"] == run_string]
        assert [row["step"] for row in selected_rows] == list(range(1, 6))
        for row in selected_rows:
            path = run / f"results/step{row['step']:04d}.npz"
            retain(path)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["body_component_permutation"], static.parts[0].permutation)
                vector = np.empty(static.body.basis.N, dtype=np.longdouble)
                vector[static.parts[0].permutation] = saved["displacement"][: static.body.basis.N]
            rotation, _, objective = finite_frame(static.body, np.asarray(vector, dtype=float))
            tensor, vm, _ = point_fields(static.body, vector, points)
            tensor = rotation.T @ tensor @ rotation
            _, _, displacement = point_fields(static.body, objective, points)
            assert row["time_s"] == row["step"] * config["dt"]
            assert float(vm.max()) <= row["raw_peak_upper_Pa"] * (1 + 1e-10) + 1e-5
            if index == 0 and row["step"] == 5:
                np.testing.assert_allclose(tensor, original_tensor, rtol=1e-12, atol=1e-6)
            old = previous.get(row["step"])
            if old is not None:
                for source in [0, 1, 2]:
                    for depth in np.unique(depths):
                        selected = (depths == depth) & ((source_ids == source) if source else True)
                        changes.append(
                            {
                                "step": row["step"],
                                "time_s": row["time_s"],
                                "first_barrier_distance_m": old["distance"],
                                "second_barrier_distance_m": row["barrier_distance_m"],
                                "source_zero_means_both": source,
                                "facet_normal_depth_m": float(depth),
                                "probes": int(selected.sum()),
                                "tensor_equal_point_weight_diagnostic": weighted_change(
                                    old["tensor"][selected][None], tensor[selected][None]
                                ),
                            }
                        )
            previous[row["step"]] = {"distance": row["barrier_distance_m"], "tensor": tensor}
            result = {
                "run": run_string,
                "step": row["step"],
                "time_s": row["time_s"],
                "barrier_distance_m": row["barrier_distance_m"],
                "supplemental_sample_vm_maximum_Pa": float(vm.max()),
                "original_raw_global_lower_Pa": row["raw_peak_lower_Pa"],
                "original_raw_global_upper_Pa": row["raw_peak_upper_Pa"],
            }
            rows.append(result)
            np.savez_compressed(
                args.output / f"case{index:02d}-step{row['step']:04d}.npz",
                shared_points_m=points,
                facet_normal_depth_m=depths,
                source_ids=source_ids,
                objective_tensor_Pa=tensor,
                vm_Pa=vm,
                objective_displacement_m=displacement,
            )
            print(json.dumps(result), flush=True)
    assert len(rows) == 15 and len(changes) == 180
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "rows": rows,
        "adjacent_depth_changes": changes,
        "point_evaluator_qualification": qualification,
        "qualified_baseline_final_probe_tensor_reproduced": True,
        "exact_original576_probe_coordinates_and_depths_retained": True,
        "all_geometry_and_contact_maps_rechecked_identical": True,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Every actual dynamic state uses the same576 qualified material-coordinate points, original1–50um baseline-facet depths and equal-point diagnostic norms. The closed parent separately compares complete volume quadrature, all corner tensors, source forces and raw bounds. These supplemental point comparisons neither replace those fields nor define a physical volume norm or continuous pressure metric. No probe is relocated, filtered or retuned per barrier distance. Fixed coarse spatial discretization and short windows do not establish the hard-contact limit, spatial convergence, initializer field coverage, full application feedback or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
