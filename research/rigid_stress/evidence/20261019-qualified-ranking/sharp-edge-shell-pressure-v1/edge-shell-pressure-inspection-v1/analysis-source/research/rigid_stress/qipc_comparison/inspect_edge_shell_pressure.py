"""Reduce independent sharp-edge shell pressures and finite-depth fields."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_edge_fields import field_metrics
from .study_edge_shell_pressure import cell_tensor, source_average
from .study_shell_contact_pressure import points_on_grid


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--studies", type=Path, nargs=3, required=True, help="Coarse, medium and fine body meshes.")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    axes = [np.array([-2.0, 0, 2]), np.array([-3.0, 0, 3])]
    centers = np.array([[0.1, -0.2], [-0.4, 0.3]])
    averaging = source_average(axes, centers, np.array([1.2, 1.6]))
    table_points = points_on_grid(axes)
    np.testing.assert_allclose(
        averaging.T @ np.prod(table_points, axis=1), np.prod(centers, axis=1), rtol=1e-12, atol=1e-14
    )
    inputs, records, fields, candidates, pressures = [], [], [], [], []
    for study in args.studies:
        assert (study.parent / "returncode.txt").read_text().strip() == "0"
        start = json.loads((study / "start.json").read_text())
        result = json.loads((study / "result.json").read_text())
        assert result["execution_completed"]
        for receipt in start["inputs"]:
            assert file_id(receipt["path"]) == receipt
        if records:
            assert start["problem"] == records[0][0]["problem"]
            assert start["candidate"] == records[0][0]["candidate"]
            assert start["grids"] == records[0][0]["grids"]
        inputs.extend(file_id(study / name) for name in ["start.json", "result.json", "candidate.npz"])
        with np.load(study / "candidate.npz") as saved:
            candidate = {key: saved[key] for key in saved.files}
        arrays = {}
        for count in start["grids"]:
            path = study / f"grid{count}.npz"
            inputs.append(file_id(path))
            with np.load(path) as saved:
                arrays[count] = {key: saved[key] for key in saved.files}
            current = arrays[count]
            for row in [row for row in result["rows"] if row["grid"] == count]:
                table = row["source_table"]
                force = current[f"sources{table}-force_n"]
                np.testing.assert_allclose(force.sum(), start["problem"]["load_n"], rtol=1e-10, atol=0)
                assert force.min() >= -1e-12 * start["problem"]["load_n"]
                error = np.linalg.norm(current["trial_force_n"] - force) / np.linalg.norm(force)
                np.testing.assert_allclose(error, row["trial_cell_force_l2_relative"], rtol=1e-12, atol=1e-15)
                if table:
                    measured = field_metrics(candidate["combined_pa"], current[f"sources{table}-combined_pa"])
                    for key, value in measured.items():
                        np.testing.assert_allclose(value, row["combined_point_prediction"][key], rtol=1e-11, atol=1e-14)
        final = arrays[start["grids"][-1]]
        records.append((start, result))
        fields.append(final["sources5-combined_pa"])
        candidates.append(candidate)
        pressures.append(arrays)
    for candidate in candidates[1:]:
        np.testing.assert_array_equal(candidate["physical_queries_m"], candidates[0]["physical_queries_m"])
    start, result = records[-1]
    count = start["grids"][-1]
    last = pressures[-1][count]
    queries = candidates[-1]["local_queries_m"]
    # A third integration order independently checks the retained order-4 field.
    higher = cell_tensor(queries, last["points_m"], last["widths_m"], last["sources5-force_n"], 6)
    higher[:, 2, :2] *= -1
    higher[:, :2, 2] *= -1
    integration = field_metrics(higher + last["sources5-regular_pa"], fields[-1])
    body = field_metrics(fields[1], fields[2])
    coarse_body = field_metrics(fields[0], fields[2])
    model = start["candidate"]
    load = start["problem"]["load_n"]
    expected_centroid = np.array([0.4 * (1 - model["exponent"]) * model["width"], 0])
    expected_second = np.diag(
        [
            4 / 35 * (1 - model["exponent"]) * (2 - model["exponent"]) * model["width"] ** 2,
            model["along"] ** 2 / 5,
        ]
    )
    pressure_mesh = []
    for index, arrays in enumerate(pressures):
        force = arrays[count]["sources5-force_n"]
        fine = last["sources5-force_n"]
        pressure_mesh.append(float(np.linalg.norm(force - fine) / np.linalg.norm(fine)))
    fine_rows = [row for row in result["rows"] if row["source_table"] == 5]
    pressure_rows = []
    for row in fine_rows:
        pressure_rows.append(
            {
                "grid": row["grid"],
                "cell_widths_m": row["cell_widths_m"],
                "cell_pressure_max_pa": row["cell_pressure_max_pa"],
                "cell_pressure_l2_error": row["trial_cell_force_l2_relative"],
                "shell_pressure_change": row["pressure_change_from_halfspace"],
                "source3_source5_pressure_change": row["source3_source5_pressure_change"],
                "centroid_x_relative_error": float(abs(expected_centroid[0] / row["centroid_m"][0] - 1)),
                "second_moment_axis_relative_errors": (
                    np.abs(np.sqrt(np.diag(expected_second) / np.diag(row["second_moment_m2"])) - 1)
                ).tolist(),
                "combined_field_error": row["combined_point_prediction"],
                "grid_field_change": row["grid_stress_change"],
            }
        )
    candidate_error = field_metrics(candidates[0]["combined_pa"], fields[-1])
    source_change = field_metrics(last["sources3-combined_pa"], fields[-1])
    grid_change = fine_rows[-1]["grid_stress_change"]
    assert grid_change is not None
    keys = ["tensor_relative", "vm_peak_relative", "tensile_peak_relative"]
    envelope = {
        key: sum(item[key] for item in [candidate_error, body, source_change, grid_change, integration]) for key in keys
    }
    targets = start["targets"]
    reference_checks = {
        "source_pressure": fine_rows[-1]["source3_source5_pressure_change"] < targets["source_table_pressure_change"],
        "body_pressure": pressure_mesh[1] < targets["source_table_pressure_change"],
        "grid_tensor": grid_change["tensor_relative"] < targets["reference_tensor_change"],
        "grid_peaks": max(grid_change[key] for key in keys[1:]) < targets["reference_peak_change"],
        "body_tensor": body["tensor_relative"] < targets["reference_tensor_change"],
        "body_peaks": max(body[key] for key in keys[1:]) < targets["reference_peak_change"],
        "higher_integration": integration["tensor_relative"] < targets["integration_tensor_change"],
    }
    tensor = fields[-1]
    problem = start["problem"]
    strain = (
        (1 + problem["poisson"]) * tensor
        - problem["poisson"] * np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3)
    ) / problem["young_pa"]
    principal = np.linalg.eigvalsh(strain)
    tension = np.linalg.eigvalsh(tensor)[:, -1]
    dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": inputs,
            "problem": problem,
            "candidate": model,
            "force_n": load,
            "targets": targets,
            "pressure_rows": pressure_rows,
            "body_pressure_changes_to_fine": pressure_mesh,
            "medium_fine_field_change": body,
            "coarse_fine_field_change": coarse_body,
            "source3_source5_field_change": source_change,
            "order4_order6_field_change": integration,
            "coarse_candidate_fine_reference_error": candidate_error,
            "observed_variation_sum": envelope,
            "reference_variation_checks": reference_checks,
            "reference_variation_targets_met": all(reference_checks.values()),
            "split_cell_source_interpolation_control_passed": True,
            "sampled_error_budget_met": bool(
                envelope["tensor_relative"] < targets["sample_tensor"]
                and max(envelope[key] for key in keys[1:]) < targets["sample_peak"]
            ),
            "sampling_points": len(queries),
            "depth_min_m": float(queries[:, 2].min()),
            "depth_max_m": float(queries[:, 2].max()),
            "sampled_principal_strain_range": [float(principal.min()), float(principal.max())],
            "sampled_tensile_max_pa": float(tension.max()),
            "sampled_vm_max_pa": float(vm.max()),
            "tensile_location_m": candidates[-1]["physical_queries_m"][tension.argmax()].tolist(),
            "vm_location_m": candidates[-1]["physical_queries_m"][vm.argmax()].tolist(),
            "scope": "Cell-pressure L2 is finite-resolution only; its singular continuum square norm is not claimed. "
            "The observed variation sum is not a certified or statistical bound. Stress samples occupy a local "
            "positive-depth bank, not complete volume, membrane/bending or a global peak search. "
            "Linear strain reported only at these samples does not establish validity arbitrarily near the edge. "
            "This single frictionless fixed-orientation polar-land problem does not qualify moving native scenes.",
            "independent_complete_stress_qualified": False,
            "global_peak_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
