"""Inspect complete retained pressure arrays and separate reference uncertainties."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, ellipse_pressure


def relative(first, second):
    return float(np.linalg.norm(first - second) / np.linalg.norm(second))


def read(directory):
    assert (directory.parent / "returncode.txt").read_text().strip() == "0"
    result = json.loads((directory / "result.json").read_text())
    start = json.loads((directory / "start.json").read_text())
    assert result["execution_completed"]
    return result, start


def inspect_arrays(directory, count, axes, query, major_axis):
    widths = np.array([axis[1] - axis[0] for axis in axes])
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    rigid = np.column_stack((np.ones(len(points)), points / major_axis))
    fields, metrics = {}, {}
    with np.load(directory / "pressures.npz") as data:
        np.testing.assert_array_equal(data["common_points_m"], query)
        for table in [0, 3, 5]:
            label = f"grid{count}-sources{table}"
            force = data[label + "-force_n"].reshape(-1)
            nodal = data[label + "-pressure_pa"]
            gap = data[label + "-scaled_gap"]
            np.testing.assert_allclose(nodal.reshape(-1) * widths.prod(), force, rtol=2e-15, atol=1e-19)
            normalized_force = force / 0.2
            assert normalized_force.min() >= -1e-12
            assert np.linalg.norm(rigid.T @ normalized_force - [1, 0, 0]) <= 1e-10
            assert gap.min() >= -1e-10 and np.max(np.abs(normalized_force * gap)) <= 1e-10
            observation = RegularGridInterpolator(axes, nodal)(query)
            fields[table] = observation
            second = np.einsum("n,ni,nj->ij", normalized_force, points, points) + np.diag(widths**2 / 6)
            moment_axes = np.sqrt(5 * np.linalg.eigvalsh(second)[::-1])
            positive = normalized_force.reshape(count, count) > 1e-10
            assert not (positive[0].any() or positive[-1].any() or positive[:, 0].any() or positive[:, -1].any())
            # The continuous pressure is bilinear on each rectangle. Its exact
            # integral is the corner average times area, independent of nodal sums.
            cell_integral = (nodal[:-1, :-1] + nodal[1:, :-1] + nodal[:-1, 1:] + nodal[1:, 1:]) / 4
            force_by_cells = cell_integral.sum() * widths.prod()
            assert abs(force_by_cells - 0.2) <= 0.2e-10
            support_min, support_max = [], []
            for width, coordinate in zip(widths, points.T):
                support_min.append(float(coordinate[positive.reshape(-1)].min() - width))
                support_max.append(float(coordinate[positive.reshape(-1)].max() + width))
            metrics[table] = {
                "force_by_cell_integral_n": float(force_by_cells),
                "force_by_coefficients_n": float(force.sum()),
                "maximum_nodal_pressure_pa": float(nodal.max()),
                "moment_equivalent_axes_m": moment_axes.tolist(),
                "discrete_hat_support_minimum_xy_m": support_min,
                "discrete_hat_support_maximum_xy_m": support_max,
                "minimum_scaled_gap": float(gap.min()),
                "maximum_complementarity_product": float(np.max(np.abs(normalized_force * gap))),
            }
    return fields, metrics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parents", type=Path, nargs=3, required=True, metavar=("COARSE", "MEDIUM", "FINE"))
    cli.add_argument("--refinements", type=Path, nargs=3, required=True, metavar=("COARSE", "MEDIUM", "FINE"))
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    axes, direction, _, peak = ellipse(0.2, 1e10 / (1 - 0.3**2), np.diag([50.0, 80.0]))
    query_axes = [np.linspace(-1.3 * width, 1.3 * width, 129) for width in axes]
    query = np.stack(np.meshgrid(*query_axes, indexing="ij"), axis=-1).reshape(-1, 2)
    candidate = ellipse_pressure(query, axes, direction, peak)
    rows, references, all_bindings = [], [], []
    control, reference_targets = [], None
    for level, parent, refined in zip(["coarse", "medium", "fine"], args.parents, args.refinements):
        result, start = read(parent)
        refined_result, refined_start = read(refined)
        assert refined_start["parent_result"] == file_id(parent / "result.json")
        assert refined_start["source_table"] == file_id(parent / "normal-source-table.npz")
        assert refined_start["previous_pressures"] == file_id(parent / "pressures.npz")
        assert start["mesh"] == file_id(Path(start["mesh"]["path"]))
        assert len(result["normal_source_checks"]) == 25
        assert max(r["force_error_n"] for r in result["normal_source_checks"]) < 1e-5
        assert max(r["moment_error_over_wall_nm"] for r in result["normal_source_checks"]) < 1e-5
        assert max(r["elastic_residual_relative"] for r in result["normal_source_checks"]) < 1e-7
        assert result["targets"] == refined_result["targets"]
        if reference_targets is None:
            reference_targets = result["targets"]
        assert reference_targets == result["targets"]
        fields65, _ = inspect_arrays(parent, 65, [np.linspace(-1.3 * w, 1.3 * w, 65) for w in axes], query, axes[0])
        fields97, metrics = inspect_arrays(
            refined, 97, [np.linspace(-1.3 * w, 1.3 * w, 97) for w in axes], query, axes[0]
        )
        reference = fields97[5]
        references.append(reference)
        control.append(fields97[0])
        corrected_curvature = np.diag([50.0, 80.0]) + np.array(result["one_pass_curvature_correction_per_m"])
        corrected_axes, corrected_direction, _, corrected_peak = ellipse(0.2, 1e10 / (1 - 0.3**2), corrected_curvature)
        corrected = ellipse_pressure(query, corrected_axes, corrected_direction, corrected_peak)
        axes_reference = np.array(metrics[5]["moment_equivalent_axes_m"])
        row = {
            "mesh": level,
            "grid65_grid97_pressure_relative_change": relative(fields65[5], reference),
            "source3_source5_pressure_relative_change": relative(fields97[3], reference),
            "hertz_pressure_relative_error_to_discrete_reference": relative(candidate, reference),
            "halfspace_hertz_pressure_relative_error": relative(candidate, fields97[0]),
            "hertz_peak_pressure_relative_error": float(peak / metrics[5]["maximum_nodal_pressure_pa"] - 1),
            "hertz_moment_axes_relative_error": (axes / axes_reference - 1).tolist(),
            "curvature_corrected_pressure_relative_error": relative(corrected, reference),
            "kernel_rule_relative_change": result["rows"][0]["kernel_rule_relative_change"],
            "arrays": metrics,
        }
        rows.append(row)
        all_bindings.append(
            {
                "mesh": level,
                "parent_result": file_id(parent / "result.json"),
                "refinement_result": file_id(refined / "result.json"),
                "parent_pressures": file_id(parent / "pressures.npz"),
                "refined_pressures": file_id(refined / "pressures.npz"),
                "mesh_file": start["mesh"],
            }
        )
    mesh_changes = {
        "coarse_fine": relative(references[0], references[2]),
        "medium_fine": relative(references[1], references[2]),
    }
    halfspace_mesh_differences = [relative(field, control[0]) for field in control[1:]]
    assert max(halfspace_mesh_differences) < 1e-10
    pressure_variation_targets_met = (
        all(
            row["grid65_grid97_pressure_relative_change"] <= reference_targets["pressure_grid_change"]
            and row["source3_source5_pressure_relative_change"] <= reference_targets["source_table_change"]
            and row["kernel_rule_relative_change"] <= reference_targets["halfspace_kernel_rule_change"]
            for row in rows
        )
        and max(mesh_changes.values()) <= reference_targets["pressure_grid_change"]
    )
    candidate_targets_met = all(
        row["hertz_pressure_relative_error_to_discrete_reference"]
        <= reference_targets["pressure_prediction_relative_l2"]
        and max(abs(v) for v in row["hertz_moment_axes_relative_error"]) <= reference_targets["moment_axes_relative"]
        for row in rows
    )
    envelope = (
        rows[-1]["hertz_pressure_relative_error_to_discrete_reference"]
        + rows[-1]["grid65_grid97_pressure_relative_change"]
        + rows[-1]["source3_source5_pressure_relative_change"]
        + mesh_changes["medium_fine"]
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "rows": rows,
            "bindings": all_bindings,
            "targets": reference_targets,
            "pressure_mesh_relative_changes": mesh_changes,
            "halfspace_mesh_relative_differences": halfspace_mesh_differences,
            "tested_reference_pressure_variations_within_targets": bool(pressure_variation_targets_met),
            "hertz_pressure_and_moment_axes_within_targets": bool(candidate_targets_met),
            "observed_conditional_pressure_error_envelope": float(envelope),
            "curvature_correction_selected": False,
            "independent_complete_stress_qualified": False,
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
            "scope": "One force-controlled, freely tilting rigid paraboloid above a flat polar land on a fixed hollow shell with inertial relief. Pressure and discrete support are outputs. Moment axes differ from support edges. Sampled complementarity and observed mesh/grid/source variations do not certify continuum pressure, complete/peak stress, friction or general moving geometry. The summed observed variation is a conditional engineering envelope, not a statistical or certified continuum bound.",
        },
    )


if __name__ == "__main__":
    main()
