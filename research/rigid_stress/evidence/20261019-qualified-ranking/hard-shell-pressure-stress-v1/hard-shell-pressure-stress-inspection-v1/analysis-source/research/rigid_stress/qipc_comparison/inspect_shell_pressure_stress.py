"""Independently reduce complete normal-shell tensors and conditional error gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json


def weighted_error(first, second, weights):
    return float(
        np.sqrt(
            np.dot(weights, np.sum((first - second) ** 2, axis=(-1, -2)))
            / np.dot(weights, np.sum(second**2, axis=(-1, -2)))
        )
    )


def extrema(field, points, common_count):
    eigen = np.linalg.eigvalsh(field)
    tensile = np.maximum(eigen[:, -1], 0)
    trace = np.trace(field, axis1=-2, axis2=-1)
    deviator = field - trace[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(1, 2)))
    result = {}
    for name, value in [("tensile", tensile), ("von_mises", vm)]:
        index = int(value.argmax())
        result[name + "_max_pa"] = float(value[index])
        result[name + "_location_m"] = points[index].tolist()
        result["common_" + name + "_max_pa"] = float(value[:common_count].max())
    return result, tensile, vm


def difference(first, second, volumes, areas):
    row = {
        "complete_tensor_relative": weighted_error(first["volume"], second["volume"], volumes),
        "membrane_relative": weighted_error(first["membrane"], second["membrane"], areas),
        "bending_relative": weighted_error(first["bending"], second["bending"], areas),
    }
    for name in ["tensile", "von_mises"]:
        row[name + "_peak_relative"] = abs(first[name + "_max_pa"] / second[name + "_max_pa"] - 1)
        row["common_" + name + "_peak_relative"] = abs(
            first["common_" + name + "_max_pa"] / second["common_" + name + "_max_pa"] - 1
        )
        row[name + "_peak_location_distance_m"] = float(
            np.linalg.norm(np.array(first[name + "_location_m"]) - second[name + "_location_m"])
        )
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--study", type=Path, required=True)
    cli.add_argument("--mesh-study", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert (args.study.parent / "returncode.txt").read_text().strip() == "0"
    report = json.loads((args.study / "result.json").read_text())
    assert report["execution_completed"]
    with np.load(args.study / "queries.npz") as data:
        unique, common = data["unique_points_m"], data["common_points_m"]
        volumes, areas = data["volume_weights_m3"], data["column_areas_m2"]
        maps = {key[4:]: data[key] for key in data.files if key.startswith("map_")}
    common_map = np.concatenate([maps[key] for key in ["volume", "columns", "probes", "ring"]])
    np.testing.assert_array_equal(unique[common_map], common)
    with np.load(args.mesh_study / "coarse-mesh.npz") as data:
        face = data["mid_faces"]
        outer, inner = data["outer_vertices"][face].mean(axis=1), data["inner_vertices"][face].mean(axis=1)
        mid = (data["outer_vertices"] + data["inner_vertices"])[face] / 2
    vector = np.cross(mid[:, 1] - mid[:, 0], mid[:, 2] - mid[:, 0])
    normal = vector / np.linalg.norm(vector, axis=1)[:, None]
    projection = np.eye(3) - normal[:, :, None] * normal[:, None, :]
    thickness = np.abs(np.einsum("ni,ni->n", outer - inner, normal))
    gauss, weights = np.polynomial.legendre.leggauss(8)
    depths, weights = (gauss + 1) / 2, weights / 2
    column_slice = slice(len(volumes), len(volumes) + len(maps["columns"]))
    rows, observations, bindings = [], {}, []
    primary_values = None
    for row in report["rows"]:
        name = row["mesh"]
        assert file_id(Path(row["fields"]["path"])) == row["fields"]
        points = np.concatenate((common, unique[maps["corners-" + name]]))
        summaries = {}
        with np.load(row["fields"]["path"]) as data:
            for label, claimed in row["variants"].items():
                field = data[label + "-common_tensor_pa"]
                corners = data[label + "-corner_tensor_pa"]
                all_values = np.concatenate((field, corners))
                summary, tension, vm = extrema(all_values, points, len(common))
                for metric in ["tensile", "von_mises"]:
                    for key in [metric + "_max_pa", "common_" + metric + "_max_pa"]:
                        np.testing.assert_allclose(summary[key], claimed[key], rtol=1e-10, atol=1e-6)
                    np.testing.assert_array_equal(summary[metric + "_location_m"], claimed[metric + "_location_m"])
                columns = field[column_slice].reshape(len(areas), 8, 3, 3)
                tangent = projection[:, None] @ columns @ projection[:, None]
                membrane = np.sum(tangent * weights[None, :, None, None], axis=1) * thickness[:, None, None]
                bending = (
                    np.sum(tangent * (weights * (0.5 - depths))[None, :, None, None], axis=1)
                    * thickness[:, None, None] ** 2
                )
                np.testing.assert_allclose(membrane, data[label + "-membrane_n_per_m"], rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(bending, data[label + "-bending_n"], rtol=1e-10, atol=1e-11)
                summary.update(volume=field[: len(volumes)].copy(), membrane=membrane, bending=bending)
                summaries[label] = summary
                rows.append(
                    {
                        "mesh": name,
                        "variant": label,
                        **{key: value for key, value in summary.items() if not isinstance(value, np.ndarray)},
                    }
                )
                if name == "fine" and label == "reference97":
                    primary_values = (points, tension, vm)
        observations[name] = summaries
        bindings.append(row["fields"])
    reference = observations["fine"]["reference97"]
    changes = {
        "body_medium_fine": difference(observations["medium"]["reference97"], reference, volumes, areas),
        "body_coarse_fine": difference(observations["coarse"]["reference97"], reference, volumes, areas),
        "pressure65_97": difference(observations["fine"]["reference65"], reference, volumes, areas),
        "source3_5": difference(observations["fine"]["reference97-sources3"], reference, volumes, areas),
        "integration_lower_higher": difference(
            observations["fine"]["reference97-lower-integration"], reference, volumes, areas
        ),
    }
    candidate = {
        name: {
            label: difference(values[label], reference, volumes, areas)
            for label in ["hertz-only", "hertz-point", "hertz-distributed"]
        }
        for name, values in observations.items()
    }
    for name, row in changes.items():
        for key, value in row.items():
            np.testing.assert_allclose(value, report["reference_changes"][name][key], rtol=1e-9, atol=1e-12)
    for mesh, variants in candidate.items():
        for label, row in variants.items():
            for key, value in row.items():
                np.testing.assert_allclose(value, report["candidate_errors"][mesh][label][key], rtol=1e-9, atol=1e-12)
    targets = report["targets"]
    reference_gate = all(
        max(row[key] for key in ["complete_tensor_relative", "membrane_relative", "bending_relative"])
        <= targets["reference_field_change"]
        and max(
            row[key]
            for key in [
                "tensile_peak_relative",
                "von_mises_peak_relative",
                "common_tensile_peak_relative",
                "common_von_mises_peak_relative",
            ]
        )
        <= targets["reference_peak_change"]
        for name, row in changes.items()
        if name != "body_coarse_fine"
    )
    numerical = report["numerical_checks"]["97"]
    reference_gate = (
        reference_gate
        and numerical["near256_512_maximum_over_pressure"] <= targets["near_integration_over_pressure"]
        and numerical["far9_17_maximum_over_pressure"] <= targets["far_integration_over_pressure"]
        and max(report["normal_source_table_versus_exact_prescribed_hertz_volume_changes"].values())
        <= targets["reference_field_change"]
    )
    assert bool(reference_gate) == report["conditional_single_case_reference_targets_met"]
    location = []
    points, tension, vm = primary_values
    for metric, values in [("tensile", tension), ("von_mises", vm)]:
        variation = sum(row[metric + "_peak_relative"] for name, row in changes.items() if name != "body_coarse_fine")
        near_maximum = values >= values.max() * (1 - min(variation, 0.5))
        tree = cKDTree(points[near_maximum])
        for mesh, variants in observations.items():
            for label in ["hertz-only", "hertz-point", "hertz-distributed"]:
                coordinate = variants[label][metric + "_location_m"]
                location.append(
                    {
                        "mesh": mesh,
                        "variant": label,
                        "metric": metric,
                        "raw_distance_m": candidate[mesh][label][metric + "_peak_location_distance_m"],
                        "distance_to_observed_near_maximum_set_m": float(tree.query(coordinate)[0]),
                        "relative_peak_variation_used": variation,
                        "reference_near_maximum_queries": int(near_maximum.sum()),
                    }
                )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "study_result": file_id(args.study / "result.json"),
            "queries": file_id(args.study / "queries.npz"),
            "field_bindings": bindings,
            "rows": rows,
            "reference_changes": changes,
            "candidate_errors": candidate,
            "peak_location_diagnostics": location,
            "conditional_single_case_reference_targets_met": bool(reference_gate),
            "conditional_candidate_targets_met": report["conditional_candidate_targets_met"],
            "scope": "Independent reductions of every retained compatible tensor and material-column resultant, raw/common sampled maxima and coordinates, reference variations and unchanged candidate budgets. Near-maximum location sets use observed reference peak variations and do not replace raw coordinate differences or certify continuum extrema. No native scene, arbitrary geometry or friction qualification.",
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
