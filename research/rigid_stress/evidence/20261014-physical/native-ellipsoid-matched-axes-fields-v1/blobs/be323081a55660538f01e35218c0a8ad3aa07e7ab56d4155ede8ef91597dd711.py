"""Partition every covered-CAD probe's tensor change by distance to original contact samples."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .summarize_fine_references import closed_source


def partition(selected, weights, tensor_sq, reference_sq, hydro_sq, deviator_sq):
    numerator = float(tensor_sq[selected].sum())
    denominator = float(reference_sq[selected].sum())
    return {
        "probes": int(selected.sum()),
        "seed_diagnostic_weight_m3": float(weights[selected].sum()),
        "weighted_squared_tensor_change_Pa2_m3": numerator,
        "fraction_of_complete_squared_tensor_change": numerator / float(tensor_sq.sum()),
        "weighted_squared_hydrostatic_change_Pa2_m3": float(hydro_sq[selected].sum()),
        "weighted_squared_deviatoric_change_Pa2_m3": float(deviator_sq[selected].sum()),
        "weighted_squared_fine_tensor_Pa2_m3": denominator,
        "absolute_weighted_L2_change_Pa_m_3over2": float(np.sqrt(numerator)),
        "relative_to_same_region_fine_tensor": float(np.sqrt(numerator / denominator)) if denominator else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == value
        inputs[str(path)] = value

    for run in [args.comparison, args.source_audit]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        retain(run / "results/summary.json")
    comparison = json.loads((args.comparison / "results/summary.json").read_text())
    source = json.loads((args.source_audit / "results/summary.json").read_text())
    assert source["selector_mesh_reconstruction_qualified"]
    producers = list(dict.fromkeys(row["run"] for row in source["rows"]))
    assert producers == [row["run"] for row in comparison["rows"]]
    assert len(producers) == 3 and len(comparison["comparisons"]) == 2
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, row in enumerate(comparison["rows"]):
        path = args.comparison / f"results/case{index:02d}.npz"
        retain(path)
        with np.load(path) as fields:
            points, weights, tensor = [
                fields[key].copy() for key in ["common_points_m", "common_weights_m3", "objective_probe_tensor_Pa"]
            ]
        assert len(points) == comparison["common_physical_probes"] == 158340
        assert np.all(weights > 0) and np.isfinite(tensor).all()
        source_row = [item for item in source["rows"] if item["run"] == row["run"] and item["step"] == row["step"]]
        assert len(source_row) == 1 and source_row[0]["time_s"] == row["time_s"]
        path = args.source_audit / f"results/case{index:02d}-step{row['step']:04d}.npz"
        retain(path)
        with np.load(path) as fields:
            rest, groups, forces = [
                fields[key].copy() for key in ["rest_surface_positions_m", "groups", "normal_force_by_source_N"]
            ]
        samples = []
        for force in forces:
            # Keep every nonzero normal sample, including the parent component
            # threshold tail. This is a point-cloud diagnostic, not a patch edge.
            selected = (groups == 0) & np.any(force != 0, axis=1)
            assert selected.any()
            samples.append(rest[selected])
        peak = np.asarray(row["raw_peak_reference_position_m"])
        cases.append((points, weights, tensor, samples))
        row_record = {
            "run": row["run"],
            "tetrahedra": row["tetrahedra"],
            "step": row["step"],
            "time_s": row["time_s"],
            "nonzero_body_normal_sample_counts": [len(sample) for sample in samples],
            "raw_peak_reference_position_m": peak.tolist(),
            "raw_peak_lower_Pa": row["raw_peak_lower_Pa"],
            "raw_peak_upper_Pa": row["raw_peak_upper_Pa"],
            "raw_peak_nearest_own_normal_sample_distance_m": float(cKDTree(np.concatenate(samples)).query(peak)[0]),
        }
        (args.output / f"case{index:02d}.json").write_text(json.dumps(row_record, indent=2) + "\n")
    edges = np.array([0, 25, 50, 100, 250, 500, np.inf]) * 1e-6
    rows = []
    for index, (first, second, original) in enumerate(
        zip(cases[:-1], cases[1:], comparison["comparisons"], strict=True)
    ):
        points, weights, coarse, coarse_samples = first
        fine_points, fine_weights, fine, fine_samples = second
        np.testing.assert_array_equal(points, fine_points)
        np.testing.assert_array_equal(weights, fine_weights)
        difference = fine - coarse
        hydro = np.trace(difference, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
        deviator = difference - hydro
        tensor_sq = weights * np.sum(difference**2, axis=(-2, -1))
        hydro_sq = weights * np.sum(hydro**2, axis=(-2, -1))
        deviator_sq = weights * np.sum(deviator**2, axis=(-2, -1))
        reference_sq = weights * np.sum(fine**2, axis=(-2, -1))
        np.testing.assert_allclose(hydro_sq + deviator_sq, tensor_sq, rtol=1e-12, atol=1e-24)
        absolute = float(np.sqrt(tensor_sq.sum()))
        relative = float(np.sqrt(tensor_sq.sum() / reference_sq.sum()))
        np.testing.assert_allclose(absolute, original["tensor"]["absolute_l2_change"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(relative, original["tensor"]["relative_l2_change"], rtol=1e-12, atol=0)
        # Union of both actual support clouds avoids assigning the finer
        # contact distribution an unqualified status as physical truth.
        source_samples = [np.concatenate(pair) for pair in zip(coarse_samples, fine_samples, strict=True)]
        distance_by_source = np.column_stack([cKDTree(sample).query(points)[0] for sample in source_samples])
        nearest_source = np.argmin(distance_by_source, axis=1)
        distance = np.min(distance_by_source, axis=1)
        bin_id = np.searchsorted(edges[1:], distance, side="right")
        assert np.all(bin_id < len(edges) - 1)

        partition_data = (weights, tensor_sq, reference_sq, hydro_sq, deviator_sq)

        bins = []
        for item in range(len(edges) - 1):
            bins.append(
                {
                    "lower_distance_m_inclusive": float(edges[item]),
                    "upper_distance_m_exclusive": float(edges[item + 1]) if np.isfinite(edges[item + 1]) else None,
                    **partition(bin_id == item, *partition_data),
                }
            )
        assert sum(item["probes"] for item in bins) == len(points)
        np.testing.assert_allclose(
            sum(item["weighted_squared_tensor_change_Pa2_m3"] for item in bins), tensor_sq.sum(), rtol=1e-12, atol=0
        )
        top = np.argsort(tensor_sq, kind="stable")[-20:][::-1]
        rows.append(
            {
                "coarse_tetrahedra": original["coarse_tetrahedra"],
                "fine_tetrahedra": original["fine_tetrahedra"],
                "complete_absolute_weighted_L2_change_Pa_m_3over2": absolute,
                "complete_relative_weighted_L2_change": relative,
                "hydrostatic_squared_change_fraction": float(hydro_sq.sum() / tensor_sq.sum()),
                "deviatoric_squared_change_fraction": float(deviator_sq.sum() / tensor_sq.sum()),
                "distance_bins": bins,
                "nearest_captured_source_partitions": [
                    {"source": number + 1, **partition(nearest_source == number, *partition_data)}
                    for number in range(2)
                ],
                "largest_twenty_weighted_contributions": [
                    {
                        "probe": int(probe),
                        "reference_position_m": points[probe].tolist(),
                        "distance_to_union_normal_sample_cloud_m": float(distance[probe]),
                        "fraction_of_complete_squared_change": float(tensor_sq[probe] / tensor_sq.sum()),
                    }
                    for probe in top
                ],
            }
        )
        np.savez_compressed(
            args.output / f"pair{index:02d}.npz",
            common_points_m=points,
            common_weights_m3=weights,
            tensor_difference_Pa=difference,
            distance_to_source_union_normal_samples_m=distance_by_source,
            nearest_source_zero_based=nearest_source,
            distance_bin=bin_id,
            weighted_squared_tensor_change_Pa2_m3=tensor_sq,
            weighted_squared_hydrostatic_change_Pa2_m3=hydro_sq,
            weighted_squared_deviatoric_change_Pa2_m3=deviator_sq,
            weighted_squared_fine_tensor_Pa2_m3=reference_sq,
        )
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "comparison": str(args.comparison),
        "source_audit": str(args.source_audit),
        "complete_probe_count": comparison["common_physical_probes"],
        "rows": rows,
        "input_sha256": inputs,
        "all_original_probes_retained": True,
        "original_complete_weighted_tensor_changes_reproduced": True,
        "hydrostatic_deviatoric_and_spatial_partitions_recompose_complete_squared_change": True,
        "goal_completion": False,
        "limitations": "Five interior probes per seed tetrahedron retain their original seed-volume diagnostic weights; this is not exact cross-mesh continuum integration. Distance is measured in shared reference coordinates to the union of every nonzero body normal-force sample on the two compared meshes, not to a continuous contact boundary or a current finger edge. Source proximity does not assign mechanical causation. No probes, near-contact region or raw global maxima are removed. Every complete norm is reproduced before partitioning. Bulk, geometry and contact discretization remain coupled; localization alone proves neither singularity nor physical convergence, contact-model accuracy, feedback or warmed application cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
