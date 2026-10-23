"""Audit full-domain stress errors separately from integration and peak cutoffs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_edge_shell_fields import compare, summarize


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--studies", type=Path, nargs=2, required=True, help="Lower and higher integration studies.")
    cli.add_argument("--samplings", type=Path, nargs=2, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"quadrature_norm_relative_change": 0.02, "quadrature_error_norm_relative_change": 0.05}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": targets,
            "scope": "Full-domain norm and error-integral refinement, distinct from pointwise "
            "reference changes. Finite-depth peak banks have explicit exclusion planes. No finite surface "
            "peak or general native scene qualification.",
        },
    )
    studies, samples, inputs = [], [], []
    for study, sampling in zip(args.studies, args.samplings):
        assert (study.parent / "returncode.txt").read_text().strip() == "0"
        result = json.loads((study / "result.json").read_text())
        start = json.loads((study / "start.json").read_text())
        assert result["execution_completed"]
        for receipt in start["inputs"]:
            assert file_id(receipt["path"]) == receipt
        assert result["sampling"] == file_id(sampling / "result.json")
        studies.append(result)
        samples.append(json.loads((sampling / "result.json").read_text()))
        inputs.extend(
            file_id(study / name) for name in ["start.json", "result.json", "coarse-fields.npz", "fine-fields.npz"]
        )
        inputs.append(file_id(sampling / "queries.npz"))
    low, high = studies
    keys = {
        "complete_tensor_relative": "volume_squared_norm_pa2_m3",
        "membrane_relative": "membrane_squared_norm_n2",
        "bending_relative": "bending_squared_norm_n2_m2",
    }
    integration = {}
    for metric, energy in keys.items():
        values = []
        for result in studies:
            row = next(row for row in result["rows"] if row["body"] == "fine")
            norm = row["variants"]["reference48"][energy] ** 0.5
            error = result["candidate_errors"]["coarse"][metric]
            values.append((norm, error * norm, error))
        integration[metric] = {
            "reference_norm_relative_change": abs(values[0][0] / values[1][0] - 1),
            "candidate_error_norm_relative_change": abs(values[0][1] / values[1][1] - 1),
            "candidate_error_fraction_change": abs(values[0][2] - values[1][2]),
            "lower_candidate_error": values[0][2],
            "higher_candidate_error": values[1][2],
        }
    with np.load(args.samplings[1] / "queries.npz") as saved:
        queries = {key: saved[key] for key in saved.files}
    queries["peak_points_m"] = queries["peak_points_m"][: int(np.prod(queries["peak_grid_shape"]))]
    points = np.concatenate([queries[key] for key in ["volume_points_m", "column_points_m", "peak_points_m"]])
    with np.load(args.studies[1] / "fine-fields.npz") as saved:
        reference = saved["reference48-tensor_pa"]
        reference_corners, reference_corner_points = saved["reference48-corner_tensor_pa"], saved["corner_points_m"]
    with np.load(args.studies[1] / "coarse-fields.npz") as saved:
        candidate = saved["candidate-point-tensor_pa"]
        candidate_corners, candidate_corner_points = saved["candidate-point-corner_tensor_pa"], saved["corner_points_m"]
    ref_summary = summarize(reference, queries, points, reference_corners, reference_corner_points)
    candidate_summary = summarize(candidate, queries, points, candidate_corners, candidate_corner_points)
    recomputed = compare(candidate_summary, ref_summary, queries)
    for metric in keys:
        np.testing.assert_allclose(
            recomputed[metric], high["candidate_errors"]["coarse"][metric], rtol=1e-12, atol=1e-14
        )
    for old, new in zip(high["candidate_errors"]["coarse"]["peaks"], recomputed["peaks"]):
        for key in old:
            np.testing.assert_allclose(old[key], new[key], rtol=1e-12, atol=1e-14)
    nv = len(queries["volume_points_m"])
    parent = json.loads((args.studies[1] / "start.json").read_text())
    along, width = parent["candidate"]["along"], parent["candidate"]["width"]
    local = queries["volume_points_m"] - [0, 0, 0.03]
    near = np.linalg.norm(local, axis=1) < 8 * along
    energy = queries["volume_weights_m3"] * np.sum(reference[:nv] ** 2, axis=(1, 2))
    contact_fraction = float(energy[near].sum() / energy.sum())
    integration_passed = all(
        row["reference_norm_relative_change"] <= targets["quadrature_norm_relative_change"]
        and row["candidate_error_norm_relative_change"] <= targets["quadrature_error_norm_relative_change"]
        for row in integration.values()
    )
    changes = {name: value for name, value in high["reference_changes"].items() if name != "coarse"}
    numerical_targets = high["targets"]
    field_gates = {
        metric: all(row[metric] <= numerical_targets["reference_field_change"] for row in changes.values())
        for metric in keys
    }
    envelopes = {metric: recomputed[metric] + sum(row[metric] for row in changes.values()) for metric in keys}
    peak_records = []
    low_ref = next(row for row in low["rows"] if row["body"] == "fine")["variants"]["reference48"]["peaks"]
    high_ref = ref_summary["peaks"]
    for index, (coarse, fine) in enumerate(zip(low_ref, high_ref)):
        row = {
            "cutoff_m": fine["cutoff_m"],
            "cutoff_over_width": fine["cutoff_m"] / width,
            "reference_values": fine,
            "candidate_errors": recomputed["peaks"][index],
            "checks": {},
            "envelope": {},
        }
        for bank in ["all", "dedicated"]:
            for name in ["tensile", "vm"]:
                key = f"{bank}_{name}"
                variation = abs(coarse[key + "_pa"] / fine[key + "_pa"] - 1)
                reference_changes = [value["peaks"][index][key + "_relative"] for value in changes.values()]
                row["checks"][key] = {
                    "integration_peak_relative_change": variation,
                    "maximum_reference_change": max(reference_changes),
                    "reference_targets_met": bool(
                        max([variation, *reference_changes]) <= numerical_targets["reference_peak_change"]
                    ),
                }
                row["envelope"][key] = (
                    recomputed["peaks"][index][key + "_relative"] + sum(reference_changes) + variation
                )
        peak_records.append(row)
    # Halve the common dedicated polar-plane lattice in both in-plane directions.
    peak_start = nv + len(queries["column_points_m"])
    shape = tuple(queries["peak_grid_shape"])
    size = int(np.prod(shape))
    grid = reference[peak_start : peak_start + size].reshape(*shape, 3, 3)
    peak_sampling = []
    for index, cutoff in enumerate(queries["peak_cutoffs_m"]):
        field, half = grid[:, :, index], grid[::2, ::2, index]
        row = {"cutoff_m": float(cutoff)}
        for label, value in [("full", field), ("half", half)]:
            eigen = np.linalg.eigvalsh(value)
            dev = value - np.trace(value, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            row[label + "_tensile_pa"] = float(eigen[..., -1].max())
            row[label + "_vm_pa"] = float(np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1))).max())
        peak_sampling.append(row)
    report = {
        "execution_completed": True,
        "inputs": inputs,
        "targets": targets,
        "integration": integration,
        "integration_targets_met": integration_passed,
        "reference_field_checks": field_gates,
        "reference_changes": high["reference_changes"],
        "candidate_errors": high["candidate_errors"],
        "field_observed_variation_sums": envelopes,
        "conditional_field_targets_met": {
            metric: bool(
                integration_passed and field_gates[metric] and envelope <= numerical_targets["candidate_field_error"]
            )
            for metric, envelope in envelopes.items()
        },
        "peak_cutoffs": peak_records,
        "dedicated_plane_sampling": peak_sampling,
        "reference_volume_energy_fraction_within_8_half_lengths": contact_fraction,
        "sample_counts": [
            {
                key: sample[key]
                for key in [
                    "volume_points",
                    "column_points",
                    "material_columns",
                    "peak_points",
                    "minimum_volume_depth_m",
                    "minimum_column_depth_m",
                ]
            }
            for sample in samples
        ],
        "scope": "Observed sums use same-query body, pressure, source and far-integration differences. "
        "The separate quadrature check compares integrated norms and error norms on two exact partitions, "
        "not a pointwise field difference or a certified bound. Surface/global stress remains unbounded. "
        "No native configuration, nonlinear material or general friction/moving-contact qualification.",
        "finite_surface_global_peak": False,
        "native_scene_accuracy_qualified": False,
        "whole_goal_complete": False,
    }
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
