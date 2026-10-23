"""Reinspect exact-column and local-window integral convergence from saved arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .study_hertz_residual import field_difference


def changes(first_norm, second_norm, first_error, second_error):
    norm = abs(first_norm / second_norm - 1)
    error = abs(first_error / second_error - 1)
    return {
        "reference_norm_relative_change": norm,
        "error_norm_relative_change": error,
        "targets_met": bool(norm <= 0.02 and error <= 0.05),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--columns", type=Path, nargs=2, required=True)
    cli.add_argument("--windows", type=Path, nargs=2, required=True)
    cli.add_argument("--samplings", type=Path, nargs=2, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, checked = [], {}
    columns, windows = [], []
    for path, group in [(p, columns) for p in args.columns] + [(p, windows) for p in args.windows]:
        assert (path.parent / "returncode.txt").read_text().strip() == "0"
        start = json.loads((path / "start.json").read_text())
        for receipt in start["inputs"]:
            key = receipt["path"]
            if key not in checked:
                checked[key] = file_id(key)
            assert checked[key] == receipt
        result = json.loads((path / "result.json").read_text())
        assert result["execution_completed"]
        group.append(result)
        inputs.extend(file_id(path / name) for name in ["start.json", "result.json"])
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": inputs,
            "scope": "Separate global-partition, local polar-rule and through-thickness checks. Observed variation sums "
            "are not certified continuum bounds. Original failed gates and all material contributions remain retained.",
        },
    )
    for path, sampling, result in zip(args.columns, args.samplings, columns):
        inputs.append(file_id(sampling / "queries.npz"))
        with np.load(sampling / "queries.npz") as saved:
            areas = saved["column_areas_m2"]
        with np.load(path / "coarse-columns.npz") as saved:
            candidate = [saved["candidate-point-" + key] for key in ["membrane_n_per_m", "bending_n"]]
        with np.load(path / "fine-columns.npz") as saved:
            reference = [saved["reference48-" + key] for key in ["membrane_n_per_m", "bending_n"]]
        inputs.extend(file_id(path / (body + "-columns.npz")) for body in ["coarse", "fine"])
        for name, a, b in zip(["membrane", "bending"], candidate, reference):
            np.testing.assert_allclose(
                field_difference(a, b, areas), result["candidate_errors"]["coarse"][name + "_relative"], rtol=1e-12
            )
            expected = next(row for row in result["rows"] if row["body"] == "fine")["squared_norms"]["reference48"][
                name
            ]
            np.testing.assert_allclose(areas @ np.sum(b**2, axis=(1, 2)), expected, rtol=1e-12)
    column_report = {}
    for name in ["membrane", "bending"]:
        values = []
        for result in columns:
            norm = (
                next(row for row in result["rows"] if row["body"] == "fine")["squared_norms"]["reference48"][name]
                ** 0.5
            )
            error = result["candidate_errors"]["coarse"][name + "_relative"]
            values.append((norm, error * norm))
        check = changes(values[0][0], values[1][0], values[0][1], values[1][1])
        high = columns[1]
        variations = {
            label: row[name + "_relative"] for label, row in high["reference_changes"].items() if label != "coarse"
        }
        error = high["candidate_errors"]["coarse"][name + "_relative"]
        envelope = error + sum(variations.values())
        column_report[name] = {
            "integration": check,
            "candidate_errors": {body: row[name + "_relative"] for body, row in high["candidate_errors"].items()},
            "reference_changes": variations,
            "observed_variation_sum": envelope,
            "conditional_targets_met": bool(
                check["targets_met"] and max(variations.values()) <= 0.02 and envelope <= 0.05
            ),
        }
    for path, result in zip(args.windows, windows):
        inputs.append(file_id(path / "window-fields.npz"))
        with np.load(path / "window-fields.npz") as saved:
            weights = saved["weights_m3"]
            reference, candidate, pressure = [
                saved[label + "_local_pa"] for label in ["reference48", "candidate", "reference36"]
            ]
        for label, value in [
            ("reference", reference),
            ("candidate_error", candidate - reference),
            ("pressure_change", pressure - reference),
        ]:
            np.testing.assert_allclose(
                weights @ np.sum(value**2, axis=(1, 2)), result["local_window_squared_norms"][label], rtol=1e-12
            )
    low, high = windows
    local = changes(
        low["local_window_squared_norms"]["reference"] ** 0.5,
        high["local_window_squared_norms"]["reference"] ** 0.5,
        low["local_window_squared_norms"]["candidate_error"] ** 0.5,
        high["local_window_squared_norms"]["candidate_error"] ** 0.5,
    )
    low_q, high_q = low["rows"][1]["corrected_squared_norms"], high["rows"][1]["corrected_squared_norms"]
    polar = changes(
        low_q["reference"] ** 0.5,
        high_q["reference"] ** 0.5,
        low_q["candidate_error"] ** 0.5,
        high_q["candidate_error"] ** 0.5,
    )
    low_full, high_full = [row["corrected_squared_norms"] for row in high["rows"]]
    global_partition = changes(
        low_full["reference"] ** 0.5,
        high_full["reference"] ** 0.5,
        low_full["candidate_error"] ** 0.5,
        high_full["candidate_error"] ** 0.5,
    )
    variations = {key: value for key, value in high["rows"][1]["reference_changes"].items() if key != "coarse"}
    error = high["rows"][1]["candidate_errors"]["coarse"]
    envelope = error + sum(variations.values())
    pressure_norm_changes = {
        "local_polar_rule": abs(
            (
                low["local_window_squared_norms"]["pressure_change"]
                / high["local_window_squared_norms"]["pressure_change"]
            )
            ** 0.5
            - 1
        ),
        "complete_polar_rule": abs((low_q["pressure_change"] / high_q["pressure_change"]) ** 0.5 - 1),
        "remaining_global_partition": abs((low_full["pressure_change"] / high_full["pressure_change"]) ** 0.5 - 1),
    }
    volume = {
        "local_rule": local,
        "polar_rule_on_complete_integral": polar,
        "remaining_global_partition": global_partition,
        "candidate_errors": high["rows"][1]["candidate_errors"],
        "reference_changes": variations,
        "observed_variation_sum": envelope,
        "pressure_difference_norm_relative_changes": pressure_norm_changes,
        "conditional_targets_met": bool(
            local["targets_met"]
            and polar["targets_met"]
            and global_partition["targets_met"]
            and max(pressure_norm_changes.values()) <= 0.05
            and max(variations.values()) <= 0.02
            and envelope <= 0.05
        ),
    }
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "inputs": inputs,
            "columns": column_report,
            "volume": volume,
            "finite_surface_global_peak": False,
            "native_scene_accuracy_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
