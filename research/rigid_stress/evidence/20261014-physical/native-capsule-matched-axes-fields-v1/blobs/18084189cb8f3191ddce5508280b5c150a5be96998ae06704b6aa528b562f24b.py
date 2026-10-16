"""Locate the unchanged thickness-axis probe discrepancy without replacing its declared observable."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .summarize_fine_references import CASES, closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.audit, args.previous, args.current]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    for run in [args.previous, args.current]:
        for child in ["fem", "probes"]:
            closed_source(run / child)
            assert int((run / child / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    audit_path = args.audit / "results/summary.json"
    retain(audit_path)
    audit = json.loads(audit_path.read_text())
    layers = [json.loads((run / "mesh.json").read_text())["shell_layers"] for run in [args.previous, args.current]]
    assert layers == [4, 5]
    rows, csv_rows = [], []
    for case in CASES:
        fields = []
        for run, layer in zip([args.previous, args.current], layers):
            field = run / f"probes/results/{case}/reference_field.npz"
            qualified = next(row for row in audit["rows"] if row["case"] == case and row["layers"] == layer)
            assert retain(field) == qualified["probe_field_sha256"]
            with np.load(field) as saved:
                fields.append({key: saved[key].copy() for key in saved.files})
        first, second = fields
        np.testing.assert_array_equal(first["points_m"], second["points_m"])
        assert np.all(first["valid"]) and np.all(second["valid"])
        delta = second["tensor_Pa"] - first["tensor_Pa"]
        trace = np.trace(delta, axis1=-2, axis2=-1)
        hydrostatic = trace[:, None, None] * np.eye(3) / 3
        deviatoric = delta - hydrostatic
        squared = np.sum(delta**2, axis=(-2, -1))
        total_squared = float(squared.sum())
        hydro_squared = float(np.sum(hydrostatic**2))
        deviator_squared = float(np.sum(deviatoric**2))
        np.testing.assert_allclose(total_squared, hydro_squared + deviator_squared, rtol=1e-12, atol=1e-12)
        reference_norm = float(np.linalg.norm(second["tensor_Pa"]))
        relative = float(np.sqrt(total_squared) / reference_norm)
        original = next(
            row
            for row in audit["adjacent_changes"]
            if row["case"] == case and row["from_layers"] == layers[0] and row["to_layers"] == layers[1]
        )
        np.testing.assert_allclose(relative, original["tensor_Pa"]["relative_change"], rtol=1e-12)
        snapshot = args.current / f"fem/results/{case}/snapshot.json"
        retain(snapshot)
        config = json.loads(snapshot.read_text())
        patches = []
        for index in range(len(config["patches"])):
            path = args.current / f"fem/results/{case}/reference_patch{index}.npz"
            retain(path)
            with np.load(path) as saved:
                active = np.linalg.norm(saved["tractions_Pa"], axis=1) > 0
                patches.append(saved["positions_m"][active])
        traction_positions = np.concatenate(patches)
        distances = cKDTree(traction_positions).query(second["points_m"])[0]
        ordering = np.argsort(squared)[::-1]
        top = ordering[:5]
        rows.append(
            {
                "case": case,
                "unchanged_probe_count": len(delta),
                "original_full_tensor_relative_change": relative,
                "full_tensor_absolute_change_Pa": float(np.sqrt(total_squared)),
                "hydrostatic_fraction_of_squared_tensor_difference": hydro_squared / total_squared,
                "deviatoric_fraction_of_squared_tensor_difference": deviator_squared / total_squared,
                "component_squared_difference_Pa2": np.sum(delta**2, axis=0).tolist(),
                "top_five_probe_indices": top.tolist(),
                "top_five_fraction_of_squared_tensor_difference": float(squared[top].sum() / total_squared),
                "top_five_positions_m": second["points_m"][top].tolist(),
                "top_five_distance_to_nonzero_traction_samples_m": distances[top].tolist(),
                "unchanged_raw_global_peak_Pa": config["reference_peak_Pa"],
                "unchanged_raw_global_peak_position_m": config["reference_peak_position_m"],
            }
        )
        for index, point in enumerate(second["points_m"]):
            row = {
                "case": case,
                "probe": index,
                "x_m": point[0],
                "y_m": point[1],
                "z_m": point[2],
                "distance_to_nonzero_traction_samples_m": distances[index],
                "tensor_difference_Frobenius_Pa": float(np.sqrt(squared[index])),
                "fraction_of_squared_tensor_difference": float(squared[index] / total_squared),
            }
            for i in range(3):
                for j in range(3):
                    row[f"previous_sigma{i}{j}_Pa"] = first["tensor_Pa"][index, i, j]
                    row[f"current_sigma{i}{j}_Pa"] = second["tensor_Pa"][index, i, j]
            csv_rows.append(row)
        np.savez_compressed(
            args.output / f"{case}.npz",
            points_m=second["points_m"],
            previous_tensor_Pa=first["tensor_Pa"],
            current_tensor_Pa=second["tensor_Pa"],
            tensor_difference_Pa=delta,
            hydrostatic_tensor_difference_Pa=hydrostatic,
            deviatoric_tensor_difference_Pa=deviatoric,
            distance_to_nonzero_traction_samples_m=distances,
        )
    with (args.output / "all_original_probe_components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "audit": str(args.audit),
        "previous": str(args.previous),
        "current": str(args.current),
        "rows": rows,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "All original36 physical probes and the original full-tensor Frobenius observable remain unchanged; all180 point/component records are exported. Hydrostatic/deviatoric squared norms form an orthogonal tensor decomposition, not an additive decomposition of geometry/bulk/contact errors. The five largest differences are diagnostic reporting only; no point is removed or renormalized. Distance to finite traction quadrature samples is not a continuum boundary distance or a singularity proof. Raw global peaks remain those of the independently checked full fields. Probe localization does not establish volume-integrated accuracy, physical convergence or the full application/model/feedback/cost objective.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "original_probe_records": len(csv_rows)}), flush=True)


if __name__ == "__main__":
    main()
