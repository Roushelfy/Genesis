"""Retain all eight original/extended CAD levels and their original five stress diagnostics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import CASES, closed_source, relative_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, rows, tails, audits = {}, [], [], []

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for shape in ["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"]:
        auditor = args.runtime_root / f"extended-finite-{shape}-three-level-audit-v1"
        closed_source(auditor)
        assert int((auditor / "returncode.txt").read_text()) == 0
        path = auditor / "results/summary.json"
        retain(path)
        audit = json.loads(path.read_text())
        assert audit["all_original_and_new_fields_freshly_audited"] and len(audit["rows"]) == 15
        assert len(audit["adjacent_changes"]) == 10
        audits.append({"geometry": shape, "audit": str(auditor), "fresh_fields": 15})
        runs = [args.runtime_root / f"load-matrix-{shape}-l{level}-v3" for level in range(3)]
        runs += [args.runtime_root / f"fine-ref-{shape}-l{level}-v1" for level in range(3)]
        runs += [args.runtime_root / f"extended-finite-{shape}-level{level}-v1" for level in [1, 2]]
        for run in runs:
            closed_source(run)
            assert int((run / "returncode.txt").read_text()) == 0
        for case in CASES:
            previous, baseline, case_rows = None, None, []
            for level, run in enumerate(runs):
                path = run / f"results/{case}/snapshot.json"
                retain(path)
                snapshot = json.loads(path.read_text())
                probe_path = (
                    Path(str(run) + "-probes") / f"results/{case}/reference_field.npz"
                    if level < 6
                    else run / f"results/{case}/probe_field.npz"
                )
                if level < 6:
                    probe_run = Path(str(run) + "-probes")
                    closed_source(probe_run)
                    assert int((probe_run / "returncode.txt").read_text()) == 0
                retain(probe_path)
                with np.load(probe_path) as saved:
                    field = {
                        name: saved[name].copy()
                        for name in ["points_m", "valid", "tensor_Pa", "vm_Pa", "displacement_m"]
                    }
                physical = {
                    key: snapshot[key]
                    for key in ["material", "pose", "omega_rad_s", "patch_radius_m", "reference_quadrature_level"]
                }
                physical["patch_anchors"] = [patch["physical_anchor_m"] for patch in snapshot["patches"]]
                physical["patch_frictions"] = [patch["friction"] for patch in snapshot["patches"]]
                if baseline is None:
                    baseline = physical
                assert physical == baseline
                assert snapshot["reference_complete_budget_fraction"] <= 1
                if level < 6:
                    summary_path = Path(str(run) + "-probes") / "results/summary.json"
                    retain(summary_path)
                    count = json.loads(summary_path.read_text())["tetrahedra"]
                else:
                    count = snapshot["tetrahedra"]
                row = {
                    "geometry": shape,
                    "case": case,
                    "level": level,
                    "run": str(run),
                    "tetrahedra": count,
                    "raw_global_peak_Pa": snapshot["reference_peak_Pa"],
                    "raw_global_peak_position_m": snapshot["reference_peak_position_m"],
                    "energy_J": snapshot["reference_energy_J"],
                    "mass_kg": snapshot["mass_kg"],
                    "com_m": snapshot["com_m"],
                    "complete_budget_fraction": snapshot["reference_complete_budget_fraction"],
                    "original_probes": len(field["points_m"]),
                    "outside_probe_indices": np.flatnonzero(~field["valid"]).tolist(),
                    "separately_fresh_complete_field_audit": level >= 5,
                }
                if previous is not None:
                    old, old_field = previous
                    assert count > old["tetrahedra"]
                    np.testing.assert_array_equal(field["points_m"], old_field["points_m"])
                    common = field["valid"] & old_field["valid"]
                    assert common.any()
                    row["common_valid_probes"] = int(common.sum())
                    row["excluded_pair_probe_indices"] = np.flatnonzero(~common).tolist()
                    for name, label in [("tensor_Pa", "tensor"), ("vm_Pa", "vm"), ("displacement_m", "displacement")]:
                        relative, absolute = relative_change(old_field[name][common], field[name][common])
                        row[f"adjacent_probe_{label}_relative_change"] = relative
                        row[f"adjacent_probe_{label}_absolute_change"] = absolute
                    row["adjacent_raw_peak_relative_change"] = (
                        abs(row["raw_global_peak_Pa"] - old["raw_global_peak_Pa"]) / row["raw_global_peak_Pa"]
                    )
                    row["adjacent_raw_peak_absolute_change_Pa"] = abs(
                        row["raw_global_peak_Pa"] - old["raw_global_peak_Pa"]
                    )
                    row["adjacent_energy_relative_change"] = abs(row["energy_J"] - old["energy_J"]) / abs(
                        row["energy_J"]
                    )
                    row["adjacent_energy_absolute_change_J"] = abs(row["energy_J"] - old["energy_J"])
                rows.append(row)
                case_rows.append(row)
                previous = row, field
            keys = ["probe_tensor", "probe_vm", "probe_displacement", "raw_peak", "energy"]
            latest = {key: case_rows[-1][f"adjacent_{key}_relative_change"] for key in keys}
            both = all(
                row[f"adjacent_{key}_relative_change"] is not None and row[f"adjacent_{key}_relative_change"] < 0.03
                for row in case_rows[-2:]
                for key in keys
            )
            audited_tail = next(row for row in audit["tails"] if row["case"] == case)
            assert both == audited_tail["both_latest_pairs_below_initial_diagnostic"]
            for key in keys:
                np.testing.assert_allclose(
                    latest[key], audited_tail["latest_pair_changes"][f"{key}_relative_change"], rtol=1e-10, atol=1e-13
                )
            tails.append(
                {
                    "geometry": shape,
                    "case": case,
                    "useful_levels": 8,
                    "finest_tetrahedra": case_rows[-1]["tetrahedra"],
                    "latest_changes": latest,
                    "both_latest_pairs_below_initial_diagnostic": both,
                    "physical_convergence_accepted": False,
                }
            )
    assert len(rows) == 240 and len(tails) == 30
    with (args.output / "refinement.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "rows": rows,
        "tails": tails,
        "independent_latest_three_level_audits": audits,
        "initial_diagnostic_target": 0.03,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "All six earlier levels and two added levels retain original physical inputs, probes, outside indices, raw whole-domain peaks and energies. Fresh complete force/corner/quadrature/probe audits cover the latest three levels per shape; earlier levels retain their documented historical audit scopes. Both latest3percent diagnostics are supplemental indicators and do not establish continuum error bounds. CAD/volume approximation remains coupled; local/contact regions, separated axes, native model/feedback/application and warmed accuracy/cost remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
