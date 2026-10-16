"""Separate profile-law changes, traction admissibility and quadrature changes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .profile_reference_sensitivity import tensor_l2
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for geometry, size in (("u", "0.00175"), ("capsule", "0.0025")):
        with np.load(args.runtime_root / f"matrix-mesh-{geometry}-v1/meshes/h{size}.npz") as mesh:
            vertices, tets = mesh["vertices"], mesh["tetrahedra"]
        volumes = abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
        runs = {level: args.runtime_root / f"profile-{geometry}-q{level}-v1" for level in (3, 4)}
        for run in runs.values():
            closed_source(run)
        reports = {level: json.loads((run / "results/summary.json").read_text()) for level, run in runs.items()}
        earlier = {(r["case"], r["profile"]): r for r in reports[3]["rows"]}
        for row in reports[4]["rows"]:
            case, profile = row["case"], row["profile"]
            old = earlier[case, profile]
            assert old["frozen_inputs_sha256"] == row["frozen_inputs_sha256"]
            with np.load(runs[3] / f"results/{case}/{profile}/reference_field.npz") as field:
                old_tensor = field["tensor_Pa"]
            with np.load(runs[4] / f"results/{case}/{profile}/reference_field.npz") as field:
                tensor = field["tensor_Pa"]
            with np.load(runs[4] / f"results/{case}/baseline/reference_field.npz") as field:
                base = field["tensor_Pa"]
            baseline = next(r for r in reports[4]["rows"] if r["case"] == case and r["profile"] == "baseline")
            rows.append(
                {
                    "geometry": geometry,
                    "case": case,
                    "profile": profile,
                    "traction_admissible": row["traction_admissible"],
                    "raw_global_peak_Pa": row["raw_global_peak_Pa"],
                    "energy_J": row["energy_J"],
                    "volume_l2_tensor_change_from_baseline": tensor_l2(tensor - base, volumes)
                    / tensor_l2(base, volumes),
                    "raw_peak_change_from_baseline": abs(
                        row["raw_global_peak_Pa"] / baseline["raw_global_peak_Pa"] - 1
                    ),
                    "quadrature_volume_l2_tensor_relative_change": tensor_l2(tensor - old_tensor, volumes)
                    / tensor_l2(tensor, volumes),
                    "quadrature_raw_peak_relative_change": abs(
                        row["raw_global_peak_Pa"] / old["raw_global_peak_Pa"] - 1
                    ),
                    "frozen_inputs_sha256": row["frozen_inputs_sha256"],
                    "models": row["models"],
                    "patches": row["patches"],
                }
            )
    with (args.output / "profiles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = json.loads((args.runtime_root / "fine-all-summary-v1/results/summary.json").read_text())
    audit = []
    for row in original["rows"]:
        snapshot = json.loads((Path(row["run"]) / f"results/{row['case']}/snapshot.json").read_text())
        values = [p["observables"] for p in snapshot["patches"]]
        admissible = all(
            v["maximum_coulomb_violation_Pa"] <= 1e-10 * max(v["maximum_pressure_Pa"], v["maximum_shear_Pa"], 1)
            for v in values
        )
        audit.append(
            {
                "geometry": row["geometry"],
                "case": row["case"],
                "level": row["level"],
                "admissible": admissible,
                "maximum_coulomb_violation_Pa": max(v["maximum_coulomb_violation_Pa"] for v in values),
                "run": row["run"],
            }
        )
    tails = []
    for tail in original["tails"]:
        selected = [r for r in audit if (r["geometry"], r["case"]) == (tail["geometry"], tail["case"])]
        finest = max(selected, key=lambda r: r["level"])
        tails.append(
            {
                "geometry": tail["geometry"],
                "case": tail["case"],
                "initial_last_change_diagnostic": tail["all_last_changes_below_diagnostic_target"],
                "finest_prescribed_traction_admissible": finest["admissible"],
                "all_levels_prescribed_traction_admissible": all(r["admissible"] for r in selected),
            }
        )
    report = {
        "scope": "Fixed original wrenches with alternative reference laws, independent quadrature bounds and explicit local traction admissibility",
        "profile_rows": rows,
        "original_reference_traction_audit": audit,
        "tail_admissibility": tails,
        "limitations": "Admissibility describes the chosen traction field at sampled quadrature points. A rejected affine-corrected profile does not prove general wrench infeasibility. Last-change diagnostics are not continuum-error or resolved-contact acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile_rows": len(rows),
                "admissible_profiles": sum(r["traction_admissible"] for r in rows),
                "last_change_passes": sum(t["initial_last_change_diagnostic"] for t in tails),
                "last_change_passes_with_finest_admissible_traction": sum(
                    t["initial_last_change_diagnostic"] and t["finest_prescribed_traction_admissible"] for t in tails
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
