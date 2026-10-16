"""Compare contact integration and intrinsic ports on identical frozen physical snapshots."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for geometry in ("u", "capsule"):
        baseline = args.runtime_root / f"load-matrix-{geometry}-l2-v3"
        base = json.loads((baseline / "results/summary.json").read_text())["cases"]
        for quadrature, ports in ((6, 8), (14, 8), (10, 4), (10, 16)):
            run = args.runtime_root / f"axis-{geometry}-q{quadrature}-p{ports}-v1"
            assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
            cases = json.loads((run / "results/summary.json").read_text())["cases"]
            for current in cases:
                previous = next(case for case in base if case["case"] == current["case"])
                path = run / "results" / current["case"]
                base_path = baseline / "results" / current["case"]
                assert (path / "frozen_inputs.npz").read_bytes() == (base_path / "frozen_inputs.npz").read_bytes()
                for model in current["models"]:
                    old_model = next(row for row in previous["models"] if row["model"] == model["model"])
                    row = {
                        "geometry": geometry,
                        "case": current["case"],
                        "model": model["model"],
                        "quadrature": quadrature,
                        "ports": ports,
                        "classification": model["classification"],
                        "frozen_inputs_sha256": hashlib.sha256((path / "frozen_inputs.npz").read_bytes()).hexdigest(),
                    }
                    if model["accepted"] and old_model["accepted"]:
                        with (
                            np.load(path / f"{model['model']}_field.npz") as changed,
                            np.load(base_path / f"{model['model']}_field.npz") as old,
                        ):
                            for label, key in (
                                ("tensor", "raw_native_tensor_Pa"),
                                ("vm", "vm_Pa"),
                                ("displacement", "displacement"),
                            ):
                                row[f"{label}_relative_change_from_q10_p8"] = float(
                                    np.linalg.norm(changed[key] - old[key]) / np.linalg.norm(old[key])
                                )
                        row.update(
                            {
                                "raw_global_peak_Pa": model["peak_Pa"],
                                "raw_peak_relative_change_from_q10_p8": abs(
                                    model["peak_Pa"] / old_model["peak_Pa"] - 1
                                ),
                                "energy_relative_change_from_q10_p8": abs(
                                    model["energy_J"] / old_model["energy_J"] - 1
                                ),
                                "complete_budget_fraction": model["complete_budget_fraction"],
                                "maximum_coulomb_violation_Pa": model["maximum_coulomb_violation_Pa"],
                                "maximum_island_force_movement_N": max(
                                    float(np.linalg.norm(island["force_shift_from_prior_N"]))
                                    for island in model["islands"]
                                ),
                            }
                        )
                    rows.append(row)
    assert len(rows) == 160
    keys = sorted({key for row in rows for key in row})
    with (args.output / "axis_scan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Native quadrature6/10/14 and intrinsic ports4/8/16, fixed fine-matrix meshes and byte-identical frozen inputs",
                "rows": rows,
                "unverified": "These axes do not establish volume/geometry convergence, radius/profile accuracy or resolved grasp feedback",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"rows": len(rows), "qualified": sum(row["classification"] == "numerically_qualified" for row in rows)}
        )
    )


if __name__ == "__main__":
    main()
