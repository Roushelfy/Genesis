"""Report fixed-probe refinement tails, raw peaks and local redistribution without inferring physical acceptance."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def relative_change(actual, expected):
    denominator = np.linalg.norm(expected)
    if denominator == 0:
        return None
    return float(np.linalg.norm(actual - expected) / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, missing = [], []
    for geometry in ("ellipsoid", "capsule", "u", "thin_neck", "cavity", "narrow_gap"):
        runs = [args.runtime_root / f"load-matrix-{geometry}-l{level}-{args.version}" for level in range(3)]
        if any(not (run / "source-after.sha256").exists() for run in runs):
            missing.append(geometry)
            continue
        for run in runs:
            assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
        snapshots = [json.loads((run / "results/summary.json").read_text())["cases"] for run in runs]
        probe_roots = [Path(str(run) + "-probes") / "results" for run in runs]
        if any(not (root / "summary.json").exists() for root in probe_roots):
            missing.append(geometry + ":probes")
            continue
        probe_reports = [json.loads((root / "summary.json").read_text()) for root in probe_roots]
        assert len({report["probe_coordinate_sha256"] for report in probe_reports}) == 1
        if any(report["outside_probe_indices"] for report in probe_reports):
            missing.append(geometry + ":outside-fixed-probes")
            continue
        for case in ("opposing", "bending", "asymmetric", "near_slip", "multiple"):
            cases = [next(row for row in snapshot if row["case"] == case) for snapshot in snapshots]
            with np.load(probe_roots[-1] / case / "reference_field.npz") as field:
                finest_reference_tensor = field["tensor_Pa"]
            for name in ("reference", "pressure", "componentwise", "pair_union", "body_source_prior"):
                previous_tensor, previous_vm, previous_peak, previous_energy = None, None, None, None
                for level, (current, probe_report) in enumerate(zip(cases, probe_reports)):
                    row = {
                        "geometry": geometry,
                        "case": case,
                        "model": name,
                        "level": level,
                        "tetrahedra": probe_report["tetrahedra"],
                    }
                    if name == "reference":
                        peak, energy = current["reference_peak_Pa"], current["reference_energy_J"]
                        row["classification"] = "independent_continuous_traction_reference"
                        row["peak_position_m"] = current["reference_peak_position_m"]
                    else:
                        model = next(item for item in current["models"] if item["model"] == name)
                        row["classification"] = model["classification"]
                        if not model["accepted"]:
                            rows.append(row)
                            previous_tensor, previous_vm, previous_peak, previous_energy = None, None, None, None
                            continue
                        peak, energy = model["peak_Pa"], model["energy_J"]
                        raw_probe = next(
                            item
                            for item in probe_report["rows"]
                            if item["case"] == case and item["field"] == f"{name}_field"
                        )
                        row["peak_position_m"] = raw_probe["raw_native_peak_position_m"]
                        row["independent_replay_peak_position_m"] = model["peak_position_m"]
                        row["maximum_coulomb_violation_Pa"] = model["maximum_coulomb_violation_Pa"]
                        row["maximum_island_force_shift_N"] = max(
                            float(np.linalg.norm(island["force_shift_from_prior_N"])) for island in model["islands"]
                        )
                        row["maximum_island_moment_shift_Nm"] = max(
                            float(np.linalg.norm(island["moment_shift_from_prior_Nm"])) for island in model["islands"]
                        )
                    with np.load(probe_roots[level] / case / f"{name}_field.npz") as field:
                        tensor, vm = field["tensor_Pa"], field["vm_Pa"]
                    with np.load(probe_roots[level] / case / "reference_field.npz") as field:
                        same_reference = field["tensor_Pa"]
                    row.update(
                        {
                            "raw_global_peak_Pa": peak,
                            "energy_J": energy,
                            "probe_tensor_error_vs_same_mesh_reference": relative_change(tensor, same_reference),
                            "probe_tensor_error_vs_finest_reference": relative_change(tensor, finest_reference_tensor),
                        }
                    )
                    if previous_tensor is not None:
                        row.update(
                            {
                                "adjacent_tensor_relative_change": relative_change(previous_tensor, tensor),
                                "adjacent_vm_relative_change": relative_change(previous_vm, vm),
                                "adjacent_raw_peak_relative_change": abs(previous_peak / peak - 1),
                                "adjacent_energy_relative_change": abs(previous_energy / energy - 1),
                            }
                        )
                        row["all_four_tail_changes_below_initial_3_percent"] = all(
                            row[key] is not None and row[key] <= 0.03
                            for key in (
                                "adjacent_tensor_relative_change",
                                "adjacent_vm_relative_change",
                                "adjacent_raw_peak_relative_change",
                                "adjacent_energy_relative_change",
                            )
                        )
                    previous_tensor, previous_vm, previous_peak, previous_energy = tensor, vm, peak, energy
                    rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with (args.output / "refinement.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Combined volume and piecewise-affine geometry refinement with fixed continuum profiles, radii and friction",
                "criterion": "Initial 3% peak/energy/tensor/vm tail diagnostic; not a universal physical-accuracy gate",
                "unseparated_axes": [
                    "geometry versus volume",
                    "reference traction quadrature",
                    "native quadrature",
                    "intrinsic ports",
                ],
                "missing": missing,
                "rows": rows,
                "remaining": "Independent finer reference, separate refinement axes, radius/profile sensitivity, changing histories, egg and resolved interior-hotspot contact",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(rows), "missing": missing}))


if __name__ == "__main__":
    main()
