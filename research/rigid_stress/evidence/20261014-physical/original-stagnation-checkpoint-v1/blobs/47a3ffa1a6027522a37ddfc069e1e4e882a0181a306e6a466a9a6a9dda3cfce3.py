"""Compare independently remeshed FEM fields using unchanged physical probes and raw peaks."""

import argparse
import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

GEOMETRIES = ("ellipsoid", "capsule", "u", "thin_neck", "cavity", "narrow_gap", "egg")
CASES = ("opposing", "bending", "asymmetric", "near_slip", "multiple")


def closed_source(run):
    before = (run / "source.sha256").read_bytes()
    assert before == (run / "source-after.sha256").read_bytes(), str(run)
    for line in before.decode().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == expected, (run, relative)


def relative_change(first, second):
    denominator = float(np.linalg.norm(second))
    absolute = float(np.linalg.norm(first - second))
    return absolute / denominator if denominator else None, absolute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--geometries", nargs="+", choices=GEOMETRIES, default=list(GEOMETRIES))
    parser.add_argument("--diagnostic-target", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, tails, inputs = [], [], {}
    for geometry in args.geometries:
        initial = (
            [args.runtime_root / "load-matrix-egg-l0-v1"]
            if geometry == "egg"
            else [args.runtime_root / f"load-matrix-{geometry}-l{level}-v3" for level in range(3)]
        )
        runs = initial + [args.runtime_root / f"fine-ref-{geometry}-l{level}-v1" for level in range(3)]
        summaries = []
        for run in runs:
            closed_source(run)
            probes = Path(str(run) + "-probes")
            closed_source(probes)
            summary = json.loads((probes / "results/summary.json").read_text())
            assert summary["geometry"] == geometry
            summaries.append(summary)
        assert len({s["probe_coordinate_sha256"] for s in summaries}) == 1
        assert all(b["tetrahedra"] > a["tetrahedra"] for a, b in pairwise(summaries))
        for case in CASES:
            previous, case_rows = None, []
            for level, (run, summary) in enumerate(zip(runs, summaries)):
                snapshot_path = run / f"results/{case}/snapshot.json"
                probe_path = Path(str(run) + "-probes") / f"results/{case}/reference_field.npz"
                snapshot = json.loads(snapshot_path.read_text())
                assert snapshot["reference_complete_budget_fraction"] <= 1, str(snapshot_path)
                assert snapshot["mesh_sha256"] == summary["mesh_sha256"]
                for path in (snapshot_path, probe_path):
                    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(probe_path) as saved:
                    field = {key: saved[key].copy() for key in saved.files}
                row = {
                    "geometry": geometry,
                    "case": case,
                    "level": level,
                    "run": str(run),
                    "tetrahedra": summary["tetrahedra"],
                    "mesh_sha256": summary["mesh_sha256"],
                    "valid_probes": summary["valid_count"],
                    "outside_probe_indices": summary["outside_probe_indices"],
                    "raw_global_peak_Pa": snapshot["reference_peak_Pa"],
                    "raw_global_peak_position_m": snapshot["reference_peak_position_m"],
                    "energy_J": snapshot["reference_energy_J"],
                    "mass_kg": snapshot["mass_kg"],
                    "com_m": snapshot["com_m"],
                    "complete_budget_fraction": snapshot["reference_complete_budget_fraction"],
                }
                if previous is not None:
                    old_snapshot, old_field = previous
                    for key in ("material", "pose", "omega_rad_s", "patch_radius_m", "reference_quadrature_level"):
                        assert snapshot[key] == old_snapshot[key], (geometry, case, key)
                    assert [p["physical_anchor_m"] for p in snapshot["patches"]] == [
                        p["physical_anchor_m"] for p in old_snapshot["patches"]
                    ]
                    assert [p["friction"] for p in snapshot["patches"]] == [
                        p["friction"] for p in old_snapshot["patches"]
                    ]
                    np.testing.assert_array_equal(field["points_m"], old_field["points_m"])
                    common = field["valid"] & old_field["valid"]
                    assert common.any()
                    row["common_valid_probes"] = int(common.sum())
                    row["excluded_pair_probe_indices"] = np.flatnonzero(~common).tolist()
                    for label, key, units in (
                        ("tensor", "tensor_Pa", "Pa"),
                        ("vm", "vm_Pa", "Pa"),
                        ("displacement", "displacement_m", "m"),
                    ):
                        relative, absolute = relative_change(old_field[key][common], field[key][common])
                        row[f"adjacent_probe_{label}_relative_change"] = relative
                        row[f"adjacent_probe_{label}_absolute_change_{units}"] = absolute
                    for label, key in (("raw_peak", "reference_peak_Pa"), ("energy", "reference_energy_J")):
                        row[f"adjacent_{label}_relative_change"] = abs(old_snapshot[key] / snapshot[key] - 1)
                    row["adjacent_raw_peak_position_shift_m"] = float(
                        np.linalg.norm(
                            np.array(snapshot["reference_peak_position_m"])
                            - np.array(old_snapshot["reference_peak_position_m"])
                        )
                    )
                rows.append(row)
                case_rows.append(row)
                previous = snapshot, field
            last = case_rows[-1]
            changes = {
                label: last[f"adjacent_{label}_relative_change"]
                for label in ("probe_tensor", "probe_vm", "probe_displacement", "raw_peak", "energy")
            }
            tails.append(
                {
                    "geometry": geometry,
                    "case": case,
                    "useful_levels": len(case_rows),
                    "finest_tetrahedra": last["tetrahedra"],
                    "last_changes": changes,
                    "all_last_changes_below_diagnostic_target": all(
                        value is not None and value < args.diagnostic_target for value in changes.values()
                    ),
                    "raw_peak_increases_on_last_two_refinements": all(
                        b["raw_global_peak_Pa"] > a["raw_global_peak_Pa"]
                        for a, b in zip(case_rows[-3:], case_rows[-2:])
                    ),
                    "complete_equilibrium_qualified": all(r["complete_budget_fraction"] <= 1 for r in case_rows),
                    "finest_outside_probe_indices": last["outside_probe_indices"],
                }
            )
    with (args.output / "refinement.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "scope": "Combined CAD approximation and volume refinement of independent finite-traction FEM references",
        "diagnostic_target": args.diagnostic_target,
        "goal_completion": False,
        "rows": rows,
        "tails": tails,
        "input_sha256": inputs,
        "limitations": (
            "Adjacent changes are convergence diagnostics, not continuum error or native accuracy. "
            "Common probes remain fixed and outside points are explicit. Near-slip normals follow each "
            "approximate material boundary. This curve combines volume and geometry changes; frozen "
            "polyhedron and quadrature studies are separate. Full contact/profile/history/cost acceptance remains required."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fields": len(rows),
                "tails": len(tails),
                "last_changes_below_target": sum(t["all_last_changes_below_diagnostic_target"] for t in tails),
                "goal_completion": False,
            }
        )
    )


if __name__ == "__main__":
    main()
