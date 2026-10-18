"""Retain numerical rejection and deformed-moment drift in complete Shell dead-load replay summaries."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import time_integrated_error
from .history_audit import returncode
from .tables import describe


def summarize(folder):
    result = json.loads((folder / "result.json").read_text())
    rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    task_rc = returncode(folder.parent / f"{folder.name}.returncode.txt")
    group_rc = returncode(folder.parent / "returncode.txt")
    with np.load(folder / "rest.npz") as saved:
        actual_wall = {
            "actual_lumped_mass_kg": saved["masses_kg"].sum().item(),
            "minimum_face_thickness_m": saved["face_thickness_m"].min().item(),
            "maximum_face_thickness_m": saved["face_thickness_m"].max().item(),
        }
    records = {}
    for key in (
        "static_relative_force_residual",
        "static_force_residual_l2_n",
        "balanced_load_l2_n",
        "balance_correction_l2_n",
        "raw_vm_peak_pa",
    ):
        records[key] = describe([(row["time_s"], row.get(key)) for row in rows])
    for kind in ("contact", "inertial"):
        for metric in ("net_norm_nm", "sum_local_moment_drift_norms_nm"):
            samples = []
            for row in rows:
                entry = row.get(f"deformed_{kind}_moment_drift")
                if entry is not None:
                    value = (
                        np.linalg.norm(entry["net_moment_drift_nm"]).item()
                        if metric == "net_norm_nm"
                        else entry[metric]
                    )
                    samples.append((row["time_s"], value))
            records[f"{kind}_deformed_moment_drift_{metric}"] = describe(samples)
    temporal = {}
    full_grid = len(rows) == 600 and np.allclose(
        [row["time_s"] for row in rows], np.arange(1, 601) * 0.01, rtol=0, atol=1e-12
    )
    all_finite = all(row["all_fields_finite"] and row["errors"] is not None for row in rows)
    if full_grid and all_finite:
        for key, reference_key, error_key, floor in (
            ("common_tangent", "reference_tensor_rms_pa", "absolute_tensor_rms_error_pa", 1.0),
            ("membrane", "reference_weighted_rms", "absolute_weighted_rms_error", 5e-4),
            ("bending", "reference_weighted_rms", "absolute_weighted_rms_error", 2.5e-7),
        ):
            temporal[key] = time_integrated_error(
                [row["time_s"] for row in rows],
                [row["errors"][key][reference_key] for row in rows],
                [row["errors"][key][error_key] for row in rows],
                floor,
            )
    phases = []
    for start, stop, name in (
        (0, 0.9, "approach"),
        (0.9, 1.8, "squeeze"),
        (1.8, 3.6, "lift-hold"),
        (3.6, 4.2, "lateral"),
        (4.2, 4.56, "weak"),
        (4.56, 5.4, "hold-after-weak"),
        (5.4, 6.00001, "release-recontact"),
    ):
        selected = [row for row in rows if start < row["time_s"] <= stop]
        phases.append(
            {
                "phase": name,
                "replays": len(selected),
                "admitted": sum(row["static_numerically_admitted"] for row in selected),
            }
        )
    return {
        "source": file_id(folder / "result.json"),
        "journal": file_id(folder / "steps.jsonl"),
        "wall_input": result["wall_input"],
        "actual_wall": actual_wall,
        "transfer_controls": result["transfer_controls"],
        "task_returncode": task_rc,
        "group_returncode": group_rc,
        "complete_replay_set_established": bool(
            result["completed_requested_replay_set"]
            and len(rows) == result["requested_replays"]
            and (task_rc == 0 or (task_rc is None and group_rc == 0))
        ),
        "full600_observation_grid": bool(full_grid),
        "all_recorded_fields_finite": bool(all_finite),
        "admitted_replays": sum(row["static_numerically_admitted"] for row in rows),
        "rejected_replays": sum(not row["static_numerically_admitted"] for row in rows),
        "pcg_cap_steps": sum(row["pcg_cap_steps"] for row in rows),
        "statistics": records,
        "phase_coverage": phases,
        "diagnostic_all_finite_replay_time_norms": temporal,
        "all_static_replays_numerically_admitted": all(row["static_numerically_admitted"] for row in rows),
        "original_scope": result["scope"],
        "scope": "Time norms retain every finite replay, including rejected static states; they are diagnostic field discrepancies, not admitted static or physical accuracy scores. Phase counts retain the exact source outcomes. Moment drift is measured at deformed Shell vertices, with both a net norm and a sum of local norms to expose cancellation. Near-zero denominators, balance corrections and strict relative-force failures stay explicit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "result.json",
        {
            "kind": "full-shell-load-replay-admission-and-moment-audit",
            "identity": runtime_identity(),
            "replays": [summarize(folder) for folder in args.replays],
            "reference_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
