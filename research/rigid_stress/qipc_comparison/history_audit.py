"""Summarize complete recorded histories and failed attempts without treating finite prefixes as qualified references."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def audit(folder):
    receipt = json.loads((folder / "result.json").read_text())
    rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    statistics = {}
    for key in (
        "maximum_arm_tracking_error_rad",
        "maximum_finger_tracking_error_m",
        "maximum_robot_authored_origin_error_m",
        "maximum_robot_com_tracking_error_m",
        "maximum_robot_rotation_tracking_error_rad",
        "force_residual_l2_n",
        "force_residual_relative_contact",
        "complete_force_relative_residual",
        "raw_vm_peak_pa",
        "raw_vm_peak_last_physical_substep_pa",
        "raw_maximum_principal_pa",
        "newton",
        "pcg_max",
        "line_search_max",
        "egg_energy_change_minus_contact_work_j",
    ):
        selected = [(row["time_s"], row[key]) for row in rows if key in row and row[key] is not None]
        if selected:
            values = np.array([value for _, value in selected])
            index = np.argmax(np.abs(values)).item()
            statistics[key] = {
                "maximum_absolute_value": np.abs(values[index]).item(),
                "value_at_maximum_absolute": values[index].item(),
                "time_at_maximum_absolute_s": selected[index][0],
                "p99_absolute": np.quantile(np.abs(values), 0.99).item(),
                "samples": len(values),
            }
    return {
        "folder": str(folder),
        "receipt": receipt,
        "receipt_identity": file_id(folder / "result.json"),
        "journal_identity": file_id(folder / "steps.jsonl"),
        "last_recorded_time_s": rows[-1]["time_s"] if rows else None,
        "recorded_rows": len(rows),
        "nonfinite_recorded_rows": sum(not row.get("all_fields_finite", True) for row in rows),
        "newton_cap_recorded_rows": sum(row.get("newton_cap_reached", row.get("newton", 0) >= 256) for row in rows),
        "statistics": statistics,
        "reference_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "result.json",
        {
            "kind": "complete-history-tracking-residual-and-failure-audit",
            "identity": runtime_identity(),
            "histories": [audit(folder) for folder in args.histories],
            "scope": "Actual recorded fields/diagnostics. A native exception can occur after the last recorded accepted state; completion and process status must be checked as well as recorded cap/finite counters. No quantity is reference-qualified by this descriptive audit.",
        },
    )


if __name__ == "__main__":
    main()
