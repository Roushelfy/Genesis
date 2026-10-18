"""Summarize complete recorded histories and failed attempts without treating finite prefixes as qualified references."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def read_receipt(path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Receipt must be a JSON object")
        return value, None
    except (OSError, ValueError) as error:
        return None, str(error)


def read_journal(path):
    rows, offset, error = [], 0, None
    if not path.is_file():
        return rows, {"error": "Journal missing", "valid_prefix_rows": 0}
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or "time_s" not in row:
                    raise ValueError("Journal row must be an object with time_s")
            except (ValueError, UnicodeError) as exception:
                error = {"line": number, "byte_offset": offset, "error": str(exception)}
                break
            rows.append(row)
            offset += len(line)
    return rows, {
        "valid_prefix_rows": len(rows),
        "first_unreadable_row": error,
        "unparsed_tail_bytes": path.stat().st_size - offset,
        "scope": "Only the structurally valid prefix is summarized. An unreadable tail is explicit and remains unchanged in the source; it is not accepted as a complete history.",
    }


def returncode(path):
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def audit(folder):
    receipt_path, journal_path = folder / "result.json", folder / "steps.jsonl"
    receipt, receipt_error = read_receipt(receipt_path)
    inputs, input_error = read_receipt(folder / "inputs.json")
    rows, journal_read = read_journal(journal_path)
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
        "body_energy_change_minus_all_recorded_work_j",
        "total_energy_change_minus_actuator_work_j",
        "maximum_reported_contact_penetration_m",
        "maximum_contact_friction_utilization",
        "minimum_det_F",
        "minimum_surface_height_m",
        "minimum_vertex_height_m",
        "equality_inertia_coefficient_relative_difference",
        "minimum_reported_gap_ratio",
        "ccd_maximum_iterations",
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
                "minimum_value": values.min().item(),
                "time_at_minimum_s": selected[values.argmin()][0],
                "maximum_value": values.max().item(),
                "time_at_maximum_s": selected[values.argmax()][0],
            }
    force_vectors = [
        (row["time_s"], row["force_residual_resultant_n"]) for row in rows if "force_residual_resultant_n" in row
    ]
    if force_vectors:
        norms = np.linalg.norm([value for _, value in force_vectors], axis=1)
        with np.load(folder / "rest.npz") as saved:
            weight = saved["masses_kg"].sum().item() * 9.81
        statistics["net_force_residual"] = {
            "maximum_n": norms.max().item(),
            "maximum_time_s": force_vectors[norms.argmax()][0],
            "rms_n": np.sqrt(np.mean(norms**2)).item(),
            "maximum_relative_to_egg_weight": (norms.max() / weight).item(),
            "scope": "Actual residual resultant, distinct from a residual normalized by opposing pinch-force norm",
        }
    config = (receipt or {}).get("config", (inputs or {}).get("config", {}))
    pcg_cap = config.get("max_pcg")
    newton_cap = config.get("max_newton")
    newton_flags, pcg_flags, finite_flags = [], [], []
    for row in rows:
        if "newton_cap_reached" in row:
            newton_flags.append(bool(row["newton_cap_reached"]))
        elif "newton_budget_exhausted" in row.get("status", {}):
            newton_flags.append(bool(row["status"]["newton_budget_exhausted"]))
        elif newton_cap is not None and "newton" in row:
            newton_flags.append(row["newton"] >= newton_cap)
        if "pcg_cap_reached" in row:
            pcg_flags.append(bool(row["pcg_cap_reached"]))
        elif pcg_cap is not None and "pcg_max" in row:
            pcg_flags.append(row["pcg_max"] >= pcg_cap)
        if "all_fields_finite" in row:
            finite_flags.append(bool(row["all_fields_finite"]))
    task_rc = returncode(folder.parent / f"{folder.name}.returncode.txt")
    group_rc = returncode(folder.parent / "returncode.txt")
    expected_rows = (receipt or {}).get("executed_steps", (receipt or {}).get("executed_control_ticks"))
    shell_caps = [row["cap_exhausted_substeps"] for row in rows if "cap_exhausted_substeps" in row]
    et_values = [
        (row["time_s"], row["edge_triangle_intersection_count"])
        for row in rows
        if row.get("edge_triangle_intersection_count") is not None
    ]
    log_path = folder.parent / f"{folder.name}.log"
    et_frames = None
    if log_path.is_file():
        et_frames = [
            int(value)
            for value in re.findall(
                r"ET intersection detected at frame (\d+), forcing Newton exit", log_path.read_text(errors="replace")
            )
        ]
    et_detected = (
        True
        if et_frames or any(value > 0 for _, value in et_values)
        else (False if rows and len(et_values) == len(rows) else None)
    )
    first_et_time = next((time for time, value in et_values if value > 0), None)
    if (
        first_et_time is None
        and et_frames
        and (receipt or {}).get("kind", "").startswith("QIPC-prescribed")
        and "dt" in config
    ):
        first_et_time = (et_frames[0] + 1) * config["dt"]
    return {
        "folder": str(folder),
        "receipt": receipt,
        "receipt_read_error": receipt_error,
        "receipt_identity": file_id(receipt_path) if receipt_path.is_file() else None,
        "input_receipt_read_error": input_error,
        "input_receipt_identity": file_id(folder / "inputs.json") if (folder / "inputs.json").is_file() else None,
        "journal_identity": file_id(journal_path) if journal_path.is_file() else None,
        "journal_read": journal_read,
        "task_returncode": task_rc,
        "group_returncode": group_rc,
        "completed_history_reported": (receipt or {}).get("completed_requested_history"),
        "complete_history_established": bool(
            (receipt or {}).get("completed_requested_history")
            and expected_rows is not None
            and len(rows) == expected_rows
            and journal_read.get("unparsed_tail_bytes") == 0
            and len(finite_flags) == len(rows)
            and all(finite_flags)
            and (task_rc == 0 or (task_rc is None and group_rc == 0))
        ),
        "last_recorded_time_s": rows[-1]["time_s"] if rows else None,
        "recorded_rows": len(rows),
        "nonfinite_recorded_rows": sum(not value for value in finite_flags) if finite_flags else None,
        "finite_flag_observations": len(finite_flags),
        "newton_cap_recorded_rows": sum(newton_flags) if newton_flags else None,
        "newton_cap_observations": len(newton_flags),
        "pcg_cap_recorded_rows": sum(pcg_flags) if pcg_flags else None,
        "pcg_cap_observations": len(pcg_flags),
        "shell_cap_exhausted_physical_substeps": sum(shell_caps) if shell_caps else None,
        "shell_cap_observations": len(shell_caps),
        "edge_triangle_count_observations": len(et_values),
        "edge_triangle_recorded_failure_rows": sum(value > 0 for _, value in et_values) if et_values else None,
        "edge_triangle_failure_detected": et_detected,
        "first_edge_triangle_failure_time_s": first_et_time,
        "edge_triangle_log": file_id(log_path) if log_path.is_file() else None,
        "logged_edge_triangle_native_frames": et_frames,
        "logged_edge_triangle_forced_exit_events": len(et_frames) if et_frames is not None else None,
        "edge_triangle_scope": "An ET detection forces the native Newton loop to exit independently of PCG/Newton caps. Finite complete-record admission is not physical admission. Historical receipts/journals remain unchanged; explicit per-step counts or retained native warnings identify rejected histories. Missing ET observations remain unknown; a clean log alone is not a per-step zero counter. Native frame numbers are not mapped across unknown resets.",
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
