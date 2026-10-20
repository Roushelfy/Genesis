"""Fail closed before field analysis unless each complete parent is independently admitted."""

import argparse
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--expected-steps", type=int, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(args.runs) == len(args.expected_steps) and min(args.expected_steps) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    rows, receipts = [], []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    for run, expected_steps in zip(args.runs, args.expected_steps):
        checks = {}
        try:
            code = run.parent / "returncode.txt"
            checks["parent_terminal_success_receipt"] = code.read_text().strip() == "0"
            receipts.append(file_id(code))
            result, audit = read(run / "result.json"), read(run.parent / "moment/result.json")
            checks["complete_history"] = result["completed"] and result["steps"] == expected_steps
            checks["all_history_gates"] = (
                result["mechanical_history_passed"]
                and bool(result["history_checks"])
                and all(result["history_checks"].values())
            )
            checks["both_complete_equilibrium_windows"] = (
                len(result["windows"]) == 2
                and {w["level"] for w in result["windows"]} == {0, 1}
                and all(
                    w["samples"] >= 51 and w["mechanical_window_passed"] and all(w["checks"].values())
                    for w in result["windows"]
                )
            )
            checks["independent_current_momentum"] = audit["passed"] and audit["history_admitted"]
            checks["audit_binds_actual_result"] = audit["run"] == file_id(run / "result.json")
            fields = {frame["field"]["path"]: frame for frame in audit["frames"]}
            checks["exact_two_endpoint_fields"] = set(fields) == {
                str(run / "frames/level0.npz"),
                str(run / "frames/level1.npz"),
            }
            checks["audited_endpoint_bytes_unchanged"] = all(
                file_id(path) == row["field"] and row["source_transfer_passed"] and row["physical_momentum_passed"]
                for path, row in fields.items()
            )
            receipts.extend(row["field"] for row in fields.values())
            journal_path = run / "steps.jsonl"
            receipts.append(file_id(journal_path))
            journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
            checks["complete_sequential_journal"] = [row["step"] for row in journal] == list(
                range(1, expected_steps + 1)
            )
            checks["original_mechanical_targets_every_step"] = bool(journal) and all(
                max(
                    row["equation_residual_relative"],
                    row["normal_law_relative_error"],
                    row["friction_law_relative_error"],
                    row["current_moment_balance_relative_error"],
                )
                <= 1e-7
                and row["point_velocity_prediction_error_m_per_s"] <= 1e-9
                and row["minimum_omitted_gap_m"] >= -1e-12
                for row in journal
            )
            rows.append(
                {"run": str(run), "expected_steps": expected_steps, "checks": checks, "passed": all(checks.values())}
            )
        except (OSError, KeyError, ValueError) as error:
            rows.append(
                {
                    "run": str(run),
                    "expected_steps": expected_steps,
                    "checks": checks,
                    "passed": False,
                    "error": str(error),
                }
            )
    passed = bool(rows) and all(row["passed"] for row in rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "runs": rows,
            "receipts": receipts,
            "scope": "Explicit preflight for immutable complete parent trajectories; scheduler dependencies alone do not establish physical admission. Rechecks unchanged whole-history equations/laws/current moment, both windows and independently audited endpoint bytes. No spatial qualification or error rank is inferred.",
        },
    )
    if not passed:
        raise SystemExit("Parent history/momentum/field admission failed; dependent scientific analysis must not run")


if __name__ == "__main__":
    main()
