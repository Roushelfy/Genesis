"""Check that missing, damaged and contradictory worker evidence is never admitted as a complete history."""

import argparse
import json
from pathlib import Path

from .artifacts import runtime_identity, write_json
from .history_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [
        {"time_s": 0.001, "all_fields_finite": True, "newton": 1, "pcg_max": 4},
        {"time_s": 0.002, "all_fields_finite": True, "newton": 2, "pcg_max": 5},
    ]
    lines = [(json.dumps(row) + "\n").encode() for row in rows]
    config = {"max_newton": 8, "max_pcg": 16}
    receipt = {"completed_requested_history": True, "executed_steps": 2, "config": config}
    records = []

    def fixture(name, journal, result=receipt, group_rc=0, task_rc=None, log=None):
        group = args.output / name
        folder = group / "history"
        folder.mkdir(parents=True)
        write_json(folder / "inputs.json", {"config": config})
        if journal is not None:
            (folder / "steps.jsonl").write_bytes(journal)
        if result is not None:
            if isinstance(result, str):
                (folder / "result.json").write_text(result)
            else:
                write_json(folder / "result.json", result)
        if group_rc is not None:
            (group / "returncode.txt").write_text(str(group_rc) + "\n")
        if task_rc is not None:
            (group / "history.returncode.txt").write_text(str(task_rc) + "\n")
        if log is not None:
            (group / "history.log").write_text(log)
        value = audit(folder)
        if journal is not None:
            assert (folder / "steps.jsonl").read_bytes() == journal, "An audit must preserve damaged evidence"
        records.append({"case": name, "audit": value})
        return value

    value = fixture("complete-success", b"".join(lines))
    assert value["complete_history_established"] and value["recorded_rows"] == 2
    assert value["nonfinite_recorded_rows"] == value["newton_cap_recorded_rows"] == value["pcg_cap_recorded_rows"] == 0
    assert value["newton_cap_observations"] == value["pcg_cap_observations"] == 2

    value = fixture("missing-result", b"".join(lines), result=None, group_rc=None)
    assert value["receipt"] is None and value["receipt_read_error"]
    assert value["completed_history_reported"] is None and not value["complete_history_established"]
    assert value["recorded_rows"] == 2 and value["pcg_cap_observations"] == 2

    damaged = b'{"time_s":'
    value = fixture("truncated-tail", lines[0] + damaged)
    assert not value["complete_history_established"] and value["recorded_rows"] == 1
    assert value["journal_read"]["first_unreadable_row"]["line"] == 2
    assert value["journal_read"]["first_unreadable_row"]["byte_offset"] == len(lines[0])
    assert value["journal_read"]["unparsed_tail_bytes"] == len(damaged)

    value = fixture("middle-corruption", lines[0] + damaged + b"\n" + lines[1])
    assert value["recorded_rows"] == 1 and not value["complete_history_established"]
    assert value["journal_read"]["unparsed_tail_bytes"] == len(damaged) + 1 + len(lines[1])

    value = fixture("corrupt-result", b"".join(lines), result="{")
    assert value["receipt"] is None and value["receipt_identity"] and value["receipt_read_error"]
    assert not value["complete_history_established"]

    value = fixture("failed-terminal", b"".join(lines), group_rc=1)
    assert value["completed_history_reported"] and not value["complete_history_established"]

    value = fixture("missing-terminal", b"".join(lines), group_rc=None)
    assert value["completed_history_reported"] and not value["complete_history_established"]

    value = fixture("task-failure-precedence", b"".join(lines), task_rc=1)
    assert value["task_returncode"] == 1 and not value["complete_history_established"]

    value = fixture("missing-journal", None)
    assert value["journal_identity"] is None and value["journal_read"]["error"]
    assert not value["complete_history_established"]

    value = fixture("unobserved-flags", b'{"time_s": 0.001}\n', result=None, group_rc=None)
    assert value["nonfinite_recorded_rows"] is None and value["finite_flag_observations"] == 0
    assert value["newton_cap_recorded_rows"] is None and value["pcg_cap_recorded_rows"] is None

    value = fixture(
        "missing-finite-evidence",
        b'{"time_s": 0.001}\n',
        result={"completed_requested_history": True, "executed_steps": 1},
    )
    assert value["completed_history_reported"] and not value["complete_history_established"]

    changed = {"completed_requested_history": False, "executed_steps": 1, "config": {"max_newton": 4, "max_pcg": 8}}
    value = fixture("actual-configured-caps", b'{"time_s": 0.001, "newton": 4, "pcg_max": 8}\n', result=changed)
    assert value["newton_cap_recorded_rows"] == value["pcg_cap_recorded_rows"] == 1
    assert not value["complete_history_established"]

    et_receipt = {**receipt, "kind": "QIPC-prescribed-link-motion-candidate", "config": {**config, "dt": 0.001}}
    value = fixture(
        "finite-with-logged-ET-forced-exit",
        b"".join(lines),
        result=et_receipt,
        log="[QIPC][warning] ET intersection detected at frame 0, forcing Newton exit\n",
    )
    assert value["complete_history_established"] and value["edge_triangle_failure_detected"]
    assert value["logged_edge_triangle_forced_exit_events"] == 1
    assert value["first_edge_triangle_failure_time_s"] == 0.001
    for count, expected in ((0, False), (2, True)):
        et_rows = [{**row, "edge_triangle_intersection_count": count} for row in rows]
        journal = "".join(json.dumps(row) + "\n" for row in et_rows).encode()
        value = fixture(f"recorded-ET-count-{count}", journal)
        assert value["edge_triangle_failure_detected"] is expected
        assert value["edge_triangle_count_observations"] == 2

    write_json(
        args.output / "result.json",
        {
            "kind": "interrupted-history-evidence-admission-controls",
            "identity": runtime_identity(),
            "cases": records,
            "all_controls_passed": True,
            "scope": "Synthetic metadata corruption/completion controls only. No physical trajectory, stress accuracy, numerical solver convergence or reference qualification is inferred.",
        },
    )


if __name__ == "__main__":
    main()
