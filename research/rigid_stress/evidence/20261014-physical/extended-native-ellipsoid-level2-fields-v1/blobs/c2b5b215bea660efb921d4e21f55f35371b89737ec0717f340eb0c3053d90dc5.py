"""Classify saved original-history data without treating a zero wrapper code as completion."""

import json


def history_status(run, producer_code, requested_steps, allow_incomplete=False):
    """Require a complete report and consecutive saved states for completion.

    The caller must separately establish source closure and scheduler status.
    A closed TIMEOUT wrapper can write zero, so its code alone proves neither
    application success nor a complete requested history.
    """
    files = sorted((run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
    assert files
    assert all(path.name == f"step{index:05d}.npz" for index, path in enumerate(files, 1))
    assert len(files) <= requested_steps
    path = run / "results/summary.json"
    report = None
    report_status = "missing"
    if path.exists():
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            report_status = "invalid_json"
        else:
            steps = report.get("steps") if isinstance(report, dict) else None
            if isinstance(steps, list) and len(steps) == len(files):
                report_status = "saved_state_count_matches"
            else:
                report_status = "saved_state_count_mismatch"
    complete = producer_code == 0 and report_status == "saved_state_count_matches" and len(files) == requested_steps
    assert complete or allow_incomplete, (producer_code, report_status, len(files), requested_steps)
    return len(files), complete, report_status
