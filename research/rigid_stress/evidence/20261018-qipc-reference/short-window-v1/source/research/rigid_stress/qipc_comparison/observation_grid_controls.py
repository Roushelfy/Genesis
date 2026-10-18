"""Reject sparse/prefix/shifted grids while retaining actual complete six- and nine-second comparisons."""

import argparse
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import complete_control_grid, requested_duration_s
from .reference_uncertainty import frame_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--sparse-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, ticks, duration, expected in (
        ("complete-six-seconds", range(1, 601), 6.0, True),
        ("complete-nine-seconds", range(1, 901), 9.0, True),
        ("contiguous-short-prefix", range(1, 117), 6.0, False),
        ("missing-middle-observation", [tick for tick in range(1, 901) if tick != 451], 9.0, False),
        ("missing-terminal-observation", range(1, 900), 9.0, False),
        ("shifted-observation-clock", range(2, 902), 9.0, False),
        ("unknown-requested-duration", range(1, 601), None, False),
    ):
        admitted = complete_control_grid(ticks, duration)
        assert admitted is expected, name
        cases.append({"case": name, "admitted": admitted})
    actual = []
    for path in args.complete_comparisons:
        value = json.loads(path.read_text())
        receipt = json.loads(Path(value["reference_receipt"]["path"]).read_text())
        duration = requested_duration_s(receipt)
        ticks = [round(row["time_s"] / 0.01) for row in value["history"]]
        assert value["complete_common_observation_coverage"] and complete_control_grid(ticks, duration)
        actual.append({"source": file_id(path), "duration_s": duration, "observations": len(ticks), "admitted": True})
    receipt = json.loads((args.sparse_reference / "result.json").read_text())
    ticks = frame_index(args.sparse_reference)
    assert receipt["completed_requested_history"] and not complete_control_grid(ticks, requested_duration_s(receipt))
    write_json(
        args.output / "result.json",
        {
            "kind": "common-observation-grid-admission-controls",
            "identity": runtime_identity(),
            "cases": cases,
            "actual_complete_comparisons": actual,
            "actual_sparse_source": file_id(args.sparse_reference / "result.json"),
            "actual_sparse_observations": len(ticks),
            "sparse_whole_history_admitted": False,
            "all_controls_passed": True,
            "scope": "Completing the requested simulation does not imply a complete10ms field observation grid. All expected ticks from the original requested duration must exist. No phase cut, missing-prefix normalization or sparse-frame whole-history norm is admitted.",
        },
    )


if __name__ == "__main__":
    main()
