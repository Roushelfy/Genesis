"""Declare full-grasp field times from recorded command phases before evaluating new keyframe errors."""

import argparse
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def select(rows, phases):
    eligible = [row for row in rows if abs(round(row["time_s"] / 0.01) * 0.01 - row["time_s"]) < 1e-12]
    return [
        {
            "target_phase": phase,
            "time_s": min(eligible, key=lambda row: (abs(row["phase"] - phase), row["time_s"]))["time_s"],
            "actual_phase": min(eligible, key=lambda row: (abs(row["phase"] - phase), row["time_s"]))["phase"],
        }
        for phase in phases
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--phases", type=float, nargs="+", default=[0.22, 0.46, 0.67, 0.73, 0.92])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    capped = []
    for root in args.references:
        result = json.loads((root / "result.json").read_text())
        assert result["completed_requested_history"]
        inputs = json.loads((root / "inputs.json").read_text())
        rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
        first = [row for row in rows if row["reset_epoch"] == 0]
        after_reset = [row for row in rows if row["reset_epoch"] == 1]
        assert after_reset
        capped.append(
            {
                "case": inputs["case"],
                "source": file_id(root / "steps.jsonl"),
                "first_grasp": select(first, args.phases),
                "after_first_reset": select(after_reset, [0.01]),
            }
        )
    motion_receipt = json.loads((args.motion.parent / "result.json").read_text())
    motion_source = Path(motion_receipt["source_rollout"]["path"])
    rows = [json.loads(line) for line in motion_source.read_text().splitlines()]
    rows = [row for row in rows if row["time_s"] <= motion_receipt["duration_s"] + 1e-12]
    prescribed = select(rows, args.phases)
    write_json(
        args.output / "result.json",
        {
            "kind": "predeclared-phase-selected-full-grasp-spatial-keyframes",
            "identity": runtime_identity(),
            "target_phases": args.phases,
            "stages": ["loaded squeeze", "lift", "lateral motion", "weak grip", "release/recontact"],
            "prescribed": {
                "motion": file_id(args.motion),
                "source_schedule": file_id(motion_source),
                "keyframes": prescribed,
            },
            "capped": capped,
            "selection_scope": "Nearest recorded commanded phase on the10ms geometry grid, earliest time on ties. First-grasp phase keys plus one post-reset key per capped case. Prescribed keys use the actual source rollout of the frozen common motion. Selection uses no candidate/reference stress error or peak. Historical dense results are known supplementary evidence; this rule is frozen before the new sparse-grasp field comparisons.",
            "reference_qualified": False,
        },
    )


if __name__ == "__main__":
    main()
