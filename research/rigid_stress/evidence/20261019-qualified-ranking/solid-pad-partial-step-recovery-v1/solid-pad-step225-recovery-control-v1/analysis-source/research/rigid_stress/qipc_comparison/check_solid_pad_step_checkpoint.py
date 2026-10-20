"""Exercise actual partial-level receipts, immutable state binding and truncated-write selection."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id, runtime_identity, write_json
from .plate_solid_pad import step_checks
from .solid_pad_checkpoint import select_saved_state, selected_state_file, validate_checkpoint


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--expected-step", type=int, default=300)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    proof = json.loads(args.checkpoint.read_text())
    parent = Path(proof["run"])
    parameters = json.loads((parent / "start.json").read_text())["args"]
    parameters.update(mesh=Path(parameters["mesh"]), output=args.output / "unused", resume_checkpoint=args.checkpoint)
    base = SimpleNamespace(**parameters)
    context = validate_checkpoint(base, step_checks)
    checks = {
        "actual_declared_partial_level_step": proof["checkpoint_kind"] == "step"
        and proof["prefix_steps"] == len(context[1]) == args.expected_step and proof["level"] == 0,
        "minimal_state_selected": selected_state_file(proof)["path"] == str(parent / "checkpoints" / f"step{args.expected_step}.npz"),
        "only_completed_level_frame_inherited": [row["level"] for row in proof["frames"]] == [0],
        "partial_phase_has_no_false_equilibrium_requirement": context[1][-1]["level"] == 1,
    }
    for name in ("state_hash", "state_path", "missing_state_binding", "missing_completed_frame", "shifted_step", "live_state"):
        changed = copy.deepcopy(proof)
        if name == "state_hash":
            changed["state_frame"]["sha256"] = "0" * 64
        elif name == "state_path":
            changed["state_frame"] = changed["frames"][0]["file"]
        elif name == "missing_state_binding":
            changed["receipts"] = [row for row in changed["receipts"] if row != changed["state_frame"]]
        elif name == "missing_completed_frame":
            changed["frames"] = []
        elif name == "shifted_step":
            changed["prefix_steps"] -= 1
        else:
            changed["terminal_scheduler_state"] = "RUNNING"
        path = args.output / (name + ".json")
        write_json(path, changed)
        candidate = copy.deepcopy(base)
        candidate.resume_checkpoint = path
        try:
            validate_checkpoint(candidate, step_checks)
        except (AssertionError, KeyError, ValueError):
            checks[name + "/rejected"] = True
        else:
            raise AssertionError("Altered step receipt admitted: " + name)

    # The only damaged write is private test data. Real terminal states and journals stay untouched.
    fixture = args.output / "selection-fixture"
    (fixture / "frames").mkdir(parents=True)
    (fixture / "checkpoints").mkdir()
    (fixture / "frames/level0.npz").symlink_to(parent / "frames/level0.npz")
    for step in (375, 400):
        (fixture / "checkpoints" / f"step{step}.npz").symlink_to(parent / "checkpoints" / f"step{step}.npz")
    (fixture / "checkpoints/step399.npz").write_bytes(b"PK\x03\x04interrupted compressed state")
    lines = (parent / "steps.jsonl").read_bytes().splitlines(keepends=True)
    for name, selector, expected in (
        ("latest", {}, 375), ("endpoint", {"level": 0}, 200),
        ("truncated", {"step": 399}, None), ("final", {"step": 400}, None),
    ):
        try:
            selected, _, path = select_saved_state(fixture, parameters, lines, **selector)
        except AssertionError:
            assert expected is None
            checks[name + "/not_prepared"] = True
        else:
            assert expected is not None
            checks[name + "/exact_valid_selection"] = selected == expected and path.is_file()
    checks = {key: bool(value) for key, value in checks.items()}
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "passed": all(checks.values()),
        "checks": checks, "receipts": [file_id(args.checkpoint), *proof["receipts"]],
        "scope": "Actual declared terminal partial-level state and original complete-prefix mechanics; changed binding controls; endpoint/latest selection over actual saved states with a private truncated newest file and excluded final state. Full restoration/conformance and multihop lineage require separate actual dynamics controls.",
        "fine_reference_qualified": False, "accepted_reference_frames": [], "whole_goal_complete": False,
    })
    if not all(checks.values()):
        raise SystemExit("Partial-level saved-state controls failed")


if __name__ == "__main__":
    main()
