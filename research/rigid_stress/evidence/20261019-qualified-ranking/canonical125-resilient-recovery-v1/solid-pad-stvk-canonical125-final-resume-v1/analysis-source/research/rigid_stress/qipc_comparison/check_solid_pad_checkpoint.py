"""Check actual terminal-prefix provenance and reject altered checkpoint contracts."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id, runtime_identity, write_json
from .plate_solid_pad import step_checks
from .solid_pad_checkpoint import validate_checkpoint


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--live-rejection", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    proof = json.loads(args.checkpoint.read_text())
    saved = json.loads(Path(proof["start"]["path"]).read_text())
    parameters = copy.deepcopy(saved["args"])
    parameters.update(mesh=Path(parameters["mesh"]), output=args.output / "unused", resume_checkpoint=args.checkpoint)
    base = SimpleNamespace(**parameters)
    context = validate_checkpoint(base, step_checks)
    checks = {
        "actual_terminal_endpoint_receipt": proof["passed"],
        "exact_complete200_step_prefix": len(context[1]) == proof["prefix_steps"] == 200,
        "all_original_prefix_mechanical_gates": all(all(step_checks(row).values()) for row in context[1]),
    }
    for key, value in (("dt", 0.002), ("gravity_x", 0.0), ("candidate_radius", 0.004), ("loads", [0.1, 0.3])):
        changed = copy.deepcopy(base)
        setattr(changed, key, value)
        try:
            validate_checkpoint(changed, step_checks)
        except (AssertionError, ValueError):
            checks["changed_" + key + "_rejected"] = True
        else:
            raise AssertionError("Changed checkpoint argument admitted: " + key)
    for name in (
        "frame_hash",
        "prefix_hash",
        "prefix_count",
        "numerical_implementation",
        "no_remaining_steps",
        "missing_module",
        "aliased_module",
    ):
        changed = copy.deepcopy(proof)
        if name == "frame_hash":
            changed["frames"][0]["file"]["sha256"] = "0" * 64
        elif name == "prefix_hash":
            changed["prefix_sha256"] = "0" * 64
        elif name == "prefix_count":
            changed["prefix_steps"] -= 1
        elif name == "numerical_implementation":
            changed["physics_modules"][0]["semantic_sha256"] = "0" * 64
        elif name == "no_remaining_steps":
            changed["level"] = len(base.loads) - 1
        elif name == "missing_module":
            changed["physics_modules"].pop()
        else:
            changed["physics_modules"][0]["file"] = changed["physics_modules"][1]["file"]
        path = args.output / ("changed-" + name + ".json")
        write_json(path, changed)
        candidate = copy.deepcopy(base)
        candidate.resume_checkpoint = path
        try:
            validate_checkpoint(candidate, step_checks)
        except (AssertionError, ValueError):
            checks[name + "_rejected"] = True
        else:
            raise AssertionError("Altered checkpoint receipt admitted: " + name)
    rejected = json.loads(args.live_rejection.read_text())
    assert not rejected["passed"] and rejected["terminal_scheduler_state"] == "RUNNING"
    candidate = copy.deepcopy(base)
    candidate.resume_checkpoint = args.live_rejection
    try:
        validate_checkpoint(candidate, step_checks)
    except AssertionError:
        checks["actual_live_parent_receipt_rejected"] = True
    else:
        raise AssertionError("Live-parent checkpoint admitted")
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": all(checks.values()),
            "checks": checks,
            "receipts": [file_id(args.checkpoint), file_id(args.live_rejection)],
            "scope": "Actual terminal coarse200-step prefix and changed configuration/receipt controls only. Full resumed dynamics and endpoint field equivalence require separate execution. No live fine run is resumed, cancelled or admitted here.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
