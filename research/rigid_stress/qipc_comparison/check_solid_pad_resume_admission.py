"""Check actual continued-parent lineage and reject misbound final-reference metadata."""

import argparse
import copy
import json
import sys
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .qualify_friction_reference import main as qualify
from .solid_pad_resume_admission import audit_resume_lineage


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.data_root / "solid-pad-resume-coarse-control-v1/group0/reference"
    parent = args.data_root / "solid-pad-stvk-physical-force-coarse-v1/group0/reference"
    start = json.loads((run / "start.json").read_text())
    result = json.loads((run / "result.json").read_text())
    checks, receipts = {}, {}

    def read(path):
        receipts[str(path)] = file_id(path)
        return json.loads(Path(path).read_text())

    def verify(receipt):
        actual = file_id(receipt["path"])
        assert actual == receipt
        receipts[receipt["path"]] = actual

    def require(name, passed):
        assert passed, name

    positive = {}

    def positive_check(name, passed):
        positive[name] = bool(passed)
        require(name, passed)

    lineage = audit_resume_lineage(run, parent, read(run / "start.json"), read, verify, positive_check)
    checks.update({"actual_coarse/" + key: value for key, value in positive.items()})
    mutations = (
        "missing_provenance", "wrong_original_parent", "changed_parent_job", "changed_selected_frame",
        "changed_prefix_count", "whole_trajectory_cost_claim", "changed_inherited_count", "changed_new_count",
        "changed_mass", "changed_output",
    )
    for name in mutations:
        changed, outcome, origin = copy.deepcopy(start), copy.deepcopy(result), parent
        if name == "missing_provenance":
            changed["resume_provenance"] = None
        elif name == "wrong_original_parent":
            origin = args.data_root / "unrelated-parent"
        elif name == "changed_parent_job":
            changed["resume_provenance"]["terminal_parent_job"] = "unrelated-job"
        elif name == "changed_selected_frame":
            changed["resume_provenance"]["selected_frame"]["sha256"] = "0" * 64
        elif name == "changed_prefix_count":
            changed["resume_provenance"]["prefix_steps"] -= 1
        elif name == "whole_trajectory_cost_claim":
            changed["resume_provenance"]["new_segment_elapsed_is_whole_trajectory_cost"] = True
        elif name == "changed_inherited_count":
            outcome["inherited_steps"] -= 1
        elif name == "changed_new_count":
            outcome["newly_executed_steps"] += 1
        elif name == "changed_mass":
            changed["mass_kg"] *= 1.1
        else:
            changed["args"]["output"] = str(args.output / "unrelated-run")

        def altered_read(path):
            return outcome if Path(path) == run / "result.json" else read(path)

        try:
            audit_resume_lineage(run, origin, changed, altered_read, verify, require)
        except (AssertionError, KeyError, ValueError):
            checks[name + "/rejected"] = True
        else:
            raise AssertionError("Altered continuation lineage admitted: " + name)

    original_argv = sys.argv
    plans = {
        "unknown_parent": {"unknown": "relative/run"},
        "absolute_path": {"final125": "/absolute/run"},
        "parent_escape": {"final125": "../outside/run"},
        "empty_plan": {},
        "non_path_value": {"final125": True},
        "valid_plan_incomplete_parents": {"final125": "solid-pad-stvk-canonical125-final-resume-v2/group0/reference"},
    }
    try:
        for name, plan in plans.items():
            path = args.output / (name + ".json")
            write_json(path, plan)
            output = args.output / name
            sys.argv = ["qualify_friction_reference", "--data-root", str(args.data_root), "--parent-admission",
                        str(args.data_root / "surface2-pair-parent-admission-v1/group0/admission/result.json"),
                        "--parent-overrides", str(path), "--output", str(output)]
            try:
                qualify()
            except SystemExit as failure:
                require(name + "/nonzero_exit", failure.code is not None and failure.code != 0)
            else:
                raise AssertionError("Incomplete or invalid parent plan admitted: " + name)
            rejected = read(output / "result.json")
            require(name + "/no_admitted_frames", not rejected["passed"] and not rejected["fine_reference_qualified"]
                    and rejected["accepted_reference_frames"] == [])
            key = "all_seven_required_parent_histories" if name == "valid_plan_incomplete_parents" else "explicit_relative_continuation_parent_map"
            checks[name + "/correct_gate_rejected"] = rejected["checks"][key] is False
    finally:
        sys.argv = original_argv
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "passed": all(checks.values()),
        "checks": checks, "actual_coarse_lineage": lineage, "receipts": list(receipts.values()),
        "scope": "Actual admitted coarse continuation lineage and ten changed-provenance/accounting controls, five invalid parent maps and one valid map with incomplete parent admission. Original arrays/journals remain unchanged. Fine physical history/resolution and final ranks remain pending.",
        "fine_reference_qualified": False, "accepted_reference_frames": [], "whole_goal_complete": False,
    })
    if not all(checks.values()):
        raise SystemExit("Continued-parent lineage controls failed")


if __name__ == "__main__":
    main()
