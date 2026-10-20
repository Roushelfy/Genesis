"""Reject a copied inherited endpoint before model setup; retain advancing/root receipts."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id, runtime_identity, write_json
from .plate_solid_pad import step_checks
from .solid_pad_checkpoint import inherited_prefix_steps, select_saved_state, validate_checkpoint


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--preparation-refusal", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, receipts = {}, []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(Path(path).read_text())

    def parameters(path):
        proof = read(path)
        start = read(Path(proof["start"]["path"]))
        values = copy.deepcopy(start["args"])
        values.update(mesh=Path(values["mesh"]), output=args.output / "unused", resume_checkpoint=path)
        return proof, start, SimpleNamespace(**values)

    root = args.data_root
    source = root / "solid-pad-step225-recovery-control-v1/analysis-source/research/rigid_stress/qipc_comparison/solid_pad_checkpoint.py"
    receipts.append(file_id(source))
    spec = importlib.util.spec_from_file_location("research.rigid_stress.qipc_comparison._legacy_checkpoint", source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    proof, start, base = parameters(root / "solid-pad-step225-preparation-v1/result.json")
    parent = Path(proof["run"])
    lines = (parent / "steps.jsonl").read_bytes().splitlines(keepends=True)
    receipts.append(file_id(parent / "steps.jsonl"))
    counterexample = copy.deepcopy(proof)
    counterexample.update(checkpoint_kind="endpoint", state_frame=proof["frames"][0]["file"], prefix_steps=200,
                          prefix_sha256=hashlib.sha256(b"".join(lines[:200])).hexdigest())
    path = args.output / "legacy-valid-copied-endpoint.json"
    write_json(path, counterexample)
    receipts.append(file_id(path))
    candidate = copy.deepcopy(base)
    candidate.resume_checkpoint = path
    old = legacy.validate_checkpoint(candidate, step_checks)
    checks["actual_old_validator_accepts_copied200_endpoint"] = len(old[1]) == 200
    checks["same200_steps_already_inherited_by_parent"] = start["resume_provenance"]["prefix_steps"] == 200
    try:
        validate_checkpoint(candidate, step_checks)
    except AssertionError as failure:
        checks["new_validator_rejects_before_model_setup"] = "does not advance" in str(failure) and "ancestor" in str(failure)
    else:
        raise AssertionError("A copied inherited endpoint was admitted")
    cases = [
        ("original_coarse200", "solid-pad-resume-preparation-v1", 200),
        ("advancing225", "solid-pad-step225-preparation-v1", 225),
        ("advancing300", "solid-pad-step300-preparation-v1", 300),
        ("fine_final200", "solid-pad-stvk-canonical125-final-resume-preparation-v1", 200),
        ("fine_contact200", "solid-pad-stvk-canonical125-contact-resume-preparation-v1", 200),
        ("fine_halfdt400", "solid-pad-stvk-canonical125-halfdt-resume-preparation-v1", 400),
    ]
    for name, directory, expected in cases:
        checkpoint = root / directory / "result.json"
        actual, _, settings = parameters(checkpoint)
        context = validate_checkpoint(settings, step_checks)
        checks[name + "/original_prefix_admitted"] = len(context[1]) == actual["prefix_steps"] == expected
    fixture = args.output / "no_new_state"
    (fixture / "frames").mkdir(parents=True)
    (fixture / "checkpoints").mkdir()
    (fixture / "frames/level0.npz").symlink_to(parent / "frames/level0.npz")
    old_selected = legacy.select_saved_state(fixture, start["args"], lines)
    checks["old_latest_selector_returns_nonadvancing_endpoint"] = old_selected[0] == 200
    for name, target, selector, minimum in (
        ("latest_without_new_state", fixture, {}, 200),
        ("explicit_inherited_level", parent, {"level": 0}, 200),
        ("explicit_inherited_step", parent, {"step": 225}, 225),
    ):
        try:
            select_saved_state(target, start["args"], lines, minimum_steps=minimum, **selector)
        except AssertionError as failure:
            checks[name + "/early_rejection_with_ancestor_action"] = "ancestor" in str(failure)
        else:
            raise AssertionError("Nonadvancing state selected: " + name)
    for name, selector, minimum, expected in (
        ("latest_advancing", {}, 200, 375), ("explicit225", {"step": 225}, 200, 225),
        ("explicit300", {"step": 300}, 200, 300), ("root_endpoint", {"level": 0}, 0, 200),
    ):
        selected = select_saved_state(parent, start["args"], lines, minimum_steps=minimum, **selector)
        checks[name + "/exact_selection"] = selected[0] == expected
    refusal = read(args.preparation_refusal)
    checks["actual_terminal_preparation_refused_copied_endpoint"] = (
        refusal["passed"] is False and refusal["terminal_scheduler_state"] == "FAILED"
        and "does not advance" in refusal["error"] and "ancestor" in refusal["error"]
    )
    checks["root_history_has_no_inherited_prefix"] = inherited_prefix_steps({}) == 0
    checks["continued_history_uses_recorded_inherited_prefix"] = inherited_prefix_steps(start) == 200
    checks = {name: bool(value) for name, value in checks.items()}
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False), "passed": all(checks.values()),
        "checks": checks, "receipts": receipts,
        "scope": "Actual old-validator/selector acceptance versus new early refusal for a copied inherited200-step endpoint, six original/advancing/fine receipts and unchanged valid selectors. A fresh preparation against the actual terminal coarse parent is refused before model setup. No FEM model, new dynamics or numerical conformance recomputation is performed. Select the existing terminal ancestor receipt if an intermediate attempt contributes no new saved state.",
        "fine_reference_qualified": False, "accepted_reference_frames": [], "whole_goal_complete": False,
    })
    if not all(checks.values()):
        raise SystemExit("Forward-prefix metadata controls failed")


if __name__ == "__main__":
    main()
