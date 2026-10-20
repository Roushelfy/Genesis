"""Bind a continued reference to its terminal parent and exact saved physical prefix."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_id
from .solid_pad_checkpoint import TERMINAL, selected_state_file, validate_checkpoint


def audit_prefix_chain(run, original_run, start, read, verify, require, available_steps, visited=None):
    """Follow exact inherited prefixes, including terminal parents killed before result.json."""
    from .plate_solid_pad import step_checks

    run, original_run = Path(run), Path(original_run)
    visited = set() if visited is None else visited
    require("acyclic_saved_prefix_ancestry", run.resolve() not in visited)
    visited.add(run.resolve())
    provenance = start.get("resume_provenance")
    require("explicit_saved_endpoint_provenance", bool(provenance))
    checkpoint = Path(provenance["receipt"]["path"])
    verify(provenance["receipt"])
    proof = read(checkpoint)
    parent = Path(proof["run"])
    require("acyclic_terminal_parent", parent.resolve() not in visited)
    submission = read(parent.parent.parent / "submission.json")
    job = str(submission["job"]).split(";")[0] + "_" + parent.parent.name.removeprefix("group")
    scheduler = [line.split("|") for line in proof["scheduler_output"].splitlines() if line.split("|")[0] == job]
    require(
        "bound_terminal_scheduler_receipt",
        proof["job"] == job
        and len(scheduler) == 1
        and scheduler[0][1].split()[0].removesuffix("+") == proof["terminal_scheduler_state"]
        and proof["terminal_scheduler_state"] in TERMINAL,
    )
    parameters = dict(start["args"])
    parameters.update(mesh=Path(parameters["mesh"]), output=run, resume_checkpoint=checkpoint)
    validated, prefix_rows, prefix = validate_checkpoint(SimpleNamespace(**parameters), step_checks)
    require("unchanged_complete_checkpoint_validation", validated == proof)
    for receipt in proof["receipts"]:
        verify(receipt)
    original = read(parent / "start.json")
    require("same_actual_mesh_and_mass", start["mesh"] == original["mesh"] and start["mass_kg"] == original["mass_kg"])
    require("actual_new_output_path", start["args"]["output"] == str(run))
    require(
        "exact_recorded_resume_provenance",
        provenance["terminal_parent_job"] == proof["job"]
        and provenance["parent_state"] == proof["terminal_scheduler_state"]
        and provenance["prefix_steps"] == proof["prefix_steps"] == len(prefix_rows)
        and provenance["prefix_sha256"] == proof["prefix_sha256"]
        and provenance["selected_frame"] == selected_state_file(proof),
    )
    journal = run / "steps.jsonl"
    verify(file_id(journal))
    lines = journal.read_bytes().splitlines(keepends=True)
    actual_prefix = b"".join(lines[: len(prefix_rows)])
    require(
        "copied_prefix_bytes_unchanged",
        actual_prefix == prefix and hashlib.sha256(actual_prefix).hexdigest() == proof["prefix_sha256"],
    )
    for row in proof["frames"]:
        copied = file_id(run / "frames" / f"level{row['level']}.npz")
        verify(copied)
        require(
            f"copied_level{row['level']}_frame_bytes_unchanged",
            all(copied[key] == row["file"][key] for key in ("bytes", "sha256")),
        )
    require(
        "strictly_advancing_saved_prefix",
        len(lines) >= available_steps > len(prefix_rows),
    )
    ancestry = []
    if parent != original_run:
        ancestry = audit_prefix_chain(
            parent, original_run, original, read, verify,
            lambda name, passed: require("ancestor/" + name, passed),
            len(prefix_rows), visited,
        )
    else:
        require("correct_original_parent", parent == original_run)
    return [{
        "continued_run": str(run), "terminal_parent": str(parent),
        "terminal_parent_job": job, "terminal_parent_state": proof["terminal_scheduler_state"],
        "checkpoint": provenance["receipt"], "inherited_steps": len(prefix_rows),
        "available_prefix_steps": available_steps, "prefix_sha256": proof["prefix_sha256"],
        "numerical_modules_bound": len(proof["physics_modules"]),
    }, *ancestry]


def audit_resume_lineage(run, original_run, start, read, verify, require):
    """Bind a completed continuation to its original anchor through every saved prefix."""
    run, original_run = Path(run), Path(original_run)
    result = read(run / "result.json")
    lines = (run / "steps.jsonl").read_bytes().splitlines(keepends=True)
    ancestry = audit_prefix_chain(run, original_run, start, read, verify, require, len(lines))
    inherited = ancestry[0]["inherited_steps"]
    provenance = start["resume_provenance"]
    require(
        "complete_history_and_separate_segment_accounting",
        result["completed"]
        and result["steps"] == len(lines)
        and result["inherited_steps"] == inherited
        and result["newly_executed_steps"] == len(lines) - inherited > 0
        and result["elapsed_seconds_scope"] == "newly executed segment, excluding inherited history and setup"
        and provenance["new_segment_elapsed_is_whole_trajectory_cost"] is False,
    )
    return {
        "original_parent": str(original_run),
        "continued_run": str(run),
        "checkpoint": provenance["receipt"],
        "terminal_parent_job": ancestry[0]["terminal_parent_job"],
        "terminal_parent_state": ancestry[0]["terminal_parent_state"],
        "inherited_steps": result["inherited_steps"],
        "newly_executed_steps": result["newly_executed_steps"],
        "prefix_sha256": ancestry[0]["prefix_sha256"],
        "numerical_modules_bound": ancestry[0]["numerical_modules_bound"],
        "elapsed_seconds_scope": result["elapsed_seconds_scope"],
        "ancestry": ancestry,
    }
