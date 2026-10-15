"""Retain accepted reference contact stages without changing their equations or arithmetic."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse


def retain_accepted_contact_step(
    system,
    phase,
    step,
    load_steps,
    displacement,
    previous,
    gradient,
    contact_gradient,
    external_force,
    boundary_dofs,
    coordinate_map,
    tolerance,
    method,
):
    destination = os.environ.get("REFERENCE_ACCEPTED_CONTACT_STATES_ROOT")
    if not destination:
        return
    directory = Path(destination)
    if not hasattr(system, "_retained_contact_history"):
        directory.mkdir(parents=True, exist_ok=False)
        geometry = directory / "geometry.npz"
        np.savez_compressed(
            geometry,
            rest_positions=system.rest,
            surface_triangles=system.faces,
            groups=system.groups,
        )
        system._retained_contact_history = {
            "directory": str(directory),
            "geometry_sha256": hashlib.sha256(geometry.read_bytes()).hexdigest(),
            "accepted_stages": 0,
            "calls": [],
        }
    history = system._retained_contact_history
    assert history["directory"] == str(directory)
    if step == 1:
        call = len(history["calls"])
        free = np.setdiff1d(np.arange(len(displacement)), boundary_dofs)
        mapping = sparse.eye(len(displacement), format="csc")[:, free] if coordinate_map is None else coordinate_map
        path = directory / f"call{call:03d}.map.npz"
        sparse.save_npz(path, mapping)
        history["calls"].append(
            {
                "call": call,
                "phase": phase,
                "method": method,
                "load_steps": load_steps,
                "tolerance_N": tolerance,
                "map_file": path.name,
                "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dhat_m": system.dhat,
                "friction": system.friction,
            }
        )
    call = history["calls"][-1]
    assert call["phase"] == phase and call["method"] == method and call["load_steps"] == load_steps
    ordinal = history["accepted_stages"] + 1
    path = directory / f"stage{ordinal:06d}.npz"
    assert not path.exists()
    np.savez_compressed(
        path,
        displacement=displacement,
        previous_displacement=previous,
        total_gradient_N=gradient,
        contact_force_N=-contact_gradient,
        applied_force_N=np.zeros_like(gradient) if external_force is None else external_force,
        external_force_was_none=external_force is None,
        boundary_dofs=boundary_dofs,
    )
    record = {
        "ordinal": ordinal,
        "call": call["call"],
        "phase": phase,
        "step": step,
        "load_steps": load_steps,
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "accepted_record_index": len(system.accepted_steps) - 1,
        "recorded_complete_residual_N": system.accepted_steps[-1].complete_free_residual_n,
        "is_physical_time_sample": False,
    }
    with (directory / "stages.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    history["accepted_stages"] = ordinal
    manifest = {
        "scope": __doc__,
        **history,
        "stages_file": "stages.jsonl",
        "stages_sha256": hashlib.sha256((directory / "stages.jsonl").read_bytes()).hexdigest(),
        "stage_coordinate": "Original quasi-static load/initialization stages; not added physical timesteps.",
        "complete_producer_success_not_inferred": True,
        "goal_completion": False,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
