"""Replay admitted saved pad sources and export actual native reconstruction quadrature."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import genesis as gs

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_recovery import NativePadRecovery
from .native_traction_snapshot import snapshot
from .plates import plate_xml


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--levels", nargs="+", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.run / "start.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    assert result["completed"] and result["mechanical_history_passed"] and all(result["history_checks"].values())
    assert all(w["mechanical_window_passed"] for w in result["windows"])
    params = SimpleNamespace(**start["args"])
    assert params.method in ("dpr", "scr") and params.source_reduction == "none"
    assert params.radius == 0.002 and len(set(args.levels)) == len(args.levels)
    params.assets, params.output = Path(params.assets), args.output
    params.stress_output, params.primary_cost = "full", False
    assert file_id(Path(params.mesh)) == start["mesh"]
    with np.load(params.mesh) as saved:
        outer, faces = saved["outer_vertices"], saved["mid_faces"]
    points = outer[faces].mean(axis=1)
    journal = args.run / "steps.jsonl"
    history = [json.loads(line) for line in journal.read_text().splitlines()]
    for level in args.levels:
        endpoint = next(row for row in reversed(history) if row["level"] == level)
        assert endpoint["stress"]["native_valid"] and endpoint["stress"]["errno"] == 0
        assert endpoint["stress"]["field"] == file_id(args.run / f"frames/stress-level{level}.npz")
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    xml = plate_xml(float(outer[:, 2].min() - params.clearance), float(outer[:, 2].max() + params.clearance))
    observer = NativePadRecovery.allocate(params, outer, faces, points, xml)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "original_start": file_id(args.run / "start.json"),
            "original_result": file_id(args.run / "result.json"),
            "original_journal": file_id(journal),
            "mesh": start["mesh"],
            "levels": args.levels,
            "source_replay_only": True,
        },
    )
    rows = []
    for level in args.levels:
        original = args.run / f"frames/stress-level{level}.npz"
        physical = args.run / f"frames/level{level}.npz"
        with np.load(physical) as saved:
            force, rotation, angular = saved["contact_force_n"], saved["rotation"], saved["angular_velocity_rad_per_s"]
        with np.load(original) as saved:
            np.testing.assert_allclose(saved["original_contact_points_m"], points, rtol=0, atol=1e-15)
            np.testing.assert_allclose(force @ rotation, saved["original_contact_forces_n"], rtol=1e-13, atol=0)
            friction, load = params.friction, saved["load_n"].item()
            expected = {key: saved[key] for key in ("stress_pa", "forces_n", "rhs_n")}
        replay = args.output / f"replayed-level{level}.npz"
        row = observer.sample(force, rotation, angular, friction, load, replay)
        assert row["native_valid"] and row["errno"] == 0 and row["elastic_equation_relative_residual"] <= 1e-7
        with np.load(replay) as saved:
            differences = {
                key: float(np.linalg.norm(saved[key] - value) / np.linalg.norm(value))
                for key, value in expected.items()
            }
        assert max(differences.values()) <= 1e-8, "Replay does not match the original admitted source state"
        arrays, audit = snapshot(observer)
        path = args.output / f"reconstruction-level{level}.npz"
        np.savez_compressed(path, **arrays)
        rows.append(
            {
                "level": level,
                "load_n": load,
                "original": file_id(original),
                "physical_state": file_id(physical),
                "replay": file_id(replay),
                "equivalence": differences,
                "quadrature": file_id(path),
                "capture_checks": audit,
            }
        )
        write_json(args.output / "progress.json", {"cases": rows})
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": True,
            "cases": rows,
            "whole_goal_complete": False,
            "scope": "Read-only diagnostic of native reconstruction quadrature, using actual native sample/force/shape functions. Saved full-source inputs are replayed through unchanged production recovery and must match original admitted fields and loads. Exported samples reproduce complete signed P2 nodal loads and wrench, with unchanged native force/stress arrays. Sample forces are integrated forces, not pressure obtained by dividing nodal forces by an arbitrary dual area. This supplies actual reconstructed boundary-load evidence, not a new mechanical history, primary timing or continuum-accuracy qualification.",
        },
    )


if __name__ == "__main__":
    main()
