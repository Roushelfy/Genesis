"""Check actual Franka admission with an explicitly supplied curved material asset."""

import argparse
from pathlib import Path

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_franka import capture


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stress-mesh", type=Path, required=True)
    cli.add_argument("--steps", type=int, default=30)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.stress_mesh),
            "steps": args.steps,
            "environments": 4,
            "scope": "Actual moving Franka, original rigid collision surface and controller, supplied quadratic material geometry and native Hertz/friction. Retain every step and stop on the unchanged production error. Loaded rejection does not count as useful throughput, and success alone is not physical stress qualification.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = FrankaEgg(
        4,
        level=1,
        seed=623016,
        varied=True,
        contact_model="hertz",
        output_mode="full",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        fused_rigid_post=False,
        stress_mesh=args.stress_mesh,
    )
    workload.stress = False
    workload.save_conditions(args.output / "conditions.npz")
    rows = []
    for step in range(args.steps):
        try:
            workload.step()
        except gs.GenesisException:
            rows.append(capture(workload, args.output, step))
            write_json(args.output / "result.json", {"passed": False, "rows": rows, "whole_goal_complete": False})
            raise
        rows.append(capture(workload, args.output, step))
        write_json(args.output / "progress.json", {"rows": rows})
    assert any(any(row["loaded"]) for row in rows)
    assert all(all(row["accepted"]) for row in rows)
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
