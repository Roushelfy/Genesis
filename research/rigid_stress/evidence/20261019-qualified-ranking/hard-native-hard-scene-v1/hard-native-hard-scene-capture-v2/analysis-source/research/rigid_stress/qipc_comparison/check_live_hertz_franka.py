"""Measure native Hertz admission in the actual moving Franka/hollow-egg scene."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_normal import snapshot
from .check_native_hertz_recovery import array


def capture(workload, output, step):
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    data = snapshot(entry)
    data.update(
        position_m=tensor_to_array(workload.egg.get_pos()),
        quaternion=tensor_to_array(workload.egg.get_quat()),
        friction_line=array(entry.hertz.friction.state.line),
        friction_status=array(entry.hertz.friction.state.status),
        binding_lost=array(entry.hertz.binding.is_lost),
        material_surface_nodes=array(entry.model.info.surface_nodes),
        material_face_normal=array(entry.surface.info.face_normal),
    )
    path = output / f"frame{step:04d}.npz"
    np.savez_compressed(path, **data)
    loaded = np.any(data["contact_valid"] & (np.linalg.norm(data["contact_force_n"], axis=2) > 0), axis=1)
    return {
        "step": step,
        "loaded": loaded.tolist(),
        "accepted": data["step_valid"].tolist(),
        "input_status": data["input_status"].tolist(),
        "recovery_status": data["recovery_status"].tolist(),
        "contact_status": data["contact_status"].tolist(),
        "source_status": data["source_status"].tolist(),
        "region_status": data["region_status"].tolist(),
        "law_status": data["law_status"].tolist(),
        "friction_status": data["friction_status"].tolist(),
        "fields": file_id(path),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--steps", type=int, default=200)
    cli.add_argument("--envs", type=int, default=4)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "steps": args.steps,
            "environments": args.envs,
            "scope": "Actual native Hertz/friction admission in the level1 rigid Franka/hollow egg, including the ground contact. Stop on the original production error and retain all available stage states. Rejected steps provide diagnostics only. No accepted throughput or physical error claim follows from execution alone.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        level=1,
        seed=623016,
        varied=True,
        contact_model="hertz",
        output_mode="full",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        fused_rigid_post=False,
    )
    # The Hertz law determines support. The shared controller otherwise updates
    # the manual-radius input used by its original pressure benchmark.
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
