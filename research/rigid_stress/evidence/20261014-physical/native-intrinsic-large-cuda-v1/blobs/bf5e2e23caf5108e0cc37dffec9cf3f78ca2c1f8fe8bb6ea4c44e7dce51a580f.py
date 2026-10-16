"""Qualify direct rigid force assembly against every original audited saved state without changing history."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .direct_rigid_contact import DirectOriginalRigidContact
from .original_franka_ground import build_original
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.run, args.audit]:
        closed_source(run)
    assert int((args.audit / "returncode.txt").read_text()) == 0
    audit = json.loads((args.audit / "results/summary.json").read_text())
    assert Path(audit["run"]).resolve() == args.run.resolve()
    assert all(
        max(r["body_force_equivalent_residual_N"], r["fresh_actual_robot_effort_residual"]) <= 1e-9
        for r in audit["rows"]
    )
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, frame, capture, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    system = DirectOriginalRigidContact(static, frame, capture, args.mjcf, config["dt"])
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    substeps = round(protocol["macro_dt_s"] / config["dt"])
    with np.load(args.run / "results/initial_state.npz") as saved:
        previous, q, velocity, bv = [
            saved[k].copy() for k in ["full_displacement", "joint_position", "joint_velocity", "body_velocity"]
        ]
    args.output.mkdir(parents=True, exist_ok=False)
    rows, inputs = [], {}
    for step, audited in enumerate(audit["rows"], 1):
        assert audited["step"] == step
        path = args.run / f"results/step{step:05d}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert audit["input_sha256"][str(path)] == digest
        inputs[str(path)] = digest
        with np.load(path) as saved:
            pose, new_velocity, new_full, new_q, new_bv, force = [
                saved[k].copy()
                for k in [
                    "body_pose",
                    "joint_velocity",
                    "displacement",
                    "joint_position",
                    "body_velocity",
                    "contact_force_N",
                ]
            ]
        control = controls[(step - 1) // substeps]
        system.target = np.array(control["target"])
        system.caps[7:] = control["finger_cap_N"]
        system.begin_dynamic_step(q, velocity, previous[: system.n] + config["dt"] * bv)
        result = system.evaluate_rigid(pose, new_velocity, previous, False)
        np.testing.assert_array_equal(result[2], new_full)
        np.testing.assert_array_equal(result[3], -force)
        body, robot = [float(np.linalg.norm(r)) for r in result[:2]]
        assert max(body, robot) <= 1e-9
        rows.append(
            {
                "step": step,
                "body_force_equivalent_residual_N": body,
                "actual_robot_effort_residual": robot,
                "full_displacement_and_contact_forces_exact": True,
            }
        )
        previous, q, velocity, bv = new_full, new_q, new_velocity, new_bv
    assert len(rows) == audit["states_checked"]
    assert inputs == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    report = {
        "scope": __doc__,
        "run": str(args.run),
        "independent_audit": str(args.audit),
        "states_checked": len(rows),
        "all_original_states_and_forces_preserved": True,
        "original_physical_gates_passed": True,
        "rows": rows,
        "input_sha256": inputs,
        "physical_states_added": 0,
        "goal_completion": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "states": len(rows),
                "maximum_body_residual_N": max(r["body_force_equivalent_residual_N"] for r in rows),
                "maximum_actual_robot_effort_residual": max(r["actual_robot_effort_residual"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
