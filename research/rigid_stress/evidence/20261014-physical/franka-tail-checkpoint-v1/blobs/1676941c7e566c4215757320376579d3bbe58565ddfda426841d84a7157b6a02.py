"""Audit full-arm identical-input contact-law changes at dense common times."""

import argparse
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    frozen = None
    conditions = None
    geometry = None
    controls = None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        report = json.loads((run / "results/summary.json").read_text())
        config = {k: v for k, v in report["configuration"].items() if k not in ("output", "contact_timeconst")}
        params = {
            k: report[k]
            for k in (
                "robot_joint_kp",
                "robot_joint_kv",
                "robot_joint_armature",
                "robot_joint_passive_damping",
                "stationary_actuation",
            )
        }
        with np.load(run / "results/conditions.npz") as saved:
            source = {k: saved[k].copy() for k in saved.files}
        if frozen is None:
            frozen, conditions, geometry, controls = config, source, report["geometry"], params
        assert config == frozen and params == controls and report["geometry"] == geometry
        assert set(source) == set(conditions)
        for key in source:
            np.testing.assert_array_equal(source[key], conditions[key])
        tau = report["configuration"]["contact_timeconst"]
        rows = [r for r in report["captures"] if r["mode"] == "stationary_lift_hold" and r["tick"] >= 350]
        assert len(rows) == 320 and len({(r["tick"], r["substep"]) for r in rows}) == 320
        assert all(abs(c["mixed_contact_sol_params"][0] - tau) < 1e-14 for r in rows for c in r["contacts"])
        cases.append((tau, run, rows))
    cases.sort(reverse=True)
    a, b = cases
    assert [(r["tick"], r["substep"]) for r in a[2]] == [(r["tick"], r["substep"]) for r in b[2]]
    assert all(
        [c["opposite_geom"] for c in x["contacts"]] == [c["opposite_geom"] for c in y["contacts"]]
        for x, y in zip(a[2], b[2], strict=True)
    )
    old = np.array([[c["normal_force_N"] for c in r["contacts"]] for r in a[2]])
    new = np.array([[c["normal_force_N"] for c in r["contacts"]] for r in b[2]])
    assert old.shape == new.shape == (320, 2)
    records = []
    for tau, run, rows in cases:
        last = rows[-1]
        records.append(
            {
                "run": str(run),
                "tau_s": tau,
                "frames": len(rows),
                "last_normal_forces_N": [c["normal_force_N"] for c in last["contacts"]],
                "last_penetrations_m": [c["penetration_m"] for c in last["contacts"]],
                "last_body_angular_speed_rad_s": float(np.linalg.norm(last["egg_angular_velocity_rad_s"])),
                "last_hand_speed_m_s": float(np.linalg.norm(last["hand_velocity_m_s"])),
                "last_hand_angular_speed_rad_s": float(np.linalg.norm(last["hand_angular_velocity_rad_s"])),
                "last_robot_joint_positions": last["robot_joint_position"],
                "last_robot_joint_control_force_readout": last["robot_joint_control_force_readout"],
            }
        )
    output = {
        "scope": __doc__,
        "cases": records,
        "fixed_controls": controls,
        "fixed_configuration": frozen,
        "common_physical_samples": 320,
        "normal_force_l2_relative_change": float(np.linalg.norm(new - old) / np.linalg.norm(new)),
        "goal_completion": False,
        "limitations": "These are actual full seven-arm-joint rigid-application captures, with identical saved conditions and controller/mass inputs. Per-geom contact regularization is the changed axis. Two contact-law settings do not establish a convergence tail. Dense hand/joint/force/geometry data enables corresponding reference construction; it is not a solved deformable full-arm comparison or physical accuracy/cost acceptance. Hand motion and angular body motion remain explicit.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
