"""Audit actual nine-joint inertia/kinematics and contact-law tails without fixed-hand substitution."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .audit_franka_reference import rotation
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases, frozen, conditions = [], None, None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        report = json.loads((run / "results/summary.json").read_text())
        config = report["configuration"]
        assert config["capture_articulated_operators"] and config["substeps"] == 32
        fixed = {k: v for k, v in config.items() if k not in ("contact_timeconst", "output")}
        with np.load(run / "results/conditions.npz") as saved:
            current_conditions = {key: saved[key].copy() for key in saved.files}
        if frozen is None:
            frozen, conditions = fixed, current_conditions
        assert fixed == frozen and set(current_conditions) == set(conditions)
        for key, value in current_conditions.items():
            np.testing.assert_array_equal(value, conditions[key])
        rows = [r for r in report["captures"] if r["mode"] == "stationary_lift_hold" and r["tick"] >= 350]
        assert len(rows) == 320
        kp, kv = np.array(report["robot_joint_kp"]), np.array(report["robot_joint_kv"])
        damping = np.array(report["robot_joint_passive_damping"])
        matrices, controls, records = [], [], []
        max_kinematic, max_energy, min_eigenvalue = 0.0, 0.0, float("inf")
        for row in rows:
            with np.load(run / "results" / row["articulated_operator_snapshot"]) as saved:
                mass, hand, fingers = (
                    saved[k].copy() for k in ("joint_mass_matrix", "hand_jacobian_world", "finger_jacobians_world")
                )
                q, v = saved["joint_position"], saved["joint_velocity"]
                np.testing.assert_array_equal(q, row["robot_joint_position"])
                np.testing.assert_array_equal(v, row["robot_joint_velocity"])
                np.testing.assert_allclose(saved["ctrl_pos"][:7], conditions["lift_joints"][0, :7], rtol=0, atol=1e-14)
                np.testing.assert_array_equal(saved["ctrl_pos"][7:], [0.018, 0.018])
                np.testing.assert_array_equal(saved["ctrl_vel"], np.zeros(9))
                energy = float(saved["kinetic_energy_api_J"])
                controls.append(saved["ctrl_pos"].copy())
            np.testing.assert_allclose(mass, mass.T, rtol=0, atol=1e-12)
            min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(mass).min()))
            assert min_eigenvalue > 0
            energy_difference = abs(0.5 * v @ mass @ v - energy)
            max_energy = max(max_energy, energy_difference)
            assert energy_difference <= 1e-10 * abs(energy) + 1e-17
            observed_hand = np.r_[row["hand_velocity_m_s"], row["hand_angular_velocity_rad_s"]]
            kinematic_error = float(np.linalg.norm(hand @ v - observed_hand))
            max_kinematic = max(max_kinematic, kinematic_error)
            assert kinematic_error <= 1e-10
            assert np.max(abs(hand[:, 7:])) < 1e-12
            for index, finger in enumerate(row["fingers"]):
                axis = rotation(finger["world_quaternion_wxyz"]) @ [0, 1, 0]
                np.testing.assert_allclose(fingers[index, :3, 7 + index], axis, rtol=0, atol=1e-12)
                np.testing.assert_allclose(fingers[index, 3:, 7:], 0, rtol=0, atol=1e-12)
                np.testing.assert_allclose(fingers[index, :, 8 - index], 0, rtol=0, atol=1e-12)
            generalized = np.zeros(9)
            for contact in row["contacts"]:
                index = [f["link"] for f in row["fingers"]].index(contact["opposite_link"])
                arm = np.array(contact["world_point_m"]) - row["fingers"][index]["world_position_m"]
                jacobian = fingers[index, :3] + np.cross(fingers[index, 3:].T, arm).T
                generalized += jacobian.T @ -np.array(contact["body_force_N"])
            delta_q = generalized[:7] / kp[:7]
            static_hand_twist = hand[:, :7] @ delta_q
            dt = 0.01 / config["substeps"]
            effective = mass / dt**2 + np.diag((kv + damping) / dt + kp)
            dynamic_delta = np.linalg.solve(effective, generalized)
            records.append(
                {
                    "contact_generalized_effort": generalized.tolist(),
                    "actuator_only_static_hand_translation_m": static_hand_twist[:3].tolist(),
                    "actuator_only_static_hand_rotation_rad": static_hand_twist[3:].tolist(),
                    "unconstrained_one_step_hand_twist": (hand @ dynamic_delta).tolist(),
                }
            )
            matrices.append(mass)
        np.savez_compressed(
            args.output / f"case{len(cases):02d}-operators.npz",
            mass_matrices=np.stack(matrices),
            control_positions=np.stack(controls),
        )
        cases.append(
            {
                "run": str(run),
                "contact_timeconst_s": config["contact_timeconst"],
                "rows": rows,
                "joint_kp": kp.tolist(),
                "joint_kv": kv.tolist(),
                "passive_damping": damping.tolist(),
                "armature": report["robot_joint_armature"],
                "static_equalities": report["runtime_static_equalities"],
                "minimum_mass_eigenvalue": min_eigenvalue,
                "maximum_kinematic_twist_error": max_kinematic,
                "maximum_mass_energy_difference_J": max_energy,
                "actuator_support_diagnostics": records,
            }
        )
    cases.sort(key=lambda case: case["contact_timeconst_s"], reverse=True)
    comparisons = []
    for old, new in pairwise(cases):
        assert old["joint_kp"] == new["joint_kp"] and old["joint_kv"] == new["joint_kv"]
        assert old["armature"] == new["armature"] and old["static_equalities"] == new["static_equalities"]
        assert all(
            [c["opposite_geom"] for c in a["contacts"]] == [c["opposite_geom"] for c in b["contacts"]]
            for a, b in zip(old["rows"], new["rows"], strict=True)
        )
        assert [(r["tick"], r["substep"]) for r in old["rows"]] == [(r["tick"], r["substep"]) for r in new["rows"]]
        a, b = [np.array([[c["normal_force_N"] for c in r["contacts"]] for r in case["rows"]]) for case in (old, new)]
        assert a.shape == b.shape == (320, 2)
        comparisons.append(
            {
                "previous_timeconst_s": old["contact_timeconst_s"],
                "current_timeconst_s": new["contact_timeconst_s"],
                "normal_force_l2_relative_change": float(np.linalg.norm(b - a) / np.linalg.norm(b)),
                "normal_force_max_change_N": float(abs(b - a).max()),
            }
        )
    for case in cases:
        last = case.pop("rows")[-1]
        case["final_normal_forces_N"] = [c["normal_force_N"] for c in last["contacts"]]
        case["final_penetrations_m"] = [c["penetration_m"] for c in last["contacts"]]
    output = {
        "scope": __doc__,
        "cases": cases,
        "comparisons": comparisons,
        "common_physical_samples": 320,
        "goal_completion": False,
        "limitations": "Actual full seven-arm-joint/two-finger inertial and kinematic readouts are verified at corresponding preintegration snapshots. Generalized effort uses Nm for seven arm rotations and N for two finger slides. Actuator-only static and unconstrained one-step response estimates omit equality/contact constraint compliance, follower-force and gravity derivatives; they diagnose support effects and do not replace a solved deformable articulated reference. Contact-law normal-force tails do not establish stress/patch/shear/spatial/history or accuracy-cost acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"comparisons": comparisons, "goal_completion": False}), flush=True)


if __name__ == "__main__":
    main()
