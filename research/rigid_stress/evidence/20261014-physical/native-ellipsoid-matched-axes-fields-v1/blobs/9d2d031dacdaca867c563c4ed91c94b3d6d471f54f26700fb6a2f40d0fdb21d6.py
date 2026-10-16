"""Check an independent articulated backend against every captured motion frame."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .franka_mujoco_reference import PandaSkeleton
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.capture)
    assert int((args.capture / "returncode.txt").read_text()) == 0
    report = json.loads((args.capture / "results/summary.json").read_text())
    robot = PandaSkeleton(args.mjcf, 0.01 / report["substeps"])
    (args.output / "skeleton.xml").write_text(robot.xml)
    np.testing.assert_array_equal(robot.model.dof_armature, report["robot_joint_armature"])
    np.testing.assert_array_equal(robot.model.dof_damping, report["robot_joint_passive_damping"])
    equality = report["runtime_static_equalities"][0]
    np.testing.assert_array_equal(robot.model.eq_solref[0], equality["sol_params"][:2])
    np.testing.assert_array_equal(robot.model.eq_solimp[0], equality["sol_params"][2:])
    np.testing.assert_array_equal(robot.model.eq_data[0, :5], equality["data"][:5])
    maxima, rows = {}, []
    responses = []
    selected_frames = {0, len(report["captures"]) // 2, len(report["captures"]) - 1}
    for frame_index, frame in enumerate(report["captures"]):
        with np.load(args.capture / "results" / frame["articulated_operator_snapshot"]) as saved:
            robot.set_state(saved["joint_position"], saved["joint_velocity"])
            values = {
                "physical_mass": (robot.mass(), saved["physical_joint_mass_matrix"], 1e-10, 1e-12),
                "bias": (robot.data.qfrc_bias, saved["qf_bias"], 1e-9, 1e-10),
                "passive": (robot.data.qfrc_passive, saved["qf_passive"], 1e-10, 1e-12),
                "hand_jacobian": (robot.jacobian("hand"), saved["hand_jacobian_world"], 1e-10, 1e-12),
                "finger_jacobians": (
                    np.stack([robot.jacobian(n) for n in ["left_finger", "right_finger"]]),
                    saved["finger_jacobians_world"],
                    1e-10,
                    1e-12,
                ),
            }
        record = {"mode": frame["mode"], "tick": frame["tick"], "substep": frame["substep"]}
        for key, (actual, expected, rtol, atol) in values.items():
            error = float(abs(actual - expected).max())
            record[key + "_max_absolute_error"] = error
            maxima[key] = max(maxima.get(key, 0), error)
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=key)
        for contact in frame["contacts"]:
            finger_index = [r["link"] for r in frame["fingers"]].index(contact["opposite_link"])
            name = ["left_finger", "right_finger"][finger_index]
            origin = np.array(frame["fingers"][finger_index]["world_position_m"])
            point = np.array(contact["world_point_m"])
            jacobian = robot.jacobian(name)
            expected = jacobian[:3] + np.cross(jacobian[3:].T, point - origin).T
            np.testing.assert_allclose(robot.point_jacobian(name, point)[:3], expected, rtol=1e-10, atol=1e-12)
        for name, position, quaternion in [
            ("hand", frame["hand_position_m"], frame["hand_quaternion_wxyz"]),
            *[
                (n, r["world_position_m"], r["world_quaternion_wxyz"])
                for n, r in zip(["left_finger", "right_finger"], frame["fingers"])
            ],
        ]:
            body = robot.body_ids[name]
            expected_rotation = np.empty(9)
            mujoco.mju_quat2Mat(expected_rotation, np.array(quaternion))
            np.testing.assert_allclose(robot.data.xpos[body], position, rtol=0, atol=1e-12)
            np.testing.assert_allclose(robot.data.xmat[body], expected_rotation, rtol=0, atol=1e-12)
        if frame_index in selected_frames:
            q, velocity = robot.data.qpos.copy(), robot.data.qvel.copy()
            baseline, response = robot.velocity_response(q, velocity)
            max_force_equivalent_error = 0.0
            for effort in [np.linspace(-2, 2, 9), np.array(frame["robot_joint_control_force_readout"])]:
                actual = robot.step_velocity(q, velocity, effort)
                predicted = baseline + response @ effort
                difference = np.linalg.solve(response, actual - predicted)
                max_force_equivalent_error = max(max_force_equivalent_error, float(abs(difference).max()))
                assert np.linalg.norm(difference) <= 1e-9
            np.savez_compressed(
                args.output / f"response{frame_index:04d}.npz",
                joint_position=q,
                joint_velocity=velocity,
                baseline_next_velocity=baseline,
                next_velocity_effort_response=response,
            )
            responses.append({"frame": frame_index, "max_effort_equivalent_error": max_force_equivalent_error})
        rows.append(record)
    result = {
        "scope": __doc__,
        "mujoco_version": mujoco.__version__,
        "frames_checked": len(rows),
        "source_mjcf_sha256": hashlib.sha256(args.mjcf.read_bytes()).hexdigest(),
        "max_absolute_errors": maxima,
        "velocity_response_qualifications": responses,
        "rows": rows,
        "goal_completion": False,
        "limitations": "Authored nine-joint mass, bias, passive damping, finger/hand poses and Jacobians are checked at actual motion/hold inputs with original armature and equality parameters. Original tendon actuation is disabled because the application supplies independent nine-joint PD controls. Collision geometry is supplied separately by the captured actual compound surfaces. This backend qualification does not establish coupled deformable-contact equilibrium, matching control/initialization/history, temporal/spatial/contact convergence or accuracy-cost acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
