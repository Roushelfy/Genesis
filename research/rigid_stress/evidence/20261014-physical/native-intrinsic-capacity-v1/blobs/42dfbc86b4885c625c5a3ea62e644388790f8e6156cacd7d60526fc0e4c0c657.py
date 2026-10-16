"""Independently audit original-control rigid histories with direct body force/moment and fresh Panda steps."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy.spatial.transform import Rotation

from .franka_mujoco_reference import PandaSkeleton
from .original_franka_ground import build_original
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--allow-incomplete-prefix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    code = int((args.run / "returncode.txt").read_text())
    assert code == 0 or args.allow_incomplete_prefix
    config_path = args.run / "results/configuration.json"
    config = json.loads(config_path.read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, _frame, capture, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    geometry = args.run / "results/reference_geometry.npz"
    parent_geometry = Path(config["matched_deformable"]) / "results/reference_geometry.npz"
    assert geometry.read_bytes() == parent_geometry.read_bytes()
    body, n, dt = static.body, static.body.basis.N, config["dt"]
    permutation = static.parts[0].permutation
    mass = body.mass[permutation][:, permutation]
    gravity = static.gravity_force([0, 0, -9.81])[:n]
    radius = float(np.linalg.norm(static.rest[: n // 3] - body.com, axis=1).max())
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    substeps = round(protocol["macro_dt_s"] / dt)
    paths = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
    assert paths and [path.name for path in paths] == [f"step{step:05d}.npz" for step in range(1, len(paths) + 1)]
    expected = config["macro_steps"] * substeps
    complete = code == 0 and len(paths) == expected and (args.run / "results/summary.json").exists()
    assert complete or args.allow_incomplete_prefix
    robot = PandaSkeleton(args.mjcf, dt)
    robot.model.opt.tolerance = 1e-14
    kp, kv = [np.array(capture[key]) for key in ["robot_joint_kp", "robot_joint_kv"]]
    with np.load(args.run / "results/initial_state.npz") as saved:
        previous = saved["full_displacement"].copy()
        old_q, old_velocity, old_body_velocity = [
            saved[key].copy() for key in ["joint_position", "joint_velocity", "body_velocity"]
        ]
        np.testing.assert_array_equal(saved["body_pose"], np.zeros(6))
    np.testing.assert_array_equal(old_q, protocol["initial_state"]["joint_position"])
    np.testing.assert_array_equal(old_velocity, protocol["initial_state"]["joint_velocity"])
    np.testing.assert_array_equal(old_body_velocity, np.zeros(n))
    robot.set_state(old_q, old_velocity)
    finger_names, local = ["left_finger", "right_finger"], []
    for name, nodes in zip(finger_names, static.finger_nodes, strict=True):
        body_id = robot.body_ids[name]
        points = static.rest[nodes] + static.world_coordinate_origin_m
        local.append((points - robot.data.xpos[body_id]) @ robot.data.xmat[body_id].reshape(3, 3))
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {str(config_path): hashlib.sha256(config_path.read_bytes()).hexdigest()}
    rows = []
    for step, path in enumerate(paths, 1):
        with np.load(path) as saved:
            u, pose, q, velocity, bv, acceleration, force = [
                saved[key].copy()
                for key in [
                    "displacement",
                    "body_pose",
                    "joint_position",
                    "joint_velocity",
                    "body_velocity",
                    "body_acceleration",
                    "contact_force_N",
                ]
            ]
            np.testing.assert_array_equal(saved["body_component_permutation"], permutation)
            np.testing.assert_array_equal(saved["applied_force_N"][:n], gravity)
        np.testing.assert_allclose(q, old_q + np.longdouble(dt) * velocity, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(bv, (u[:n] - previous[:n]) / dt)
        np.testing.assert_array_equal(acceleration, (bv - old_body_velocity) / dt)
        rotation = Rotation.from_rotvec(np.asarray(pose[3:], dtype=float)).as_matrix()
        expected_u = (static.rest[: n // 3] - body.com) @ (rotation - np.eye(3)).T + pose[:3]
        np.testing.assert_allclose(u[:n].reshape(-1, 3), expected_u, rtol=0, atol=1e-14)
        robot.set_state(q, velocity)
        for name, nodes, points in zip(finger_names, static.finger_nodes, local, strict=True):
            body_id = robot.body_ids[name]
            world = points @ robot.data.xmat[body_id].reshape(3, 3).T + robot.data.xpos[body_id]
            np.testing.assert_allclose(
                static.rest[nodes] + u.reshape(-1, 3)[nodes] + static.world_coordinate_origin_m,
                world,
                rtol=0,
                atol=1e-12,
            )
        normal, current = static.normal_collisions(u)
        start = static.mesh.vertices(static.rest + previous.reshape(-1, 3))
        assert not ipctk.has_intersections(static.mesh, start) and not ipctk.has_intersections(static.mesh, current)
        motion_intervals = None
        for interval_count in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            # Independently check the exact same straight trajectory. These
            # geometric queries neither integrate dynamics nor change timestep.
            safe_intervals = 0
            for index in range(interval_count):
                left = (1 - index / interval_count) * start + (index / interval_count) * current
                right = (1 - (index + 1) / interval_count) * start + ((index + 1) / interval_count) * current
                if ipctk.compute_collision_free_stepsize(static.mesh, left, right) < 1 - 1e-12:
                    break
                safe_intervals += 1
            if safe_intervals == interval_count:
                motion_intervals = interval_count
                break
        assert motion_intervals is not None
        tangential = ipctk.TangentialCollisions()
        tangential.build(static.mesh, current, normal, static.barrier, static.friction)
        delta = static.mesh.vertices((u - previous).reshape(-1, 3))
        friction = ipctk.FrictionPotential(dt * 4e-6)
        normal_force = -static.mesh.to_full_dof(static.barrier.gradient(normal, static.mesh, current))
        shear_force = -static.mesh.to_full_dof(friction.gradient(tangential, static.mesh, delta))
        actual_force = normal_force + shear_force
        np.testing.assert_allclose(actual_force, force, rtol=1e-12, atol=1e-12)
        balance = (mass @ acceleration - gravity - actual_force[:n]).reshape(-1, 3)
        relative = static.rest[: n // 3] + u[:n].reshape(-1, 3) - body.com - pose[:3]
        force_residual, moment_residual = balance.sum(axis=0), np.cross(relative, balance).sum(axis=0)
        equivalent = float(np.linalg.norm(np.r_[force_residual, moment_residual / radius]))
        assert equivalent <= 1e-9
        control = controls[(step - 1) // substeps]
        caps = np.r_[robot.model.actuator_forcerange[:7, 1], control["finger_cap_N"], control["finger_cap_N"]]
        effort = np.clip(kp * (np.array(control["target"]) - q) - kv * velocity, -caps, caps)
        for name, nodes in zip(finger_names, static.finger_nodes, strict=True):
            for node in nodes:
                if np.any(actual_force[3 * node : 3 * node + 3]):
                    point = static.rest[node] + u.reshape(-1, 3)[node] + static.world_coordinate_origin_m
                    effort += robot.point_jacobian(name, point)[:3].T @ actual_force[3 * node : 3 * node + 3]
        actual = robot.step_velocity(old_q, old_velocity, np.asarray(effort, dtype=float))
        _baseline, response = robot.velocity_response(old_q, old_velocity)
        robot_error = float(np.linalg.norm(np.linalg.solve(response, np.asarray(velocity - actual, dtype=float))))
        assert robot_error <= 1e-9
        row = {
            "step": step,
            "macro_tick": (step - 1) // substeps,
            "time_s": step * dt,
            "body_force_equivalent_residual_N": equivalent,
            "body_net_force_residual_N": float(np.linalg.norm(force_residual)),
            "body_net_moment_residual_Nm": float(np.linalg.norm(moment_residual)),
            "fresh_actual_robot_effort_residual": robot_error,
            "contact_pullback_max_difference_N": float(abs(actual_force - force).max()),
            "normal_stencils": len(normal),
            "independently_verified_straight_motion_CCD_intervals": motion_intervals,
        }
        rows.append(row)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        np.savez_compressed(
            args.output / f"step{step:05d}.npz",
            body_net_force_residual_N=force_residual,
            body_net_moment_residual_Nm=moment_residual,
            contact_force_N=actual_force,
        )
        previous, old_q, old_velocity, old_body_velocity = u, q, velocity, bv
    result = {
        "scope": __doc__,
        "run": str(args.run),
        "states_checked": len(rows),
        "requested_states": expected,
        "complete_requested_history": complete,
        "complete_original_330_macro_history": complete and config["macro_steps"] == 330 and len(rows) == 2640,
        "producer_returncode": code,
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Rigid-body net force/moment and fresh actual old-state Panda-step residuals are reconstructed independently without importing the coupled rigid solver. Rigidity is checked against an independent finite-rotation map, and joint integration/kinematics/contact forces use actual saved states. An explicitly allowed failed prefix is not a complete history. This does not establish physical contact/space/time convergence, native load-model accuracy or verified warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_checked": len(rows), "complete_requested_history": complete}), flush=True)


if __name__ == "__main__":
    main()
