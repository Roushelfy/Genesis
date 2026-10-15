"""Independently reconstruct original-prefix controls, ground contact and full dynamic balances."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np

from .franka_mujoco_reference import PandaSkeleton
from .original_franka_ground import build_original
from .reference_motion_audit import motion_moments
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete-prefix", action="store_true")
    parser.add_argument("--recovery-run", type=Path)
    args = parser.parse_args()
    closed_source(args.run)
    producer_code = int((args.run / "returncode.txt").read_text())
    assert producer_code == 0 or args.allow_incomplete_prefix
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "results/configuration.json").read_text())
    producer_path = args.run / "results/summary.json"
    if producer_code == 0:
        producer = json.loads(producer_path.read_text())
        count = len(producer["steps"])
    else:
        files = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9][0-9].npz"))
        assert files and not producer_path.exists()
        assert all(path.name == f"step{index:05d}.npz" for index, path in enumerate(files, 1))
        count = len(files)
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(config["mjcf"]).read_bytes()).digest()
    static, _frame, capture, protocol = build_original(
        Path(config["capture"]), Path(config["protocol_run"]), args.mjcf, config["dt"], config["dhat"]
    )
    body, permutation = static.body, static.parts[0].permutation
    n = body.basis.N
    dt = config["dt"]
    bulk = WidePiolaBulk(body)
    mass = body.mass[permutation][:, permutation]
    gravity = static.gravity_force([0, 0, -9.81])
    gravity[n:] = 0
    controls = [r for r in protocol["records"] if r["mode"] == "approach_lift"]
    substeps = round(protocol["macro_dt_s"] / dt)
    assert count == config["macro_steps"] * substeps if producer_code == 0 else count < config["macro_steps"] * substeps
    recovered_step = None
    if args.recovery_run is not None:
        assert producer_code != 0 and args.allow_incomplete_prefix
        closed_source(args.recovery_run)
        assert int((args.recovery_run / "returncode.txt").read_text()) == 0
        recovery = json.loads((args.recovery_run / "results/summary.json").read_text())
        assert Path(recovery["source_run"]).resolve() == args.run.resolve()
        assert recovery["recovered_original_equation"] and recovery["final_fraction"] == 1
        assert recovery["original_dt_s"] == dt and recovery["physical_previous_state_fixed"]
        recovered_step = count + 1
        count = recovered_step
    robot = PandaSkeleton(args.mjcf, dt)
    robot.model.opt.tolerance = 1e-14
    kp, kv = np.array(capture["robot_joint_kp"]), np.array(capture["robot_joint_kv"])
    with np.load(args.run / "results/reference_geometry.npz") as saved:
        for key, actual in [("rest_positions", static.rest), ("faces", static.faces), ("groups", static.groups)]:
            np.testing.assert_array_equal(saved[key], actual)
    with np.load(args.run / "results/initial_state.npz") as saved:
        previous = saved["full_displacement"].copy()
        old_q, old_velocity = saved["joint_position"].copy(), saved["joint_velocity"].copy()
        old_body_velocity = saved["body_velocity"].copy()
        np.testing.assert_array_equal(saved["body_displacement"], np.zeros(n))
    np.testing.assert_array_equal(old_q, protocol["initial_state"]["joint_position"])
    np.testing.assert_array_equal(old_velocity, protocol["initial_state"]["joint_velocity"])
    np.testing.assert_array_equal(old_body_velocity, np.zeros(n))
    robot.set_state(old_q, old_velocity)
    names = ["left_finger", "right_finger"]
    finger_local = []
    for name, nodes in zip(names, static.finger_nodes):
        body_id = robot.body_ids[name]
        world = static.rest[nodes] + static.world_coordinate_origin_m
        finger_local.append((world - robot.data.xpos[body_id]) @ robot.data.xmat[body_id].reshape(3, 3))
    rows = []
    friction = ipctk.FrictionPotential(dt * 4e-6)
    for step in range(1, count + 1):
        macro = (step - 1) // substeps
        control = controls[macro]
        state_path = (
            args.recovery_run / "results/recovered.npz"
            if step == recovered_step
            else args.run / f"results/step{step:05d}.npz"
        )
        with np.load(state_path) as saved:
            u, q, velocity, body_velocity, acceleration, force = [
                saved[k].copy()
                for k in [
                    "displacement",
                    "joint_position",
                    "joint_velocity",
                    "body_velocity",
                    "body_acceleration",
                    "contact_force_N",
                ]
            ]
            np.testing.assert_array_equal(saved["controller_target"], control["target"])
            caps = np.r_[robot.model.actuator_forcerange[:7, 1], control["finger_cap_N"], control["finger_cap_N"]]
            np.testing.assert_array_equal(saved["controller_caps"], caps)
            np.testing.assert_array_equal(saved["applied_force_N"], gravity)
        np.testing.assert_array_equal(body_velocity, (u[:n] - previous[:n]) / dt)
        np.testing.assert_array_equal(acceleration, (body_velocity - old_body_velocity) / dt)
        np.testing.assert_allclose(q, old_q + np.longdouble(dt) * velocity, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(u.reshape(-1, 3)[static.ground_nodes], np.zeros((3, 3)))
        normal, current = static.normal_collisions(u)
        tangential = ipctk.TangentialCollisions()
        tangential.build(static.mesh, current, normal, static.barrier, static.friction)
        delta = static.mesh.vertices((u - previous).reshape(-1, 3))
        actual_force = -static.mesh.to_full_dof(
            static.barrier.gradient(normal, static.mesh, current) + friction.gradient(tangential, static.mesh, delta)
        )
        np.testing.assert_allclose(actual_force, force, rtol=1e-10, atol=1e-12)
        full_ids = static.mesh.to_full_vertex_id()
        source_groups = []
        for index in range(len(normal)):
            collision = normal[index]
            ids = np.array(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
            ids = ids[ids >= 0]
            groups = set(static.groups[full_ids[ids]])
            assert groups in ({0, 1}, {0, 2}, {0, 3})
            source_groups.append(int(max(groups)))
            del collision
        vector = np.empty(n, dtype=u.dtype)
        vector[permutation] = u[:n]
        body_residual = bulk.evaluate(vector, False)[1][permutation] + mass @ acceleration - gravity[:n] - force[:n]
        assert np.linalg.norm(body_residual) <= 1e-9
        _, response = robot.velocity_response(old_q, old_velocity)
        inverse = np.linalg.inv(response)
        robot.set_state(q, velocity)
        positions = static.rest + u.reshape(-1, 3) + static.world_coordinate_origin_m
        joint_contact = np.zeros(9)
        for name, nodes, local in zip(names, static.finger_nodes, finger_local):
            body_id = robot.body_ids[name]
            origin = robot.data.xpos[body_id]
            predicted = local @ robot.data.xmat[body_id].reshape(3, 3).T + origin
            np.testing.assert_allclose(positions[nodes], predicted, rtol=0, atol=1e-12)
            point_force = force.reshape(-1, 3)[nodes]
            wrench = np.r_[point_force.sum(axis=0), np.cross(positions[nodes] - origin, point_force).sum(axis=0)]
            joint_contact += np.asarray(robot.jacobian(name).T @ wrench, dtype=float)
        pd = np.asarray(np.clip(kp * (np.array(control["target"]) - q) - kv * velocity, -caps, caps), dtype=float)
        actual_velocity = robot.step_velocity(old_q, old_velocity, pd + joint_contact)
        robot_residual = inverse @ (velocity - actual_velocity)
        assert np.linalg.norm(robot_residual) <= 1e-9
        contact_sum = force.reshape(-1, 3).sum(axis=0)
        assert np.linalg.norm(contact_sum) <= 1e-12
        motion = motion_moments(
            body, permutation, np.asarray(positions[: n // 3], dtype=float), np.asarray(body_velocity, dtype=float)
        )
        row = {
            "step": step,
            "macro_tick": macro,
            "time_s": step * dt,
            "body_dynamic_residual_N": float(np.linalg.norm(body_residual)),
            "actual_robot_step_effort_residual": float(np.linalg.norm(robot_residual)),
            "contact_pullback_max_error_N": float(abs(actual_force - force).max()),
            "contact_groups": source_groups,
            "ground_reaction_world_N": force.reshape(-1, 3)[static.ground_nodes].sum(axis=0).tolist(),
            "body_com_world_m": motion["centre_m"].tolist(),
            "body_com_velocity_world_m_s": motion["linear_velocity_world_m_s"].tolist(),
            "body_angular_velocity_world_rad_s": motion["angular_velocity_world_rad_s"].tolist(),
        }
        if step % substeps == 0:
            native = control["post_macro_state"]
            row["native_COM_position_difference_m"] = float(
                np.linalg.norm(motion["centre_m"] - native["egg_com_world_m"])
            )
            row["native_joint_position_difference_L2"] = float(np.linalg.norm(q - native["joint_position"]))
        rows.append(row)
        np.savez_compressed(
            args.output / f"step{step:05d}.npz",
            body_residual_N=body_residual,
            actual_robot_effort_residual=robot_residual,
            contact_force_N=actual_force,
        )
        previous, old_q, old_velocity, old_body_velocity = u, q, velocity, body_velocity
    result = {
        "scope": __doc__,
        "run": str(args.run),
        "steps_checked": len(rows),
        "producer_returncode": producer_code,
        "complete_requested_history": producer_code == 0,
        "recovery_run": str(args.recovery_run) if args.recovery_run else None,
        "independently_verified_recovered_step": recovered_step,
        "rows": rows,
        "goal_completion": False,
        "limitations": "This checks only the declared original-history prefix, with real ground support and original controls. Complete grasp/lift, spatial/contact/geometry/time convergence, matching rigid feedback and native model/accuracy-cost acceptance remain incomplete. Source native contacts are preintegration while macro states are postintegration; reported same-macro state differences are physical diagnostics, not contact-law equivalence.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({"states_checked": len(rows)}))


if __name__ == "__main__":
    main()
