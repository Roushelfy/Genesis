"""Capture corresponding actual Panda contacts, geometry and stationary actuation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array


def array(tensor):
    return tensor_to_array(tensor).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-steps", type=int, default=330)
    parser.add_argument("--hold-steps", type=int, default=360)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--contact-timeconst", type=float)
    parser.add_argument("--dense-hold-tail-steps", type=int, default=0)
    parser.add_argument("--capture-articulated-operators", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda, precision="64", seed=510000, logging_level="warning")
    workload = FrankaEgg(
        1,
        substeps=args.substeps,
        varied=False,
        fused_rigid_post=False,
        contact_model="intrinsic",
        intrinsic_load_partition="componentwise",
        output_mode="full",
    )
    workload.save_conditions(args.output / "conditions.npz")
    solver = workload.scene.rigid_solver
    if args.contact_timeconst is not None:
        assert args.contact_timeconst >= 2 * workload.scene.dt / args.substeps
        geom_indices = np.arange(solver.n_geoms)
        parameters = array(solver.get_sol_params(geoms_idx=geom_indices)).reshape(-1, 7)
        parameters[:, 0] = args.contact_timeconst
        solver.set_sol_params(parameters, geoms_idx=geom_indices)
    recovery = solver.stress_recovery
    entry = recovery.links[0]
    original = recovery.recover
    fingers = [workload.robot.get_link(name) for name in ("left_finger", "right_finger")]
    geometry = []
    for finger in fingers:
        for geom in finger.geoms:
            mesh = geom.get_trimesh()
            path = args.output / f"geom{geom.idx}_local.npz"
            np.savez_compressed(
                path, vertices_m=np.array(mesh.vertices), triangles=np.array(mesh.faces), data=geom.data
            )
            geometry.append(
                {
                    "geom": geom.idx,
                    "link": finger.idx,
                    "type": str(geom.type),
                    "data": geom.data.tolist(),
                    "is_convex": geom.is_convex,
                    "mesh": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (args.output / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    mode, tick, save = "approach_lift", 0, False
    macro_substep = 0
    rows = []

    def observe(substep):
        nonlocal macro_substep
        original(substep)
        physical_substep = macro_substep
        macro_substep += 1
        if not save:
            return
        position = qd_to_numpy(entry.contacts.frame_position, copy=True).reshape(3)
        quaternion = qd_to_numpy(entry.contacts.frame_quaternion, copy=True).reshape(4)
        np.testing.assert_allclose(
            position, array(workload.link.get_pos(relative=False)).reshape(3), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            abs(quaternion @ array(workload.link.get_quat(relative=False)).reshape(4)), 1, rtol=0, atol=1e-12
        )
        collider = solver.collider.collider_state
        slots = qd_to_numpy(collider.contact_sort_idx, transpose=True, copy=True)[0]
        sources = {
            name: qd_to_numpy(getattr(collider.contact_data, name), transpose=True, copy=True)[0]
            for name in ("pos", "force", "normal", "link_a", "link_b", "geom_a", "geom_b", "penetration", "sol_params")
        }
        valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)[0]
        friction = qd_to_numpy(entry.contacts.friction, transpose=True, copy=True)[0]
        contacts = []
        for index in np.flatnonzero(valid):
            source = int(slots[index])
            a, b = int(sources["link_a"][source]), int(sources["link_b"][source])
            assert workload.link.idx in (a, b)
            sign = -1 if a == workload.link.idx else 1
            force = sign * sources["force"][source]
            inward = -sign * sources["normal"][source]
            np.testing.assert_allclose(np.linalg.norm(inward), 1, rtol=0, atol=1e-12)
            normal = float(force @ inward)
            shear = float(np.linalg.norm(force - normal * inward))
            assert normal >= -1e-10 and shear <= friction[index] * normal + 1e-8
            contacts.append(
                {
                    "source_slot": source,
                    "world_point_m": sources["pos"][source].tolist(),
                    "body_force_N": force.tolist(),
                    "body_inward_normal": inward.tolist(),
                    "friction": float(friction[index]),
                    "opposite_link": b if a == workload.link.idx else a,
                    "opposite_geom": int(sources["geom_b" if a == workload.link.idx else "geom_a"][source]),
                    "normal_force_N": normal,
                    "shear_force_N": shear,
                    "penetration_m": float(sources["penetration"][source]),
                    "mixed_contact_sol_params": sources["sol_params"][source].tolist(),
                }
            )
        poses = []
        fields = {}
        for finger in fingers:
            poses.append(
                {
                    "link": finger.idx,
                    "world_position_m": array(finger.get_pos(relative=False)).reshape(3).tolist(),
                    "world_quaternion_wxyz": array(finger.get_quat(relative=False)).reshape(4).tolist(),
                }
            )
            for geom in finger.geoms:
                fields[f"geom{geom.idx}_world_vertices_m"] = array(geom.get_verts()).reshape(-1, 3)
        row = {
            "mode": mode,
            "tick": tick,
            "substep": physical_substep,
            "recovery_local_index": int(substep),
            "phase": float(array(workload.phase)[0]),
            "frame": "constraint forces evaluated, before rigid integration",
            "egg_position_m": position.tolist(),
            "egg_mass_kg": float(array(workload.link.get_mass()).reshape(-1)[0]),
            "egg_world_com_m": qd_to_numpy(solver.dyn_state.links.root_COM, transpose=True, copy=True)[
                0, workload.link.idx
            ].tolist(),
            "egg_quaternion_wxyz": quaternion.tolist(),
            "egg_velocity_m_s": array(workload.link.get_vel(relative=False)).reshape(3).tolist(),
            "egg_angular_velocity_rad_s": array(workload.link.get_ang()).reshape(3).tolist(),
            "finger_position_m": array(workload.robot.get_dofs_position(workload.fingers)).reshape(2).tolist(),
            "finger_velocity_m_s": array(workload.robot.get_dofs_velocity(workload.fingers)).reshape(2).tolist(),
            "finger_control_force_N": array(workload.robot.get_dofs_control_force(workload.fingers))
            .reshape(2)
            .tolist(),
            "fingers": poses,
            "finger_masses_kg": [float(array(finger.get_mass()).reshape(-1)[0]) for finger in fingers],
            "contacts": contacts,
            "hand_position_m": array(workload.hand.get_pos(relative=False)).reshape(3).tolist(),
            "hand_quaternion_wxyz": array(workload.hand.get_quat(relative=False)).reshape(4).tolist(),
            "hand_velocity_m_s": array(workload.hand.get_vel(relative=False)).reshape(3).tolist(),
            "hand_angular_velocity_rad_s": array(workload.hand.get_ang()).reshape(3).tolist(),
            "robot_joint_position": array(workload.robot.get_dofs_position()).reshape(9).tolist(),
            "robot_joint_velocity": array(workload.robot.get_dofs_velocity()).reshape(9).tolist(),
            "robot_joint_control_force_readout": array(workload.robot.get_dofs_control_force()).reshape(9).tolist(),
        }
        name = f"{mode}-{tick:04d}-s{physical_substep:02d}"
        row["geometry_snapshot"] = name + "_geometry.npz"
        if args.capture_articulated_operators:
            dofs = slice(workload.robot.dof_start, workload.robot.dof_end)
            mass = array(workload.robot.get_mass_mat()).reshape(9, 9)
            np.testing.assert_allclose(mass, mass.T, rtol=0, atol=1e-12)
            assert np.linalg.eigvalsh(mass).min() > 0
            operator_fields = {
                "joint_mass_matrix": mass,
                "hand_jacobian_world": array(workload.robot.get_jacobian(workload.hand)).reshape(6, 9),
                "finger_jacobians_world": np.stack(
                    [array(workload.robot.get_jacobian(finger)).reshape(6, 9) for finger in fingers]
                ),
                "joint_position": np.array(row["robot_joint_position"]),
                "joint_velocity": np.array(row["robot_joint_velocity"]),
                "kinetic_energy_api_J": array(workload.robot.get_kinetic_energy()).reshape(()),
            }
            for field in (
                "ctrl_pos",
                "ctrl_vel",
                "ctrl_mode",
                "ctrl_force",
                "qf_bias",
                "qf_applied",
                "qf_passive",
                "qf_constraint",
            ):
                operator_fields[field] = qd_to_numpy(getattr(solver.dyn_state.dofs, field), transpose=True, copy=True)[
                    0, dofs
                ]
            correction = np.zeros(9)
            if solver._options.integrator == gs.integrator.approximate_implicitfast:
                dt = workload.scene.dt / args.substeps
                damping = array(workload.robot.get_dofs_damping()).reshape(9)
                kv = array(workload.robot.get_dofs_kv()).reshape(9)
                correction = dt * (damping + kv * (operator_fields["ctrl_mode"] <= gs.CTRL_MODE.VELOCITY))
            physical_mass = mass - np.diag(correction)
            operator_fields["physical_joint_mass_matrix"] = physical_mass
            operator_fields["implicit_damping_mass_correction"] = correction
            assert np.linalg.eigvalsh(physical_mass).min() > 0
            v = operator_fields["joint_velocity"]
            np.testing.assert_allclose(
                0.5 * v @ physical_mass @ v, operator_fields["kinetic_energy_api_J"], rtol=1e-10, atol=1e-17
            )
            row["articulated_operator_snapshot"] = name + "_articulated.npz"
            np.savez_compressed(args.output / row["articulated_operator_snapshot"], **operator_fields)
        np.savez_compressed(args.output / f"{name}_geometry.npz", **fields)
        (args.output / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)

    recovery.recover = observe

    from .franka_original_protocol import control_at_tick

    with np.load(args.output / "conditions.npz") as saved:
        protocol_conditions = {key: saved[key].copy() for key in saved.files}
    protocol_rows = []

    def protocol_state():
        origin = array(workload.link.get_pos(relative=False)).reshape(3)
        com = array(solver.get_links_pos(workload.link.idx, ref=gs.link_ref_frame.link_COM)).reshape(3)
        velocity = array(solver.get_links_vel(workload.link.idx, ref=gs.link_ref_frame.link_COM)).reshape(3)
        omega = array(workload.link.get_ang()).reshape(3)
        origin_velocity = array(workload.link.get_vel(relative=False)).reshape(3)
        np.testing.assert_allclose(velocity, origin_velocity + np.cross(omega, com - origin), rtol=0, atol=1e-12)
        return {"egg_origin_world_m": origin.tolist(), "egg_com_world_m": com.tolist(),
                "egg_com_velocity_world_m_s": velocity.tolist(),
                "egg_angular_velocity_world_rad_s": omega.tolist(),
                "egg_quaternion_wxyz": array(workload.link.get_quat(relative=False)).reshape(4).tolist(),
                "joint_position": array(workload.robot.get_dofs_position()).reshape(9).tolist(),
                "joint_velocity": array(workload.robot.get_dofs_velocity()).reshape(9).tolist()}

    def record_protocol(mode, tick):
        assert workload.robot.dof_start == 0 and workload.robot.n_dofs == 9
        target = qd_to_numpy(solver.dyn_state.dofs.ctrl_pos, transpose=True, copy=True)[0, :9]
        if mode == "approach_lift":
            expected = control_at_tick(protocol_conditions, tick)
            cap = float(array(workload.controller_inputs["limit"]).reshape(-1)[0])
            radius = float(array(workload.controller_inputs["radius"]).reshape(-1)[0])
            np.testing.assert_allclose(target, expected["target"], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(cap, expected["finger_cap_N"])
            np.testing.assert_allclose(radius, expected["native_radius_m"], rtol=0, atol=1e-16)
        else:
            expected = {"target": np.r_[array(workload.lift_joints).reshape(9)[:7], 0.018, 0.018]}
            np.testing.assert_array_equal(target, expected["target"])
            cap, radius = 4.0, None
        collider = solver.collider.collider_state
        slots = qd_to_numpy(collider.contact_sort_idx, transpose=True, copy=True)[0]
        fields = {name: qd_to_numpy(getattr(collider.contact_data, name), transpose=True, copy=True)[0]
                  for name in ("pos","force","normal","link_a","link_b","geom_a","geom_b","penetration","sol_params")}
        valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)[0]
        friction = qd_to_numpy(entry.contacts.friction, transpose=True, copy=True)[0]
        contacts = []
        for index in np.flatnonzero(valid):
            source = int(slots[index])
            a,b = int(fields["link_a"][source]),int(fields["link_b"][source])
            assert workload.link.idx in (a,b)
            sign = -1 if a == workload.link.idx else 1
            contacts.append({"force_on_body_N": (sign*fields["force"][source]).tolist(),
                             "point_world_m": fields["pos"][source].tolist(),
                             "inward_normal_world": (-sign*fields["normal"][source]).tolist(),
                             "opposite_link": b if a == workload.link.idx else a,
                             "friction": float(friction[index]),"penetration_m": float(fields["penetration"][source]),
                             "sol_params": fields["sol_params"][source].tolist()})
        protocol_rows.append({"mode":mode,"tick":tick,"target":target.tolist(),"finger_cap_N":cap,
                              "native_radius_m":radius,"post_macro_state":protocol_state(),
                              "last_preintegration_constraint_origin_world_m":qd_to_numpy(entry.contacts.frame_position,copy=True).reshape(3).tolist(),
                              "last_preintegration_contacts":contacts})

    initial_protocol_state = protocol_state()

    for tick in range(args.approach_steps):
        macro_substep = 0
        save = tick in (209, 299, args.approach_steps - 1)
        workload.step()
        solver.check_errno()
        record_protocol("approach_lift", tick)
        if tick % 100 == 99:
            print("Captured actual protocol tick", tick, flush=True)
    mode = "stationary_lift_hold"
    workload.robot.control_dofs_position(workload.lift_joints[:, workload.arm], workload.arm)
    workload.robot.set_dofs_force_range([-4, -4], [4, 4], workload.fingers)
    workload.robot.control_dofs_position(0.018, workload.fingers)
    for tick in range(args.hold_steps):
        macro_substep = 0
        save = tick % 60 == 59 or tick >= args.hold_steps - max(args.dense_hold_tail_steps, 1)
        workload.scene.step()
        solver.check_errno()
        record_protocol("stationary_lift_hold", tick)
        if save:
            print("Captured stationary hold tick", tick, flush=True)
    summary = {
        "scope": __doc__,
        "substeps": args.substeps,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "robot_joint_kp": array(workload.robot.get_dofs_kp()).reshape(9).tolist(),
        "robot_joint_kv": array(workload.robot.get_dofs_kv()).reshape(9).tolist(),
        "robot_joint_armature": array(workload.robot.get_dofs_armature()).reshape(9).tolist(),
        "robot_joint_passive_damping": array(workload.robot.get_dofs_damping()).reshape(9).tolist(),
        "articulated_snapshot_schema": "physical-and-working-mass-controller-bias-v2"
        if args.capture_articulated_operators
        else None,
        "runtime_integrator": str(solver._options.integrator),
        "runtime_geom_sol_params": array(solver.get_sol_params(geoms_idx=np.arange(solver.n_geoms))).tolist(),
        "runtime_static_equalities": [
            {
                "type": str(equality.type),
                "object1": int(equality.eq_obj1id),
                "object2": int(equality.eq_obj2id),
                "data": np.asarray(equality.eq_data).tolist(),
                "sol_params": array(solver.get_sol_params(eqs_idx=[index])).reshape(7).tolist(),
            }
            for index, equality in enumerate(solver.equalities)
        ],
        "body_asset_metadata": json.loads(
            (Path("examples/rigid/assets/hollow_egg/level1/egg_shell.metadata.json")).read_text()
        ),
        "geometry": geometry,
        "captures": rows,
        "stationary_actuation": {
            "finger_target_m": 0.018,
            "finger_kp_N_per_m": 100,
            "finger_kv_N_s_per_m": 10,
            "finger_force_cap_N": 4,
            "arm_target": array(workload.lift_joints[:, workload.arm]).tolist(),
        },
        "goal_completion": False,
        "limitations": "This is an actual rigid-application input capture, not a deformable reference. The added stationary hold keeps lift targets and original PD gains/caps fixed after the original approach/grip/lift. Same-substep source forces, poses and effective friction are retained; equilibrium/velocity settling, matching full finger geometry/actuation/mass, contact-reference convergence and deformation feedback require independent verification.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "original_protocol.json").write_text(json.dumps({
        "scope":"Read actual original controls and full macro-step motion without changing dynamics.",
        "initial_state":initial_protocol_state,"records":protocol_rows,
        "macro_dt_s":workload.scene.dt,"substeps":args.substeps,
        "state_frame":"post macro integration; listed contacts are the last preintegration constraint evaluation",
        "goal_completion":False,
    },indent=2)+"\n")


if __name__ == "__main__":
    main()
