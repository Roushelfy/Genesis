"""Corresponding rigid fixed-hand Panda fixture; retain original jaw PD and geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .audit_franka_reference import rotation
from .franka_fixture_geometry import build_fixture


def array(value):
    return tensor_to_array(value).copy()


def geometry_audit(robot, manifest, expected):
    records = []
    for name in ("left_finger", "right_finger"):
        geoms = robot.get_link(name).geoms
        sources = [record for record in manifest["geometry"] if record["link_name"] == name]
        assert len(geoms) == len(sources) == 6
        vertices = [array(geom.get_verts()).reshape(-1, 3) for geom in geoms]
        target = [expected[f"geom{source['source_geom']}_world_vertices_m"] for source in sources]
        errors = np.array(
            [[max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()) for b in target] for a in vertices]
        )
        rows, cols = linear_sum_assignment(errors)
        for i, j in zip(rows, cols, strict=True):
            assert str(geoms[i].type) == sources[j]["type"]
            error = float(errors[i, j])
            assert error < 1e-10, (name, geoms[i].idx, sources[j], error)
            records.append(
                {
                    "fixture_geom": geoms[i].idx,
                    "source_geom": sources[j]["source_geom"],
                    "type": str(geoms[i].type),
                    "world_vertex_hausdorff_error_m": error,
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--contact-timeconst", type=float)
    parser.add_argument("--control-steps", type=int, default=64)
    parser.add_argument("--target-hold-seconds", type=float, default=0.5)
    parser.add_argument("--release-hold-seconds", type=float, default=20)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    row, manifest = build_fixture(args.capture, args.output / "fixture-assets")
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    gs.init(backend=gs.cuda, precision="64", seed=510000, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=args.substeps),
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=gs.contact_resolution.convex,
            iterations=100,
            tolerance=1e-12,
            contact_pruning_tolerance=None,
            enable_torsional_friction=False,
            enable_rolling_friction=False,
            use_hibernation=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=args.output / "fixture-assets/fixed_hand.urdf",
            fixed=True,
            pos=tuple(row["hand_position_m"]),
            quat=tuple(row["hand_quaternion_wxyz"]),
            align=False,
            merge_fixed_links=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=True,
            default_armature=0,
        ),
        material=gs.materials.Rigid(friction=0.1),
    )
    assets = args.capture / "source/examples/rigid/assets/hollow_egg/level1"
    egg = scene.add_entity(
        gs.morphs.URDF(
            file=assets / "egg_shell.urdf",
            pos=tuple(row["egg_position_m"]),
            quat=tuple(row["egg_quaternion_wxyz"]),
            align=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=False,
        ),
        material=gs.materials.Rigid(friction=0.6),
    )
    egg.base_link.configure_stress_recovery(
        gs.options.RigidStressOptions(
            mesh=assets / "elastic.npz",
            contact_model="intrinsic",
            intrinsic_load_partition="componentwise",
            output_mode="full",
            fused_rigid_post=False,
        )
    )
    scene.build(n_envs=1)
    solver = scene.rigid_solver
    geom_indices = np.array([geom.idx for geom in (*robot.geoms, *egg.geoms)])
    geom_parameters_before = array(solver.get_sol_params(geoms_idx=geom_indices)).reshape(-1, 7)
    if args.contact_timeconst is not None:
        assert args.contact_timeconst >= 2 * scene.dt / args.substeps
        parameters = geom_parameters_before.copy()
        parameters[:, 0] = args.contact_timeconst
        solver.set_sol_params(parameters, geoms_idx=geom_indices)
    geom_parameters_after = array(solver.get_sol_params(geoms_idx=geom_indices)).reshape(-1, 7)
    np.testing.assert_allclose(geom_parameters_after[:, 1:], geom_parameters_before[:, 1:], rtol=0, atol=0)
    assert robot.n_dofs == 2
    joints = np.arange(2)
    robot.set_dofs_position([0, 0], joints)
    robot.set_dofs_kp([100, 100], joints)
    robot.set_dofs_kv([10, 10], joints)
    # Source Panda finger joints inherit these passive joint parameters.
    robot.set_dofs_armature([0.1, 0.1], joints)
    robot.set_dofs_damping([1, 1], joints)
    robot.set_dofs_force_range([-4, -4], [4, 4], joints)
    robot.control_dofs_position([0, 0], joints)
    with np.load(args.output / "fixture-assets/expected_open_geometry.npz") as expected:
        geometry = geometry_audit(robot, manifest, expected)
    fingers = [robot.get_link(name) for name in ("left_finger", "right_finger")]
    for i, finger in enumerate(fingers):
        np.testing.assert_allclose(
            array(finger.get_mass()).reshape(-1)[0], row["finger_masses_kg"][i], rtol=0, atol=1e-14
        )
        old = row["fingers"][i]
        axis = rotation(old["world_quaternion_wxyz"]) @ [0, 1, 0]
        expected_pos = np.array(old["world_position_m"]) + (0.04 - row["finger_position_m"][i]) * axis
        np.testing.assert_allclose(array(finger.get_pos(relative=False)).reshape(3), expected_pos, rtol=0, atol=1e-10)
    np.testing.assert_allclose(array(egg.base_link.get_mass()).reshape(-1)[0], row["egg_mass_kg"], rtol=0, atol=1e-14)
    report = {
        "scope": __doc__,
        "geometry": geometry,
        "source_capture": str(args.capture),
        "source_snapshot": row["geometry_snapshot"],
        "physical_jaw_position_offset_m": 0.04,
        "contact_geom_indices": geom_indices.tolist(),
        "contact_geom_sol_params_before": geom_parameters_before.tolist(),
        "contact_geom_sol_params_after": geom_parameters_after.tolist(),
        "runtime_joint_armature_kg": array(robot.get_dofs_armature(joints)).reshape(2).tolist(),
        "runtime_joint_passive_damping_N_s_per_m": array(robot.get_dofs_damping(joints)).reshape(2).tolist(),
        "goal_completion": False,
        "limitations": manifest["protocol_difference_from_full_application"],
    }
    (args.output / "geometry-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.geometry_only:
        print(json.dumps(report), flush=True)
        return
    recovery = solver.stress_recovery
    entry, original = recovery.links[0], recovery.recover
    tick, substep_index, save, phase = 0, 0, False, "held_body_pd_closure"
    captures = []

    def observe(local_index):
        nonlocal substep_index
        original(local_index)
        physical_index = substep_index
        substep_index += 1
        if not save:
            return
        position = qd_to_numpy(entry.contacts.frame_position, copy=True).reshape(3)
        quaternion = qd_to_numpy(entry.contacts.frame_quaternion, copy=True).reshape(4)
        np.testing.assert_allclose(
            position, array(egg.base_link.get_pos(relative=False)).reshape(3), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            abs(quaternion @ array(egg.base_link.get_quat(relative=False)).reshape(4)), 1, rtol=0, atol=1e-12
        )
        collider = solver.collider.collider_state
        slots = qd_to_numpy(collider.contact_sort_idx, transpose=True, copy=True)[0]
        source = {
            key: qd_to_numpy(getattr(collider.contact_data, key), transpose=True, copy=True)[0]
            for key in ("pos", "force", "normal", "link_a", "link_b", "geom_a", "geom_b", "penetration", "sol_params")
        }
        valid = qd_to_numpy(entry.contacts.valid, transpose=True, copy=True)[0]
        mu = qd_to_numpy(entry.contacts.friction, transpose=True, copy=True)[0]
        contacts = []
        for index in np.flatnonzero(valid):
            slot = int(slots[index])
            a, b = int(source["link_a"][slot]), int(source["link_b"][slot])
            assert egg.base_link.idx in (a, b)
            sign = -1 if a == egg.base_link.idx else 1
            force, inward = sign * source["force"][slot], -sign * source["normal"][slot]
            normal = float(force @ inward)
            shear = float(np.linalg.norm(force - normal * inward))
            assert normal >= -1e-10 and shear <= mu[index] * normal + 1e-8
            contacts.append(
                {
                    "world_point_m": source["pos"][slot].tolist(),
                    "body_force_N": force.tolist(),
                    "body_inward_normal": inward.tolist(),
                    "friction": float(mu[index]),
                    "opposite_geom": int(source["geom_b" if a == egg.base_link.idx else "geom_a"][slot]),
                    "normal_force_N": normal,
                    "shear_force_N": shear,
                    "penetration_m": float(source["penetration"][slot]),
                    "mixed_contact_sol_params": source["sol_params"][slot].tolist(),
                }
            )
        frame = {
            "phase": phase,
            "tick": tick,
            "physical_substep": physical_index,
            "recovery_local_index": int(local_index),
            "frame": "constraint forces evaluated before rigid integration",
            "egg_position_m": position.tolist(),
            "egg_quaternion_wxyz": quaternion.tolist(),
            "egg_velocity_m_s": array(egg.base_link.get_vel(relative=False)).reshape(3).tolist(),
            "egg_angular_velocity_rad_s": array(egg.base_link.get_ang()).reshape(3).tolist(),
            "jaw_position_m": (0.04 + array(robot.get_dofs_position(joints)).reshape(2)).tolist(),
            "jaw_velocity_m_s": array(robot.get_dofs_velocity(joints)).reshape(2).tolist(),
            "jaw_control_force_N": array(robot.get_dofs_control_force(joints)).reshape(2).tolist(),
            "contacts": contacts,
        }
        captures.append(frame)
        last_tick = args.control_steps * round(args.target_hold_seconds / scene.dt) - 1
        final_tick = last_tick + round(args.release_hold_seconds / scene.dt)
        if tick in (last_tick, final_tick):
            name = f"{phase}-{tick:06d}-s{physical_index:02d}_geometry.npz"
            geometry_fields = {
                f"geom{geom.idx}_world_vertices_m": array(geom.get_verts()).reshape(-1, 3)
                for geom in (*robot.geoms, *egg.geoms)
            }
            np.savez_compressed(args.output / name, **geometry_fields)
            frame["geometry_snapshot"] = name
        (args.output / f"{phase}-{tick:06d}-s{physical_index:02d}.json").write_text(json.dumps(frame, indent=2) + "\n")

    recovery.recover = observe
    palm = robot.get_link("fixed_hand")
    solver.add_weld_constraint(egg.base_link.idx, palm.idx)
    assert np.count_nonzero(array(solver.get_weld_constraints()["link_a"]) == egg.base_link.idx) == 1
    assert solver.n_equalities == 0
    report["runtime_temporary_weld_sol_params"] = qd_to_numpy(
        solver.dyn_info.equalities.sol_params, transpose=True, copy=True
    )[0, 0].tolist()
    hold_steps = round(args.target_hold_seconds / scene.dt)
    for control_step in range(1, args.control_steps + 1):
        target = 0.04 + (0.018 - 0.04) * control_step / args.control_steps
        robot.control_dofs_position([target - 0.04, target - 0.04], joints)
        for inner in range(hold_steps):
            substep_index, save = 0, inner == hold_steps - 1
            scene.step()
            solver.check_errno()
            tick += 1
        print(json.dumps({"phase": phase, "target_step": control_step, "target_m": target}), flush=True)
    solver.delete_weld_constraint(egg.base_link.idx, palm.idx)
    assert np.count_nonzero(array(solver.get_weld_constraints()["link_a"]) == egg.base_link.idx) == 0
    phase = "released_free_body_fixed_hand"
    release_steps = round(args.release_hold_seconds / scene.dt)
    for inner in range(release_steps):
        substep_index, save = 0, inner % 25 == 24 or inner == release_steps - 1
        scene.step()
        solver.check_errno()
        tick += 1
    report["captures"] = captures
    report["limitations"] += (
        " The temporary weld is compliant and only initializes closure; it is removed before the final released phase. Gravity, translational/angular settling, time resolution and matched reference equilibrium must be audited before accepting a physical comparison."
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"captured_frames": len(captures), "final_frame": captures[-1], "goal_completion": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
