"""Independently audit matched rigid/deformable holds and replay their resolved body loads."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy.spatial.transform import Rotation

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .franka_contact_reference import build_from_capture
from .franka_mujoco_reference import PandaSkeleton
from .reference_motion_audit import load_initial_motion, motion_moments
from .summarize_fine_references import closed_source


def stencil_sources(system, collisions, potential, attributes, friction=False):
    sources = np.zeros((2, len(system.rest), 3))
    full_ids = system.mesh.to_full_vertex_id()
    rows = []
    for index in range(len(collisions)):
        collision = collisions[index]
        ids = np.array(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
        ids = ids[ids >= 0]
        nodes = full_ids[ids]
        groups = system.groups[nodes]
        assert set(groups) in ({0, 1}, {0, 2})
        finger = int(groups.max()) - 1
        dofs = collision.dof(attributes, system.mesh.edges, system.mesh.faces)
        force = -potential.gradient(collision, dofs).reshape(-1, 3)
        np.add.at(sources[finger], nodes, force)
        rows.append(
            {
                "stencil": index,
                "finger": finger,
                "weight_m2": float(collision.weight),
                "body_resultant_N": force[groups == 0].sum(axis=0).tolist(),
            }
        )
        if friction:
            bound = float(collision.mu_s * abs(collision.weight) * collision.normal_force_magnitude)
            magnitude = float(np.linalg.norm(force[groups == 0].sum(axis=0)))
            violation = magnitude - bound
            assert violation <= 1e-10 * max(bound, 1)
            rows[-1].update(coulomb_bound_N=bound, coulomb_violation_N=violation)
        del collision
    total = -system.mesh.to_full_dof(potential.gradient(collisions, system.mesh, attributes)).reshape(-1, 3)
    np.testing.assert_allclose(sources.sum(axis=0), total, rtol=1e-10, atol=1e-12)
    return sources, rows


def linear_quadrature(bulk, displacement):
    gradient = np.einsum("aie,ajeq->ijeq", displacement[bulk.dofs], bulk.gradients, optimize=True).transpose(2, 3, 0, 1)
    return bulk.mu * (gradient + gradient.swapaxes(-1, -2)) + bulk.lam * np.trace(gradient, axis1=-2, axis2=-1)[
        ..., None, None
    ] * np.eye(3)


def audit_case(run, rigid, steps, output):
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 0
    config = json.loads((run / "results/configuration.json").read_text())
    report = json.loads((run / "results/summary.json").read_text())
    assert len(report["steps"]) >= steps
    static, frame = build_from_capture(Path(config["capture"]), config["dhat"])
    body, permutation = static.body, static.parts[0].permutation
    n, dt = body.basis.N, config["dt"]
    mass = body.mass[permutation][:, permutation]
    gravity = static.gravity_force([0, 0, -9.81])
    robot = PandaSkeleton(Path(config["mjcf"]), dt)
    robot.model.opt.tolerance = report["robot_solver_tolerance"]
    kp, kv, target, caps = [
        np.array(report[k]) for k in ["controller_kp", "controller_kv", "controller_target", "controller_caps"]
    ]
    robot.set_state(np.r_[np.array(frame["robot_joint_position"])[:7], 0.04, 0.04], np.zeros(9))
    names, local = ["left_finger", "right_finger"], []
    for name, nodes in zip(names, static.finger_nodes):
        index = robot.body_ids[name]
        local.append(
            (static.rest[nodes] + static.world_coordinate_origin_m - robot.data.xpos[index])
            @ robot.data.xmat[index].reshape(3, 3)
        )
    with np.load(run / "results/closure_state.npz") as saved:
        previous, previous_q = saved["full_displacement"].copy(), saved["joint_position"].copy()
    previous_body_velocity, previous_joint_velocity, initial_motion = load_initial_motion(
        run, config, frame, static, previous
    )
    previous_motion = initial_motion
    previous_vector = np.empty(n)
    previous_vector[permutation] = previous[:n]
    previous_rotation = finite_frame(body, previous_vector)[0]
    friction = ipctk.FrictionPotential(dt * 4e-6)
    radius = float(np.linalg.norm(static.rest[: n // 3] - body.com, axis=1).max())
    rows, observed = [], []
    for step in range(1, steps + 1):
        with np.load(run / f"results/step{step:04d}.npz") as saved:
            u, q, joint_velocity, force = [
                saved[k].copy() for k in ["displacement", "joint_position", "joint_velocity", "contact_force_N"]
            ]
            body_velocity = (u[:n] - previous[:n]) / dt
            acceleration = (body_velocity - previous_body_velocity) / dt
            np.testing.assert_array_equal(body_velocity, saved["body_velocity"])
            np.testing.assert_array_equal(acceleration, saved["body_acceleration"])
            np.testing.assert_array_equal(saved["body_component_permutation"], permutation)
            np.testing.assert_array_equal(saved["applied_force_N"][:n], gravity[:n])
        np.testing.assert_allclose(q, previous_q + np.longdouble(dt) * joint_velocity, rtol=0, atol=1e-15)
        normal, current = static.normal_collisions(u)
        tangential = ipctk.TangentialCollisions()
        tangential.build(static.mesh, current, normal, static.barrier, static.friction)
        delta = static.mesh.vertices((u - previous).reshape(-1, 3))
        normal_sources, normal_rows = stencil_sources(static, normal, static.barrier, current)
        shear_sources, shear_rows = stencil_sources(static, tangential, friction, delta, friction=True)
        sources = normal_sources + shear_sources
        np.testing.assert_allclose(sources.sum(axis=0).reshape(-1), force, rtol=1e-10, atol=1e-12)
        vector = np.empty(n)
        vector[permutation] = u[:n]
        rotation, shift, objective_u = finite_frame(body, vector)
        centre = rotation @ body.com + shift
        positions = static.rest + u.reshape(-1, 3)
        motion = motion_moments(body, permutation, positions[: n // 3], body_velocity)
        inertial = mass @ acceleration
        balance = inertial - gravity[:n] - force[:n]
        resultant = balance.reshape(-1, 3).sum(axis=0)
        moment = np.cross(positions[: n // 3] - centre, balance.reshape(-1, 3)).sum(axis=0)
        if rigid:
            assert np.linalg.norm(objective_u[body.component_dofs], axis=1).max() <= 1e-12
            assert np.linalg.norm(resultant) <= 1e-9 and np.linalg.norm(moment) / radius <= 1e-9
            body_residual = float(np.hypot(np.linalg.norm(resultant), np.linalg.norm(moment) / radius))
        else:
            gradient = static.bulk.evaluate(vector, False)[1][permutation]
            body_residual = float(np.linalg.norm(gradient + balance))
            assert body_residual <= 1e-9
        baseline, response = robot.velocity_response(previous_q, previous_joint_velocity)
        inverse = np.linalg.inv(response)
        robot.set_state(q, joint_velocity)
        joint_contact = np.zeros(9)
        for name, nodes, local_points in zip(names, static.finger_nodes, local):
            index = robot.body_ids[name]
            expected = local_points @ robot.data.xmat[index].reshape(3, 3).T + robot.data.xpos[index]
            np.testing.assert_allclose(
                positions[nodes] + static.world_coordinate_origin_m, expected, rtol=0, atol=1e-12
            )
            point_force = force.reshape(-1, 3)[nodes]
            origin = robot.data.xpos[index] - static.world_coordinate_origin_m
            wrench = np.r_[point_force.sum(axis=0), np.cross(positions[nodes] - origin, point_force).sum(axis=0)]
            joint_contact += robot.jacobian(name).T @ wrench
        pd = np.asarray(np.clip(kp * (target - q) - kv * joint_velocity, -caps, caps), dtype=float)
        joint_error = float(np.linalg.norm(inverse @ (joint_velocity - baseline) - pd - joint_contact))
        fresh = float(
            np.linalg.norm(
                inverse
                @ (joint_velocity - robot.step_velocity(previous_q, previous_joint_velocity, pd + joint_contact))
            )
        )
        assert joint_error <= 1e-9 and fresh <= 1e-9
        material_contact, material_rhs = np.zeros(n), np.zeros(n)
        material_contact[body.component_dofs] = force[:n].reshape(-1, 3) @ rotation
        material_rhs[body.component_dofs] = (force[:n] + gravity[:n] - inertial).reshape(-1, 3) @ rotation
        full_replay = body.solve(material_rhs)
        assert full_replay.residual_n <= max(1e-11, 1e-8 * np.linalg.norm(full_replay.rhs))
        omega_world = Rotation.from_matrix(rotation @ previous_rotation.T).as_rotvec() / dt
        omega_material = rotation.T @ omega_world
        rigid_relief = body.solve(material_contact, omega_material)
        replay_tensor = linear_quadrature(static.bulk, full_replay.displacement)
        reference_tensor, nonlinear_energy, peak_lower, peak_upper = None, None, None, None
        if not rigid:
            cauchy, nonlinear_energy = quadrature_fields(static.bulk, vector)
            reference_tensor = rotation.T @ cauchy @ rotation
            _, vm, upper = static.bulk.stress_bounds(vector)
            peak_lower, peak_upper = float(vm.max()), float(upper.max())
        patch_wrenches = []
        for source in sources:
            body_force = source[: n // 3]
            patch_wrenches.append(
                np.r_[
                    body_force.sum(axis=0) @ rotation,
                    np.cross(positions[: n // 3] - centre, body_force).sum(axis=0) @ rotation,
                ]
            )
        row = {
            "step": step,
            "time_s": step * dt,
            "body_residual_N_or_force_equivalent_N": body_residual,
            "body_resultant_residual_N": float(np.linalg.norm(resultant)),
            "body_moment_residual_Nm": float(np.linalg.norm(moment)),
            "joint_effort_residual": joint_error,
            "fresh_step_effort_residual": fresh,
            "raw_replayed_peak_Pa": float(full_replay.von_mises_pa.max()),
            "replayed_energy_J": full_replay.energy_j,
            "nonlinear_body_energy_J": nonlinear_energy,
            "nonlinear_raw_peak_lower_Pa": peak_lower,
            "nonlinear_raw_peak_upper_Pa": peak_upper,
            "full_rhs_balance_correction_norm_N": float(np.linalg.norm(full_replay.rhs - material_rhs)),
            "patch_wrenches_material_N_Nm": np.array(patch_wrenches).tolist(),
            "normal_stencils": normal_rows,
            "shear_stencils": shear_rows,
            "contact_total_force_residual_N": float(np.linalg.norm(force.reshape(-1, 3).sum(axis=0))),
            "contact_total_moment_about_body_com_Nm": np.cross(positions - centre, force.reshape(-1, 3))
            .sum(axis=0)
            .tolist(),
            "body_linear_velocity_world_m_s": motion["linear_velocity_world_m_s"].tolist(),
            "body_angular_velocity_world_rad_s": motion["angular_velocity_world_rad_s"].tolist(),
            "body_angular_momentum_about_com_world_Nms": motion["angular_momentum_about_com_world_Nms"].tolist(),
            "body_kinetic_energy_J": motion["kinetic_energy_J"],
            "body_contact_moment_world_Nm": np.cross(positions[: n // 3] - centre, force[:n].reshape(-1, 3))
            .sum(axis=0)
            .tolist(),
        }
        old_positions = static.rest[: n // 3] + previous[:n].reshape(-1, 3)
        relative_increment = (positions[: n // 3] - centre - old_positions + previous_motion["centre_m"]) / dt
        transport = np.cross(relative_increment, (mass @ previous_body_velocity).reshape(-1, 3)).sum(axis=0)
        angular_rate = (
            motion["angular_momentum_about_com_world_Nms"] - previous_motion["angular_momentum_about_com_world_Nms"]
        ) / dt
        gravity_moment = np.cross(positions[: n // 3] - centre, gravity[:n].reshape(-1, 3)).sum(axis=0)
        row["angular_momentum_transport_world_Nm"] = transport.tolist()
        row["angular_momentum_balance_residual_Nm"] = float(
            np.linalg.norm(angular_rate - np.array(row["body_contact_moment_world_Nm"]) - gravity_moment - transport)
        )
        rows.append(row)
        observed.append(
            {
                "tensor": replay_tensor,
                "reference_tensor": reference_tensor,
                "patch_wrenches": np.array(patch_wrenches),
                "energy": full_replay.energy_j,
                "peak": float(full_replay.von_mises_pa.max()),
                "nonlinear_energy_J": nonlinear_energy,
                "nonlinear_raw_peak_lower_Pa": peak_lower,
            }
        )
        np.savez_compressed(
            output / f"step{step:04d}.npz",
            body_mesh_vertices=body.mesh.p.T,
            body_mesh_tetrahedra=body.mesh.t.T,
            material_contact_force_N=material_contact,
            material_full_inertia_rhs_N=material_rhs,
            omega_material_rad_s=omega_material,
            full_inertia_linear_displacement_m=full_replay.displacement,
            full_inertia_linear_tensor_Pa=full_replay.tensor_pa,
            rigid_relief_linear_tensor_Pa=rigid_relief.tensor_pa,
            objective_replay_quadrature_tensor_Pa=replay_tensor,
            rotation_world_from_material=rotation,
            patch_wrenches_material_N_Nm=np.array(patch_wrenches),
            **({"objective_nonlinear_quadrature_tensor_Pa": reference_tensor} if reference_tensor is not None else {}),
        )
        previous, previous_q = u, q
        previous_body_velocity, previous_joint_velocity, previous_rotation = body_velocity, joint_velocity, rotation
        previous_motion = motion
    return (
        {
            "run": str(run),
            "rigid": rigid,
            "steps_checked": steps,
            "initial_motion": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in initial_motion.items()},
            "rows": rows,
        },
        observed,
        static.bulk.basis.dx,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid-run", type=Path, required=True)
    parser.add_argument("--deformable-run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [args.rigid_run, args.deformable_run]
    configs = [json.loads((r / "results/configuration.json").read_text()) for r in runs]
    for key in ["capture", "dt", "dhat", "control_steps"]:
        assert configs[0][key] == configs[1][key], key
    assert configs[0].get("captured_initial_velocity", False) == configs[1].get("captured_initial_velocity", False)
    assert not configs[1].get("fixed_arm", False)
    assert len({hashlib.sha256(Path(c["mjcf"]).read_bytes()).hexdigest() for c in configs}) == 1
    reports = [json.loads((r / "results/summary.json").read_text()) for r in runs]
    for key in ["controller_target", "controller_kp", "controller_kv", "controller_caps", "robot_solver_tolerance"]:
        np.testing.assert_array_equal(reports[0][key], reports[1][key])
    cases, histories, weights = [], [], []
    for index, run in enumerate(runs):
        output = args.output / f"case{index:02d}"
        output.mkdir()
        case, history, dx = audit_case(run, index == 0, args.steps, output)
        cases.append(case)
        histories.append(history)
        weights.append(dx)
    np.testing.assert_array_equal(weights[0], weights[1])

    def norm(tensor):
        return float(np.sqrt(np.sum(weights[0] * np.sum(tensor * tensor, axis=(-2, -1)))))

    comparisons = []
    for index, (rigid, deformable) in enumerate(zip(*histories)):
        row = {
            "step": index + 1,
            "time_s": (index + 1) * configs[0]["dt"],
            "resolved_load_feedback_tensor_relative_change": norm(rigid["tensor"] - deformable["tensor"])
            / norm(deformable["tensor"]),
            "deformable_linear_replay_vs_nonlinear_tensor_relative_change": norm(
                deformable["tensor"] - deformable["reference_tensor"]
            )
            / norm(deformable["reference_tensor"]),
            "resolved_load_feedback_peak_relative_change": abs(rigid["peak"] / deformable["peak"] - 1),
            "resolved_load_feedback_energy_relative_change": abs(rigid["energy"] / deformable["energy"] - 1),
        }
        for label, section in [("force_N", slice(0, 3)), ("moment_Nm", slice(3, 6))]:
            absolute = float(
                np.linalg.norm(rigid["patch_wrenches"][:, section] - deformable["patch_wrenches"][:, section])
            )
            row["patch_" + label + "_absolute_change"] = absolute
            row["patch_" + label + "_relative_change"] = absolute / float(
                np.linalg.norm(deformable["patch_wrenches"][:, section])
            )
        comparisons.append(row)
    summary = {
        "scope": __doc__,
        "cases": cases,
        "comparisons": comparisons,
        "goal_completion": False,
        "limitations": "The same geometry/control/contact/mass with rigid versus deformable body motion isolates feedback within the declared short held-initializer protocol. Initial equilibria are solved separately under the same supports/targets, not artificially forced to have identical loads. Linear full-inertia replay separates resolved-load changes from nonlinear constitutive effects. Native recovery, finite-footprint mapping, original approach/angular history, spatial/contact/regularization tails and physical accuracy/cost remain unverified here.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"steps_checked_per_case": args.steps, "final_comparison": comparisons[-1]}))


if __name__ == "__main__":
    main()
