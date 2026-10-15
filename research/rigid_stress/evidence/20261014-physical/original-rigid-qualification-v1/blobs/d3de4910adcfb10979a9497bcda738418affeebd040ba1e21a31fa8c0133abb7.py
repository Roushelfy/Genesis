"""Solve the same original ground-supported protocol with a six-coordinate rigid egg and identical contact law."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import build_original, contact_groups
from .rigid_egg_articulated_contact import RigidEggArticulatedContact
from .run_rigid_egg_franka import qualify_kinematics, qualify_tangent
from .summarize_fine_references import closed_source


class OriginalRigidEggContact(RigidEggArticulatedContact):
    def kinematics(self, q):
        jacobian, positions = super().kinematics(q)
        ground = self.contact.rest[self.contact.ground_nodes]
        return np.vstack((jacobian, np.zeros((3 * len(ground), 9)))), np.vstack((positions, ground))

    def evaluate(self, body_u, variable, previous, hessian):
        evaluated = super().evaluate(body_u, variable, previous, hessian)
        if not self.dynamic:
            return evaluated
        q = self.previous_q + np.longdouble(self.dt) * variable
        jacobian, _ = self.kinematics(q)
        effort = self.pd(q, variable)[0] - jacobian.T @ evaluated[3][self.n :]
        actual = self.robot.step_velocity(self.previous_q, self.previous_velocity, effort)
        values = list(evaluated)
        values[1] = self.inverse_response @ (variable - actual)
        if hessian:
            response = np.empty((9, 9))
            for column in range(9):
                delta = np.eye(9)[column] * 1e-6
                plus = self.robot.step_velocity(self.previous_q, self.previous_velocity, effort + delta)
                minus = self.robot.step_velocity(self.previous_q, self.previous_velocity, effort - delta)
                response[:, column] = (plus - minus) / 2e-6
            bb, bq, qb, qq = evaluated[6]
            chain = self.inverse_response @ response
            values[6] = (bb, bq, chain @ qb, self.inverse_response + chain @ (qq - self.inverse_response))
        return tuple(values)

    def evaluate_rigid(self, pose, variable, previous, hessian):
        body_u, body_jacobian = self.body_kinematics(pose)
        rb, rj, full, contact, energy, stencils, blocks = self.evaluate(body_u, variable, previous, hessian)
        body = (body_jacobian.T @ rb) * self.body_row_scale
        if not hessian:
            return np.asarray(body, dtype=float), np.asarray(rj, dtype=float), full, contact, energy, stencils, None
        bb, bq, qb, qq = blocks
        reduced = body_jacobian.T @ (bb @ body_jacobian)
        for column in range(3, 6):
            direction = np.zeros(6)
            direction[column] = 1e-6
            plus = self.body_kinematics(pose + direction)[1].T @ rb
            minus = self.body_kinematics(pose - direction)[1].T @ rb
            reduced[:, column] += (plus - minus) / 2e-6
        projected = (
            self.body_row_scale[:, None] * reduced,
            self.body_row_scale[:, None] * (body_jacobian.T @ bq),
            qb @ body_jacobian,
            qq,
        )
        return (
            np.asarray(body, dtype=float),
            np.asarray(rj, dtype=float),
            full,
            contact,
            energy,
            stencils,
            tuple(np.asarray(block, dtype=float) for block in projected),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-deformable", type=Path, required=True)
    parser.add_argument("--matched-audit", type=Path, required=True)
    parser.add_argument("--macro-steps", type=int, default=110)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--check-jacobian", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.matched_deformable)
    assert int((args.matched_deformable / "returncode.txt").read_text()) == 0
    closed_source(args.matched_audit)
    assert int((args.matched_audit / "returncode.txt").read_text()) == 0
    matched = json.loads((args.matched_audit / "results/summary.json").read_text())
    assert Path(matched["run"]).resolve() == args.matched_deformable.resolve()
    assert matched["complete_requested_history"] and matched["steps_checked"] == len(matched["rows"]) == 2640
    assert all(
        row["body_dynamic_residual_N"] <= 1e-9 and row["actual_robot_step_effort_residual"] <= 1e-9
        for row in matched["rows"]
    )
    parent_path = args.matched_deformable / "results/configuration.json"
    parent = json.loads(parent_path.read_text())
    assert hashlib.sha256(args.mjcf.read_bytes()).digest() == hashlib.sha256(Path(parent["mjcf"]).read_bytes()).digest()
    static, frame, capture, protocol = build_original(
        Path(parent["capture"]), Path(parent["protocol_run"]), args.mjcf, parent["dt"], parent["dhat"]
    )
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    assert 0 < args.macro_steps <= len(controls) == parent["macro_steps"] == 330
    substeps = round(protocol["macro_dt_s"] / parent["dt"])
    assert abs(substeps * parent["dt"] - protocol["macro_dt_s"]) < 1e-14
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args),
        **{key: parent[key] for key in ["capture", "protocol_run", "dt", "dhat"]},
        "physical_contact_law_matched": True,
    }
    (args.output / "configuration.json").write_text(json.dumps(configuration, default=str, indent=2) + "\n")
    system = OriginalRigidEggContact(static, frame, capture, args.mjcf, parent["dt"])
    assert not system.fixed_arm
    pose = np.zeros(6, dtype=np.longdouble)
    q = np.asarray(protocol["initial_state"]["joint_position"], dtype=np.longdouble)
    velocity = np.asarray(protocol["initial_state"]["joint_velocity"], dtype=np.longdouble)
    np.testing.assert_array_equal(q, system.initial_q)
    with np.load(args.matched_deformable / "results/initial_state.npz") as saved:
        previous = saved["full_displacement"].copy()
        body_velocity = saved["body_velocity"].copy()
        np.testing.assert_array_equal(saved["joint_position"], q)
        np.testing.assert_array_equal(saved["joint_velocity"], velocity)
    np.testing.assert_array_equal(system.pack(system.body_kinematics(pose)[0], q)[0], previous)
    np.testing.assert_array_equal(body_velocity, np.zeros(system.n))
    np.savez_compressed(
        args.output / "initial_state.npz",
        body_pose=pose,
        full_displacement=previous,
        joint_position=q,
        joint_velocity=velocity,
        body_velocity=body_velocity,
    )
    geometry = args.matched_deformable / "results/reference_geometry.npz"
    (args.output / "reference_geometry.npz").write_bytes(geometry.read_bytes())
    rows, tangent_checks = [], []
    finger_tangent_checked = False
    kinematic_error = qualify_kinematics(system)
    for macro in range(args.macro_steps):
        control = controls[macro]
        assert control["tick"] == macro
        system.target = np.asarray(control["target"], dtype=float)
        system.caps[7:] = control["finger_cap_N"]
        for micro in range(substeps):
            step = macro * substeps + micro + 1
            old_full, old_velocity, old_body_velocity = previous.copy(), velocity.copy(), body_velocity.copy()
            system.begin_dynamic_step(q, velocity, previous[: system.n] + parent["dt"] * body_velocity)
            if args.check_jacobian and step == 1:
                tangent_checks.append(
                    {"phase": "original_start", "step": step, "rows": qualify_tangent(system, pose, velocity, previous)}
                )
            try:
                pose, velocity, full, contact = system.solve_rigid(
                    pose, velocity, previous, False, f"original_rigid{step:05d}"
                )
                fresh = system.verify_robot_step(velocity, contact)
            except Exception:
                np.savez_compressed(
                    args.output / "failure.npz",
                    **system.failure_state,
                    previous_full_displacement=old_full,
                    previous_body_velocity=old_body_velocity,
                    previous_joint_velocity=old_velocity,
                )
                raise
            q = system.previous_q + np.longdouble(parent["dt"]) * velocity
            body_velocity = (full[: system.n] - previous[: system.n]) / parent["dt"]
            acceleration = (body_velocity - old_body_velocity) / parent["dt"]
            body_balance = system.contact.body_mass @ acceleration - system.gravity[: system.n] + contact[: system.n]
            jacobian = system.body_kinematics(pose)[1]
            body_error = float(np.linalg.norm((jacobian.T @ body_balance) * system.body_row_scale))
            assert body_error <= 1e-9 and fresh <= 1e-9
            groups, ground = contact_groups(system, full)
            np.savez_compressed(
                args.output / f"step{step:05d}.npz",
                body_pose=pose,
                displacement=full,
                joint_position=q,
                joint_velocity=velocity,
                body_velocity=body_velocity,
                body_acceleration=acceleration,
                contact_force_N=-contact,
                body_component_permutation=static.parts[0].permutation,
                applied_force_N=system.gravity,
            )
            rows.append(
                {
                    "step": step,
                    "macro_tick": macro,
                    "time_s": step * parent["dt"],
                    "body_force_equivalent_residual_N": body_error,
                    "fresh_actual_robot_effort_residual": fresh,
                    "contact_groups": groups,
                    "ground_contact_points_local_m": ground,
                }
            )
            if args.check_jacobian and not finger_tangent_checked and any(group in (1, 2) for group in groups):
                tangent_checks.append(
                    {
                        "phase": "first_finger_contact",
                        "step": step,
                        "rows": qualify_tangent(system, pose, velocity, old_full),
                    }
                )
                finger_tangent_checked = True
            previous = full.copy()
            print(json.dumps(rows[-1]), flush=True)
    result = {
        "scope": __doc__,
        "matched_deformable": str(args.matched_deformable),
        "configuration_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "requested_macro_steps": args.macro_steps,
        "available_original_macro_steps": len(controls),
        "steps": rows,
        "complete_original_330_macro_history": args.macro_steps == 330 and len(rows) == 2640,
        "kinematic_qualification_relative_error": kinematic_error,
        "tangent_qualifications": tangent_checks,
        "reference_geometry_byte_identical": True,
        "body_mass_kg": float(static.body.total_mass),
        "goal_completion": False,
        "limitations": "The rigid egg uses six finite-rotation coordinates and the same consistent mass, original geometry, ground, full Panda model, actuation/control targets, contact law/friction/regularization and physical timestep as the independently solved deformable history. Fresh original robot-step and rigid-body force/moment gates remain1e-9. A matching independent all-state audit is required; no spatial/contact/time convergence, native finite-footprint accuracy or warmed cost follows from this counterpart. Body equilibrium is the rigid six-mode constrained balance, not a fictitious zero elastic nodal residual.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
