"""Implicit inertial contact pilot for the objective egg and actual rigid Panda jaws."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .franka_contact_reference import RigidFingerContactSystem, build_from_capture
from .franka_rigid_mode_coordinates import coordinate_map
from .refine_franka_reference import refine_body
from .summarize_fine_references import closed_source


class InertialRigidFingerContactSystem(RigidFingerContactSystem):
    """Add consistent inertia, inherited armature/passive damping and clipped PD.

    Backward Euler uses M*(u-u_previous-dt*v_previous)/dt**2. Finger PD
    uses the new joint velocity, independently clipped to the original4N cap;
    the inherited1N*s/m passive damping remains outside that actuator cap.
    """

    def __init__(self, static, dt):
        self.__dict__.update(static.__dict__)
        self.dt = dt
        assert dt > 0
        self.friction_mollifier_velocity_m_s = 4e-6
        self.friction_potential = ipctk.FrictionPotential(dt * self.friction_mollifier_velocity_m_s)
        self.kv, self.passive_damping, self.armature = 10.0, 1.0, 0.1
        count = self.body.basis.N
        mass = sparse.lil_matrix(self.stiffness.shape)
        permutation = self.parts[0].permutation
        mass[:count, :count] = self.body.mass[permutation][:, permutation]
        for node, axis, physical_mass in zip(self.finger_com_nodes, self.axes, self.masses, strict=True):
            dofs = np.arange(3 * node, 3 * node + 3)
            mass[np.ix_(dofs, dofs)] = physical_mass * np.eye(3) + self.armature * np.outer(axis, axis)
        self.inertial_mass = mass.tocsc()
        self.predicted_displacement = np.zeros(self.stiffness.shape[0])

    @staticmethod
    def clipped_primitive(raw_force, slope, cap):
        return (0.5 * raw_force**2 if abs(raw_force) <= cap else cap * abs(raw_force) - 0.5 * cap**2) / slope

    def evaluate(self, displacement, previous, tangential, hessian, project_hessian=False, external_force=None):
        energy, gradient, matrix, normal, contact_gradient = super().evaluate(
            displacement, previous, tangential, hessian, project_hessian, external_force
        )
        increment = displacement - self.predicted_displacement
        inertial_gradient = self.inertial_mass @ increment / self.dt**2
        energy += 0.5 * increment @ inertial_gradient
        gradient += inertial_gradient
        correction = sparse.lil_matrix(self.stiffness.shape) if hessian else None
        slope = self.kp + self.kv / self.dt
        for index, node in enumerate(self.finger_com_nodes):
            dofs = np.arange(3 * node, 3 * node + 3)
            axis = self.axes[index]
            q = self.q_reference[index] + displacement[dofs] @ axis
            dq = (displacement[dofs] - previous[dofs]) @ axis
            old_raw = self.kp * (q - self.q_target[index])
            new_raw = old_raw + self.kv * dq / self.dt
            old_force = np.clip(old_raw, -self.force_cap, self.force_cap)
            new_force = np.clip(new_raw, -self.force_cap, self.force_cap)
            gradient[dofs] += (new_force - old_force + self.passive_damping * dq / self.dt) * axis
            energy += self.clipped_primitive(new_raw, slope, self.force_cap)
            energy -= self.clipped_primitive(old_raw, self.kp, self.force_cap)
            energy += 0.5 * self.passive_damping / self.dt * dq**2
            if hessian:
                delta = slope * (abs(new_raw) < self.force_cap) - self.kp * (abs(old_raw) < self.force_cap)
                delta += self.passive_damping / self.dt
                correction[np.ix_(dofs, dofs)] = delta * np.outer(axis, axis)
        if hessian:
            matrix += self.inertial_mass / self.dt**2 + correction.tocsc()
        return energy, gradient, matrix, normal, contact_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-run", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--body-refinement", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--cold-closure", action="store_true")
    parser.add_argument("--control-steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.dt > 0 and args.steps > 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    closed_source(args.closure_run)
    config = json.loads((args.closure_run / "results/configuration.json").read_text())
    static, _ = build_from_capture(Path(config["capture"]), config["dhat"])
    assert args.cold_closure or args.body_refinement == 0
    static = refine_body(static, args.body_refinement)
    if args.cold_closure:
        u = np.zeros(static.stiffness.shape[0])
        held_map, held = static.coordinate_map(True)
        gravity = static.gravity_force([0, 0, -9.81])
        for control_step in range(1, args.control_steps + 1):
            static.q_target[:] = 0.04 + (0.018 - 0.04) * control_step / args.control_steps
            (args.output / "closure_progress.json").write_text(
                json.dumps(
                    {
                        "step": control_step,
                        "target_m": static.q_target.tolist(),
                        "body_tetrahedra": static.body.mesh.nelements,
                        "fixed_points_m": static.fixed_initializer_points_m.tolist(),
                        "fixed_components": static.fixed_initializer_components.tolist(),
                    }
                )
                + "\n"
            )
            u, g, _f, _energy, _iterations = static.solve_updated_friction(
                held,
                np.zeros(len(held)),
                1,
                initial_displacement=u,
                coordinate_map=held_map,
                applied_force=gravity,
                initial_force=gravity,
                phase="same_fixed_points_cold_closure",
                failure_path=args.output / "closure_failure.npz",
            )
            print(
                json.dumps(
                    {
                        "cold_closure_target_step": control_step,
                        "complete_held_residual_N": float(np.linalg.norm(held_map.T @ g)),
                    }
                ),
                flush=True,
            )
        np.savez_compressed(
            args.output / "closure_state.npz",
            displacement=u,
            fixed_points_m=static.fixed_initializer_points_m,
            fixed_components=static.fixed_initializer_components,
        )
    else:
        with np.load(args.closure_run / "results/force_failure.npz") as failed:
            np.testing.assert_array_equal(failed["rest_positions"], static.rest)
            u = failed["previous_displacement"].copy()
    system = InertialRigidFingerContactSystem(static, args.dt)
    system.q_target[:] = 0.018
    mapping = coordinate_map(system)
    external = system.gravity_force([0, 0, -9.81])
    velocity = np.zeros_like(u)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    np.savez_compressed(
        args.output / "reference_geometry.npz",
        vertices=system.body.mesh.p.T,
        tetrahedra=system.body.mesh.t.T,
        fixed_initializer_points_m=static.fixed_initializer_points_m,
        fixed_initializer_components=static.fixed_initializer_components,
    )
    records = []
    for step in range(1, args.steps + 1):
        previous, old_velocity = u.copy(), velocity.copy()
        system.predicted_displacement = previous + args.dt * old_velocity
        u, g, f, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            initial_displacement=previous,
            coordinate_map=mapping,
            applied_force=external,
            initial_force=external,
            phase=f"inertial_release_step{step}",
            failure_path=args.output / "failure.npz",
        )
        velocity = (u - previous) / args.dt
        acceleration = (velocity - old_velocity) / args.dt
        complete = float(np.linalg.norm(mapping.T @ g))
        body = float(np.linalg.norm(g[: system.body.basis.N]))
        assert complete <= 1e-9 and body <= 1e-9
        static_residual = g - system.inertial_mass @ acceleration
        row = {
            "step": step,
            "time_s": step * args.dt,
            "complete_dynamic_residual_N": complete,
            "complete_body_dynamic_residual_N": body,
            "body_force_residual_if_inertia_is_omitted_N": float(
                np.linalg.norm(static_residual[: system.body.basis.N])
            ),
        }
        records.append(row)
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=u,
            velocity=velocity,
            acceleration=acceleration,
            contact_force_N=f,
            applied_force_N=external,
            body_component_permutation=system.parts[0].permutation,
        )
        print(json.dumps(row), flush=True)
    vector = np.empty(system.body.basis.N)
    vector[system.parts[0].permutation] = u[: system.body.basis.N]
    tensor, vm, upper = system.bulk.stress_bounds(vector)
    np.savez_compressed(args.output / "body_fields.npz", tensor_Pa=tensor, vm_Pa=vm, upper_peak_Pa=upper)
    report = {
        "scope": __doc__,
        "body_tetrahedra": system.body.mesh.nelements,
        "body_refinement_levels": args.body_refinement,
        "cold_closure": args.cold_closure,
        "steps": records,
        "body_peak_lower_Pa": float(vm.max()),
        "body_peak_upper_Pa": float(upper.max()),
        "whole_volume_j_lower_bound": system.bulk.determinant_lower_bound(vector),
        "friction_mollifier_velocity_m_s": system.friction_mollifier_velocity_m_s,
        "goal_completion": False,
        "limitations": "This inertial pilot starts from the independently completed held-row closure at zero velocity. That initializer differs from the rigid compliant weld. Contact barrier/friction regularization and integration method also differ; no matched application, temporal/spatial/contact convergence, omitted-feedback accuracy or physical operating range is claimed. All body and mapped force gates include consistent mass acceleration; a nonzero static residual is not accepted as a static solve.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
