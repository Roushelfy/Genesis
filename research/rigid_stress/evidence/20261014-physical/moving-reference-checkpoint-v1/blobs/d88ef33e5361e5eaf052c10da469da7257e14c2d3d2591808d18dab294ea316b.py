"""Solve a declared prefix of the actual original controls from zero velocity and real ground support."""

import argparse
import json
from pathlib import Path

import numpy as np

from .original_franka_ground import GroundedOriginalContact, build_original, contact_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol-run", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--macro-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    static, frame, report, protocol = build_original(args.capture, args.protocol_run, args.mjcf, args.dt, args.dhat)
    controls = [row for row in protocol["records"] if row["mode"] == "approach_lift"]
    assert 0 < args.macro_steps <= len(controls)
    substeps = round(protocol["macro_dt_s"] / args.dt)
    assert abs(substeps * args.dt - protocol["macro_dt_s"]) < 1e-14
    system = GroundedOriginalContact(static, frame, report, args.mjcf, args.dt)
    q = system.initial_q.copy()
    body_u, body_velocity, velocity = np.zeros(system.n, dtype=np.longdouble), np.zeros(system.n), np.zeros(9)
    previous = system.pack(body_u, q)[0]
    np.savez_compressed(
        args.output / "initial_state.npz",
        body_displacement=body_u,
        joint_position=q,
        body_velocity=body_velocity,
        joint_velocity=velocity,
        full_displacement=previous,
    )
    np.savez_compressed(
        args.output / "reference_geometry.npz",
        vertices=static.body.mesh.p.T,
        tetrahedra=static.body.mesh.t.T,
        rest_positions=static.rest,
        faces=static.faces,
        groups=static.groups,
        ground_nodes=static.ground_nodes,
        body_component_permutation=static.parts[0].permutation,
        world_coordinate_origin_m=static.world_coordinate_origin_m,
    )
    rows = []
    for macro in range(args.macro_steps):
        control = controls[macro]
        assert control["tick"] == macro
        system.target[:] = control["target"]
        system.caps[7:] = control["finger_cap_N"]
        for micro in range(substeps):
            old_body, old_q = body_u.copy(), q.copy()
            old_body_velocity, old_velocity = body_velocity.copy(), velocity.copy()
            system.begin_dynamic_step(old_q, old_velocity, old_body + args.dt * old_body_velocity)
            step = macro * substeps + micro + 1
            try:
                body_u, velocity, full, contact = system.solve(body_u, velocity, previous, False, f"original{step:05d}")
                fresh = system.verify_robot_step(velocity, contact)
                groups, ground_points = contact_groups(system, full)
            except Exception:
                np.savez_compressed(
                    args.output / "failure.npz",
                    **system.failure_state,
                    previous_full_displacement=previous,
                    previous_joint_position=old_q,
                    previous_joint_velocity=old_velocity,
                    previous_body_displacement=old_body,
                    previous_body_velocity=old_body_velocity,
                    target=system.target,
                    caps=system.caps,
                )
                raise
            q = system.previous_q + np.longdouble(args.dt) * velocity
            body_velocity = (body_u - old_body) / args.dt
            row = {
                "step": step,
                "macro_tick": macro,
                "time_s": step * args.dt,
                "fresh_robot_step_effort_residual": fresh,
                "contact_groups": groups,
                "ground_contact_points_local_m": ground_points,
                **system.history[-1],
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"step{step:05d}.npz",
                displacement=full,
                joint_position=q,
                joint_velocity=velocity,
                body_velocity=body_velocity,
                body_acceleration=(body_velocity - old_body_velocity) / args.dt,
                contact_force_N=-contact,
                applied_force_N=system.gravity,
                controller_target=system.target,
                controller_caps=system.caps,
            )
            previous = full.copy()
            print(json.dumps(row, default=float), flush=True)
    vector = np.empty(system.n)
    vector[static.parts[0].permutation] = body_u
    tensor, vm, upper = system.contact.bulk.stress_bounds(vector)
    np.savez_compressed(args.output / "body_fields.npz", tensor_Pa=tensor, vm_Pa=vm, upper_peak_Pa=upper)
    summary = {
        "scope": __doc__,
        "steps": rows,
        "original_macro_steps_solved": args.macro_steps,
        "available_original_macro_steps": len(controls),
        "raw_global_peak_lower_Pa": float(vm.max()),
        "raw_global_peak_upper_Pa": float(upper.max()),
        "whole_volume_j_lower_bound": system.contact.bulk.determinant_lower_bound(vector),
        "goal_completion": False,
        "limitations": "This is an original ground-supported history prefix, not the complete grasp/lift. Complete independent force/state/control audits, temporal and spatial/contact convergence, corresponding rigid feedback, native models and verified accuracy/cost remain required. The ground triangle is used only where accepted contact points are strictly interior; non-body contact pairs would be rejected.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
