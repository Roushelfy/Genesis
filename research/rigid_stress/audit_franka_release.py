"""Audit retained incomplete Panda closure/release states without accepting them."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_objective_frame import finite_frame
from .franka_contact_reference import build_from_capture
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) != 0
    config = json.loads((args.run / "results/configuration.json").read_text())
    system, _ = build_from_capture(Path(config["capture"]), config["dhat"])
    system.q_target[:] = 0.018
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.run / "results/force_failure.npz") as failure:
        np.testing.assert_allclose(failure["rest_positions"], system.rest, rtol=0, atol=0)
        failed = failure["displacement"].copy()
        closure = failure["previous_displacement"].copy()
    mapping, _ = system.coordinate_map(False)
    external = system.gravity_force([0, 0, -9.81])
    count, nodes = system.body.basis.N, len(system.body.coordinates)
    records = []
    fields = {}
    for name, u in (("completed_held_closure_at_release", closure), ("failed_free_body_state", failed)):
        _energy, g, _matrix, normal, contact_gradient = system.evaluate_updated_friction(u, closure, external, False)
        f = -contact_gradient
        body_positions = (system.rest + u.reshape(-1, 3))[:nodes]
        body_forces = (f + external).reshape(-1, 3)[:nodes]
        centre = (
            body_positions.T
            @ (system.body.mass[system.body.component_dofs[:, 0]][:, system.body.component_dofs[:, 0]] @ np.ones(nodes))
            / system.body.total_mass
        )
        force = body_forces.sum(axis=0)
        moment = np.cross(body_positions - centre, body_forces).sum(axis=0)
        vector = np.empty(count)
        vector[system.parts[0].permutation] = u[:count]
        fitted_rotation, _translation, local = finite_frame(system.body, vector)
        elastic_max = float(np.linalg.norm(local[system.body.component_dofs], axis=1).max())
        tensor, vm, upper = system.bulk.stress_bounds(vector)
        q = [
            float(0.04 + u[3 * node : 3 * node + 3] @ axis)
            for node, axis in zip(system.finger_com_nodes, system.axes, strict=True)
        ]
        finger_resultants = [f.reshape(-1, 3)[indices].sum(axis=0).tolist() for indices in system.finger_nodes]
        records.append(
            {
                "state": name,
                "physical_acceptance": False,
                "norm_of_unconstrained_body_force_residual_N": float(np.linalg.norm(g[:count])),
                "norm_of_complete_free_map_residual_N": float(np.linalg.norm(mapping.T @ g)),
                "body_net_force_including_gravity_N": force.tolist(),
                "body_net_moment_about_deformed_com_Nm": moment.tolist(),
                "body_complete_rigid_wrench_residual_scaled_N": float(np.linalg.norm(np.r_[force, moment / 0.02])),
                "jaw_position_m": q,
                "jaw_pd_force_N": [float(np.clip(100 * (0.018 - value), -4, 4)) for value in q],
                "finger_contact_force_N": finger_resultants,
                "active_normal_stencils": len(normal),
                "body_peak_lower_Pa": float(vm.max()),
                "body_peak_upper_Pa": float(upper.max()),
                "body_whole_volume_j_lower_bound": system.bulk.determinant_lower_bound(vector),
                "objective_elastic_displacement_max_m": elastic_max,
                "objective_elastic_displacement_over_thickness": elastic_max / 0.0005,
                "objective_elastic_displacement_over_nominal_footprint": elastic_max / 0.006,
                "fitted_rotation_angle_rad": float(np.arccos(np.clip((np.trace(fitted_rotation) - 1) / 2, -1, 1))),
            }
        )
        fields[name + "_tensor_Pa"] = tensor
        fields[name + "_vm_Pa"] = vm
    report = {
        "scope": __doc__,
        "source_run": str(args.run),
        "states": records,
        "goal_completion": False,
        "limitations": "These fields come from incomplete free-body contact states. The held closure still contains support reactions. Neither state establishes complete equilibrium, spatial/regularization/history convergence or accuracy against the actual application. Force/moment residuals retain gravity and all nonlinear-body nodes.",
    }
    np.savez_compressed(args.output / "diagnostic_fields.npz", **fields)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
