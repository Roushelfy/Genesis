"""Reconstruct full body and joint balances independently of the coupled Newton solver."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np

from .audit_objective_frame import finite_frame
from .franka_contact_reference import build_from_capture
from .franka_mujoco_reference import PandaSkeleton
from .summarize_fine_references import closed_source


def quadrature_fields(bulk, vector):
    h = np.einsum("aie,ajeq->ijeq", vector[bulk.dofs], bulk.gradients, optimize=True).transpose(2, 3, 0, 1)
    trace = np.trace(h, axis1=-2, axis2=-1)
    delta = trace + (trace**2 - np.einsum("...ij,...ji->...", h, h)) / 2 + np.linalg.det(h)
    kirchhoff = bulk.mu * (h + h.swapaxes(-1, -2) + h @ h.swapaxes(-1, -2))
    log_j = np.log1p(delta)
    kirchhoff += bulk.lam * log_j[..., None, None] * np.eye(3)
    density = bulk.mu * (trace - log_j) + 0.5 * bulk.mu * np.sum(h * h, axis=(-2, -1)) + 0.5 * bulk.lam * log_j**2
    return kirchhoff / (1 + delta)[..., None, None], float(np.sum(bulk.basis.dx * density))


def audit_run(run, output):
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 0
    config = json.loads((run / "results/configuration.json").read_text())
    report = json.loads((run / "results/summary.json").read_text())
    assert len(report["steps"]) == config["steps"]
    capture = Path(config["capture"])
    captured = json.loads((capture / "results/summary.json").read_text())
    static, _ = build_from_capture(capture, config["dhat"])
    body, permutation = static.body, static.parts[0].permutation
    n = body.basis.N
    robot = PandaSkeleton(Path(config["mjcf"]), config["dt"])
    robot.model.opt.tolerance = report.get("robot_solver_tolerance", robot.model.opt.tolerance)
    kp, kv = np.array(captured["robot_joint_kp"]), np.array(captured["robot_joint_kv"])
    target = np.r_[np.array(captured["stationary_actuation"]["arm_target"])[0], 0.018, 0.018]
    caps = np.r_[robot.model.actuator_forcerange[:7, 1], 4.0, 4.0]
    for key, value in [
        ("controller_target", target),
        ("controller_kp", kp),
        ("controller_kv", kv),
        ("controller_caps", caps),
    ]:
        np.testing.assert_array_equal(report[key], value)
    with np.load(run / "results/reference_geometry.npz") as geometry:
        np.testing.assert_array_equal(geometry["vertices"], body.mesh.p.T)
        np.testing.assert_array_equal(geometry["tetrahedra"], body.mesh.t.T)
        np.testing.assert_array_equal(geometry["rest_positions"], static.rest)
    with np.load(run / "results/closure_state.npz") as saved:
        previous = saved["full_displacement"].copy()
        previous_q = saved["joint_position"].copy()
        closure_q = previous_q.copy()
    previous_velocity = np.zeros(9)
    dt = config["dt"]
    friction = ipctk.FrictionPotential(dt * 4e-6)
    mass = body.mass[permutation][:, permutation]
    gravity = static.gravity_force([0, 0, -9.81])
    rows, observations = [], []
    fixed_arm = bool(config.get("fixed_arm", False))
    for step in range(1, config["steps"] + 1):
        with np.load(run / f"results/step{step:04d}.npz") as saved:
            u, q, velocity, acceleration, force = [
                saved[k].copy()
                for k in ["displacement", "joint_position", "joint_velocity", "body_acceleration", "contact_force_N"]
            ]
            support = (
                saved["arm_support_generalized_effort"].copy()
                if "arm_support_generalized_effort" in saved
                else np.zeros(9)
            )
            np.testing.assert_array_equal(saved["body_component_permutation"], permutation)
        if fixed_arm:
            np.testing.assert_array_equal(q[:7], closure_q[:7])
            np.testing.assert_array_equal(velocity[:7], np.zeros(7))
            np.testing.assert_array_equal(support[7:], np.zeros(2))
        else:
            np.testing.assert_array_equal(support, np.zeros(9))
        normal, current = static.normal_collisions(u)
        tangential = ipctk.TangentialCollisions()
        tangential.build(static.mesh, current, normal, static.barrier, static.friction)
        delta = static.mesh.vertices((u - previous).reshape(-1, 3))
        reconstructed = -static.mesh.to_full_dof(
            static.barrier.gradient(normal, static.mesh, current) + friction.gradient(tangential, static.mesh, delta)
        )
        np.testing.assert_allclose(reconstructed, force, rtol=1e-10, atol=1e-12)
        vector = np.empty(n)
        vector[permutation] = u[:n]
        _energy, gradient, _ = static.bulk.evaluate(vector, False)
        body_residual = gradient[permutation] + mass @ acceleration - gravity[:n] - force[:n]
        assert np.linalg.norm(body_residual) <= 1e-9
        baseline, response = robot.velocity_response(previous_q, previous_velocity)
        inverse = np.linalg.inv(response)
        robot.set_state(q, velocity)
        joint_contact = np.zeros(9)
        positions = static.rest + u.reshape(-1, 3) + static.world_coordinate_origin_m
        finger_wrenches = []
        for name, nodes in zip(["left_finger", "right_finger"], static.finger_nodes):
            origin = robot.data.xpos[robot.body_ids[name]]
            point_force = force.reshape(-1, 3)[nodes]
            resultant = point_force.sum(axis=0)
            moment = np.cross(positions[nodes] - origin, point_force).sum(axis=0)
            wrench = np.r_[resultant, moment]
            joint_contact += robot.jacobian(name).T @ wrench
            finger_wrenches.append(wrench)
        pd = np.clip(kp * (target - q) - kv * velocity, -caps, caps)
        joint_residual = inverse @ (velocity - baseline) - pd - joint_contact - support
        assert np.linalg.norm(joint_residual) <= 1e-9
        actual = robot.step_velocity(previous_q, previous_velocity, pd + joint_contact + support)
        direct = inverse @ (velocity - actual)
        assert np.linalg.norm(direct) <= 1e-9
        frame, _, objective_u = finite_frame(body, vector)
        tensor, energy = quadrature_fields(static.bulk, vector)
        objective_tensor = frame.T @ tensor @ frame
        contact_body_force = force[:n].reshape(-1, 3)
        body_wrench = np.r_[
            contact_body_force.sum(axis=0), np.cross(positions[: n // 3], contact_body_force).sum(axis=0)
        ]
        observations.append(
            {
                "tensor": objective_tensor,
                "energy": energy,
                "body_wrench": body_wrench,
                "objective_displacement": objective_u,
                "joint_position": q.copy(),
            }
        )
        rows.append(
            {
                "step": step,
                "body_force_residual_N": float(np.linalg.norm(body_residual)),
                "joint_force_residual_N_or_Nm": float(np.linalg.norm(joint_residual)),
                "fresh_step_effort_residual": float(np.linalg.norm(direct)),
                "contact_pullback_max_error_N": float(abs(reconstructed - force).max()),
                "arm_support_effort_norm": float(np.linalg.norm(support)),
                "normal_stencils": len(normal),
                "objective_elastic_displacement_max_m": float(
                    np.linalg.norm(objective_u[body.component_dofs], axis=1).max()
                ),
            }
        )
        np.savez_compressed(
            output / f"step{step:04d}.npz",
            body_residual_N=body_residual,
            joint_residual=joint_residual,
            fresh_step_residual=direct,
            finger_contact_wrenches_world=np.array(finger_wrenches),
            objective_tensor_Pa=objective_tensor,
            body_wrench_world=body_wrench,
            objective_displacement_m=objective_u,
        )
        previous, previous_q, previous_velocity = u, q, velocity
    return (
        {"run": str(run), "fixed_arm": fixed_arm, "steps_checked": len(rows), "rows": rows},
        observations,
        static.bulk.basis.dx,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases, histories, weights = [], [], []
    for index, run in enumerate(args.run):
        directory = args.output / f"case{index:02d}"
        directory.mkdir()
        case, history, dx = audit_run(run, directory)
        cases.append(case)
        histories.append(history)
        weights.append(dx)
    comparisons = []
    if len(cases) == 2:
        configs = [json.loads((run / "results/configuration.json").read_text()) for run in args.run]
        for key in configs[0]:
            if key not in ["output", "mjcf", "fixed_arm"]:
                assert configs[0][key] == configs[1][key]
        assert len(histories[0]) == len(histories[1])
        assert (
            hashlib.sha256(Path(configs[0]["mjcf"]).read_bytes()).hexdigest()
            == hashlib.sha256(Path(configs[1]["mjcf"]).read_bytes()).hexdigest()
        )
        np.testing.assert_array_equal(weights[0], weights[1])
        for index, (old, new) in enumerate(zip(histories[0], histories[1])):
            difference = old["tensor"] - new["tensor"]
            norm = lambda x: np.sqrt(np.sum(weights[0] * np.sum(x * x, axis=(-2, -1))))
            comparisons.append(
                {
                    "step": index + 1,
                    "objective_volume_tensor_relative_change": float(norm(difference) / norm(new["tensor"])),
                    "body_elastic_energy_relative_change": float(abs(old["energy"] / new["energy"] - 1)),
                    "body_contact_force_relative_change": float(
                        np.linalg.norm(old["body_wrench"][:3] - new["body_wrench"][:3])
                        / np.linalg.norm(new["body_wrench"][:3])
                    ),
                    "body_contact_moment_relative_change": float(
                        np.linalg.norm(old["body_wrench"][3:] - new["body_wrench"][3:])
                        / np.linalg.norm(new["body_wrench"][3:])
                    ),
                }
            )
    result = {
        "scope": __doc__,
        "cases": cases,
        "support_comparisons": comparisons,
        "goal_completion": False,
        "limitations": "All retained steps reconstruct complete body and joint balances, actual contact pullback and a fresh independent robot step with original1e-9 budgets. Volume quadrature tensor comparison is objective. Prescribed-arm reactions are explicit and all other matched protocol inputs remain fixed. This support-axis comparison does not establish rigid/deformable feedback, matched original approach/angular history or physical mesh/contact/time convergence.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"cases": len(cases), "states_checked": sum(c["steps_checked"] for c in cases), "goal_completion": False}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
