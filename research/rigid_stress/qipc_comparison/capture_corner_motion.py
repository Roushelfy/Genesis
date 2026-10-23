"""Capture contact-time rigid velocities around the original sharp-corner refusal."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, nnls
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .check_live_curved_hertz_franka import diagnose, hard_snapshot
from .check_live_hertz_franka import capture
from .check_live_hertz_normal import snapshot
from .check_native_hertz_recovery import array
from .time_hard_franka import build_workload, observe


def kinematics(solver, ancestors):
    """Read rigid motion and check its degree-of-freedom reconstruction at link origins."""
    positions = tensor_to_array(solver.get_links_pos()).copy()
    velocities = tensor_to_array(solver.get_links_vel()).copy()
    angular, linear = array(solver.dyn_state.dofs.cdof_ang), array(solver.dyn_state.dofs.cdof_vel)
    rates, roots = array(solver.dyn_state.dofs.vel), array(solver.dyn_state.links.root_COM)
    jacobian = linear[:, None] + np.cross(angular[:, None], (positions - roots)[:, :, None])
    reconstructed = np.einsum("bldk,bd,ld->blk", jacobian, rates, ancestors)
    np.testing.assert_allclose(reconstructed, velocities, rtol=1e-11, atol=1e-13)
    return {
        "link_position_m": positions,
        "link_velocity_m_per_s": velocities,
        "root_com_m": roots,
        "dof_angular": angular,
        "dof_linear": linear,
        "dof_velocity": rates,
        "reconstruction_maximum_error_m_per_s": np.max(abs(reconstructed - velocities)),
    }


def point_velocity(state, ancestors, links, points, rates, env):
    delta = points - state["root_com_m"][env, links]
    jacobian = state["dof_linear"][env, None] + np.cross(state["dof_angular"][env, None], delta[:, None])
    return np.einsum("idk,d,id->ik", jacobian, rates[env], ancestors[links])


def minimum_side_force(force, friction):
    """Certify the minimum side normal load in two orthogonal circular Coulomb cones.

    With force=(-X,Y,Z), a primary-cone dual vector is (cos(phi),-sin(phi),mu).
    It gives S >= (X*cos(phi)+Y*sin(phi)-mu*Z)/(cos(phi)+mu*sqrt(sin(phi)^2+mu^2)).
    An attaining primal decomposition certifies the scalar bound globally. Its tractions carry no slip history.
    """
    x, y, z = -force[0], force[1], force[2]
    if np.linalg.norm(force[:2]) <= friction * z:
        return {
            "side_normal_lower_bound_n": 0.0,
            "primary_force_n": force.tolist(),
            "side_force_n": [0.0, 0.0, 0.0],
            "dual_vector": [0.0, 0.0, 0.0],
            "primal_dual_gap_n": 0.0,
            "scope": "The primary circular cone alone contains this force. Moments, geometry and slip history remain separate.",
        }
    assert x > friction * z > 0

    def derivative(angle):
        c, s = np.cos(angle), np.sin(angle)
        radius = np.sqrt(s * s + friction * friction)
        numerator, denominator = x * c + y * s - friction * z, c + friction * radius
        return (-x * s + y * c) * denominator - numerator * (-s + friction * s * c / radius)

    angle = brentq(derivative, -np.pi / 2, np.pi / 2, xtol=1e-15, rtol=1e-14)
    c, s = np.cos(angle), np.sin(angle)
    radius = np.sqrt(s * s + friction * friction)
    side = (x * c + y * s - friction * z) / (c + friction * radius)
    side_force = side * np.array([-1.0, friction * s / radius, -(friction**2) / radius])
    primary_force = force - side_force
    np.testing.assert_allclose(primary_force[:2], friction * primary_force[2] * np.array([-c, s]), atol=1e-14)
    assert primary_force[2] > 0 and side > 0
    dual = np.array([c, -s, friction]) / (c + friction * radius)
    np.testing.assert_allclose(-dual @ force, side, rtol=1e-13, atol=1e-16)
    return {
        "side_normal_lower_bound_n": float(side),
        "primary_force_n": primary_force.tolist(),
        "side_force_n": side_force.tolist(),
        "dual_vector": dual.tolist(),
        "primal_dual_gap_n": float(abs(side + dual @ force)),
        "scope": "Force-only exact circular-cone bound with an attaining primal. Moments, contact geometry, slip dissipation and elastic history are additional constraints.",
    }


def contact_motion(before, after, ancestors, frame, hard, raw, env):
    rotation = Rotation.from_quat(frame["frame_quaternion"][env][[1, 2, 3, 0]])
    rows = []
    for region in np.flatnonzero((hard["source_count"][env] > 0) & (hard["profile"][env] == 2)):
        contacts = np.flatnonzero(
            (hard["leader"][env] == region)
            & frame["contact_valid"][env]
            & (np.linalg.norm(frame["contact_force_n"][env], axis=1) > 0)
        )
        raw_ids = raw["sort_idx"][env, contacts]
        is_first = frame["source_geom"][env, contacts] == raw["geom_a"][env, raw_ids]
        own = np.where(is_first, raw["link_a"][env, raw_ids], raw["link_b"][env, raw_ids])
        other = np.where(is_first, raw["link_b"][env, raw_ids], raw["link_a"][env, raw_ids])
        assert own.min() >= 0 and other.min() >= 0
        force = frame["contact_force_n"][env, contacts] @ rotation.as_matrix().T
        np.testing.assert_allclose(force, raw["force"][env, raw_ids] * np.where(is_first, -1, 1)[:, None], atol=1e-14)
        own_witness = np.where(is_first[:, None], raw["witness_a"][env, raw_ids], raw["witness_b"][env, raw_ids])
        other_witness = np.where(is_first[:, None], raw["witness_b"][env, raw_ids], raw["witness_a"][env, raw_ids])
        basis = rotation.as_matrix() @ hard["geometry_basis"][env, region] @ np.diag([1, -1, -1])
        weights = np.linalg.norm(force, axis=1)
        normals = np.array([[0.0, 0, 1], [-1.0, 0, 0]])
        local_force = force.sum(axis=0) @ basis
        friction = hard["allocation_friction"][env, region]
        bound = minimum_side_force(local_force, friction)
        motions = []
        for name, points, opponent_points in [
            ("midpoint", raw["position"][env, raw_ids], raw["position"][env, raw_ids]),
            ("opponent_point", other_witness, other_witness),
            ("witness_pair", own_witness, other_witness),
        ]:
            for phase, rates in [
                ("pre_solve", before["dof_velocity"]),
                ("post_solve_frozen_geometry", after["dof_velocity"]),
            ]:
                relative = point_velocity(before, ancestors, own, points, rates, env) - point_velocity(
                    before, ancestors, other, opponent_points, rates, env
                )
                relative = relative @ basis
                mean = weights @ relative / weights.sum()
                tangent = mean - (normals @ mean)[:, None] * normals
                speed = np.linalg.norm(tangent, axis=1)
                assert speed.min() > 0
                directions = normals - friction * tangent / speed[:, None]
                loads, residual = nnls(directions.T, local_force)
                limiting_forces = np.array([bound["primary_force_n"], bound["side_force_n"]])
                shear = limiting_forces - np.einsum("ij,ij->i", limiting_forces, normals)[:, None] * normals
                motions.append(
                    {
                        "points": name,
                        "velocity_phase": phase,
                        "relative_velocities_in_face_frame_m_per_s": relative.tolist(),
                        "force_weighted_mean_m_per_s": mean.tolist(),
                        "face_tangential_speeds_m_per_s": speed.tolist(),
                        "minimum_side_bound_friction_powers_w": np.einsum("ij,ij->i", shear, tangent).tolist(),
                        "saturated_sliding_force_directions": directions.tolist(),
                        "best_nonnegative_sliding_normal_loads_n": loads.tolist(),
                        "saturated_sliding_force_residual_relative": float(residual / np.linalg.norm(local_force)),
                        "saturated_sliding_force_residual_n": (directions.T @ loads - local_force).tolist(),
                    }
                )
        rows.append(
            {
                "region": int(region),
                "contacts": contacts.tolist(),
                "force_in_face_frame_n": local_force.tolist(),
                "friction": float(friction),
                "minimum_side_force": bound,
                "motions": motions,
            }
        )
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["stress-mesh", "collision-urdf", "conditions", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "inputs": [file_id(p) for p in [args.stress_mesh, args.collision_urdf, args.conditions]],
            "scope": "Read-only rigid kinematic and two-face friction diagnostics through the original frame428 refusal. Post-solve velocities use final generalized rates at the captured pre-solve geometry. No elastic stick/slip law is inferred solely from rigid velocities.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = build_workload(args, 4, "max")
    solver = workload.scene.rigid_solver
    assert not solver.rigid_config.batch_links_info
    parents = array(solver.dyn_info.links.parent_idx)
    starts, ends = array(solver.dyn_info.links.dof_start), array(solver.dyn_info.links.dof_end)
    ancestors = np.zeros((solver.n_links, solver.n_dofs))
    for link in range(solver.n_links):
        current = link
        while current >= 0:
            ancestors[link, starts[current] : ends[current]] = 1
            current = parents[current]
    rows = []
    for step in range(429):
        before = kinematics(solver, ancestors) if step in [419, 420, 427, 428] else None
        workload.step()
        if step < 428:
            observe(workload, "max")
        if before is not None:
            after = kinematics(solver, ancestors)
            output = args.output / f"step{step}"
            output.mkdir()
            record = capture(workload, output, step)
            write_json(output / "frame.json", record)
            diagnose(workload, output, step, True, True)
            np.savez_compressed(
                output / "motion.npz",
                ancestors=ancestors,
                **{"before_" + k: v for k, v in before.items()},
                **{"after_" + k: v for k, v in after.items()},
            )
            with np.load(output / f"contacts{step:04d}.npz") as saved:
                raw = {k: saved[k] for k in saved.files}
            frame = snapshot(solver.stress_recovery.links[0])
            hard = hard_snapshot(solver.stress_recovery.links[0].hertz.hard)
            np.testing.assert_allclose(
                before["link_position_m"][:, workload.link.idx], frame["frame_position_m"], atol=1e-13
            )
            rows.append(
                {
                    "step": step,
                    "accepted": record["accepted"],
                    "input_status": record["input_status"],
                    "motion": contact_motion(before, after, ancestors, frame, hard, raw, env=0),
                    "capture": file_id(output / "motion.npz"),
                }
            )
            write_json(args.output / "progress.json", {"rows": rows})
    assert rows[-1]["accepted"] == [False, True, True, True] and rows[-1]["input_status"] == [32, 0, 0, 0]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "pre_failure_frames_accepted": 428,
            "original_frame428_refusal_reproduced": True,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
