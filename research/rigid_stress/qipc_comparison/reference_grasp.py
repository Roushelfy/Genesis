"""Intrusive complete-history QIPC hollow-solid/Franka reference candidate under native capped actuation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import qipc
import qipc._src.native.solver.qipc_solver_ext as extension
from qipc import tetmesh
from qipc.geometry import ground

from .artifacts import file_id, write_json
from .contact_audit import initial_contact_audit
from .reference import ReferenceConfig, SolidObserver, solver_statistics
from .robot_inputs import JOINT_NAMES, prepare_mjcf, robot_target
from .servo import CappedServo, EqualityParameters


def configure_contact(scene, robot, floor_geometry, friction):
    robot_element = scene.contact_tabular.create("robot")
    base_element = scene.contact_tabular.create("fixed-robot-base")
    floor_element = scene.contact_tabular.create("floor")
    floor_element.apply_to(floor_geometry)
    fixed = []
    for slot in robot.links:
        geometry = slot.geometry
        if geometry.instances["is_fixed"].cpu()[0]:
            base_element.apply_to(geometry)
            fixed.append(slot.name)
        else:
            robot_element.apply_to(geometry)
    for left, right in ((robot_element, robot_element), (robot_element, base_element), (robot_element, floor_element)):
        scene.contact_tabular.insert(left, right, friction_rate=0.1, resistance=1e10)
    # The original fixed base and floor touch. They cannot exchange dynamic
    # work; excluding only this fixed/fixed pair preserves their geometry.
    scene.contact_tabular.insert(base_element, floor_element, friction_rate=0.1, resistance=1e10, enable=False)
    return {
        "egg_pair_friction": friction,
        "robot_self_contact": True,
        "adjacent_robot_pairs": "original QIPC minimal-joint ignore_contact default",
        "fixed_base_floor_exclusion": fixed,
        "native_neutral_geom_pair_filter": "not translated to whole-link exclusions; residual collision-filter mismatch must be audited",
    }


def robot_energy(scene, velocity):
    rigid = scene.rigid_body
    count = rigid.n_public_bodies
    quaternion = rigid.quat[:count]
    w, x, y, z = quaternion.unbind(dim=1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    inertia = rotation @ rigid.inertia_tensor[:count] @ rotation.transpose(-1, -2)
    kinetic = (rigid.mass[:count, None] * rigid.v[:count] ** 2).sum() / 2
    kinetic += torch.einsum("bi,bij,bj->", rigid.omega[:count], inertia, rigid.omega[:count]) / 2
    kinetic += 0.1 * (velocity * velocity).sum() / 2
    potential = 9.81 * (rigid.mass[:count] * rigid.t[:count, 2]).sum()
    return kinetic, potential


def snapshot(path, scene, observer, field, contact, servo, step, time):
    global_ids = contact.global_vertex_ids
    np.savez_compressed(
        path,
        step=step,
        time_s=time,
        positions_m=scene.finite_element.x.cpu().numpy(),
        velocities_m_per_s=scene.finite_element.velocities.cpu().numpy(),
        cauchy_pa=field.cauchy_pa.cpu().numpy(),
        det_F=field.determinant.cpu().numpy(),
        contact_forces_n=contact.fem_forces_n.cpu().numpy(),
        active_global_ids=global_ids.cpu().numpy(),
        active_global_forces_n=contact.global_forces_n.cpu().numpy(),
        active_global_positions_m=contact.global_positions_m[global_ids].cpu().numpy(),
        q=servo.q.cpu().numpy(),
        q_velocity=servo.velocity.cpu().numpy(),
        clipped_pd_force=servo.clipped_force.cpu().numpy(),
        effective_actuator_force=servo.effective_actuator_force.cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--equality-constants", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--record-interval", type=float, default=0.01)
    parser.add_argument("--velocity-tolerance", type=float, default=1e-5)
    parser.add_argument("--linear-tolerance", type=float, default=1e-7)
    parser.add_argument("--max-pcg", type=int, default=4096)
    parser.add_argument("--max-newton", type=int, default=256)
    parser.add_argument("--d-hat", type=float, default=2e-5)
    parser.add_argument("--friction-epsilon-velocity", type=float, default=1e-4)
    parser.add_argument("--energy", choices=("snh", "hencky"), default="snh")
    parser.add_argument("--initial-contact-audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    with np.load(args.conditions) as saved:
        bank = {key: saved[key] for key in saved.files}
    with np.load(args.mesh) as mesh:
        authored_vertices, tetrahedra = mesh["vertices"], mesh["tetrahedra"]
    quaternion = bank["quaternion"][args.case]
    initial_rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    vertices = authored_vertices @ initial_rotation.T + bank["position"][args.case]
    initial = bank["initial_joints"][args.case]
    friction = 0.6 * bank["friction"][args.case, 0].item()
    config = ReferenceConfig(
        dt=args.dt,
        mas_precision="double",
        energy=args.energy,
        velocity_tolerance=args.velocity_tolerance,
        linear_tolerance=args.linear_tolerance,
        max_pcg=args.max_pcg,
        max_newton=args.max_newton,
        d_hat=args.d_hat,
        friction=friction,
        friction_epsilon_velocity=args.friction_epsilon_velocity,
    )
    input_receipt = {
        "qipc_import": qipc.__file__,
        "native": file_id(extension.__file__),
        "conditions": file_id(args.conditions),
        "mesh": file_id(args.mesh),
        "equality_constants": file_id(args.equality_constants),
        "config": asdict(config),
        "case": args.case,
        "minimum_initial_egg_height_m": vertices[:, 2].min().item(),
        "protocol": "matched capped-controller candidate; contact and reference qualification pending",
        "output_scope": "intrusive full-field/force/energy diagnostics, excluded from primary throughput",
        "reconstruction_radius_controls_reference": False,
    }
    write_json(args.output / "inputs.json", input_receipt)
    assert vertices[:, 2].min() > 0, "Authored initial egg intersects the unchanged plane"
    ticks_per_control = round(0.01 / config.dt)
    recording_stride = round(args.record_interval / config.dt)
    steps = round(args.duration / config.dt)
    assert ticks_per_control > 0 and np.isclose(ticks_per_control * config.dt, 0.01)
    assert recording_stride > 0 and np.isclose(recording_stride * config.dt, args.record_interval)
    scene = config.scene()
    robot = scene.add_mjcf(
        prepare_mjcf(args.mjcf, initial, args.output),
        link_type="rigid",
        minimal_coordinates=True,
        authored_passive_terms=True,
        enable_controller=True,
    )
    assert robot.joint_names == JOINT_NAMES
    equality = EqualityParameters.from_native(args.equality_constants)
    equality.attach(robot, config.dt)
    geometry = tetmesh(vertices, tetrahedra)
    config.material().apply_to(
        geometry, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
    )
    scene.geometries.create("egg", geometry)
    floor = ground(height=0.0, N=(0, 0, 1))
    scene.geometries.create("ground", floor)
    contact_contract = configure_contact(scene, robot, floor, friction)
    started = perf_counter()
    scene.init()
    with torch.cuda.stream(scene.solver.cuda_stream):
        observer = SolidObserver.allocate(scene, config)
        servo = CappedServo.allocate(robot, config.dt, equality)
    torch.cuda.synchronize()
    scene.solver.accd_iter_stats = True
    scene.solver.min_gap_stats = True
    write_json(
        args.output / "start.json",
        {
            **input_receipt,
            "setup_seconds": perf_counter() - started,
            "contact_contract": contact_contract,
            "equality_mapping": asdict(equality),
            "equality_mapping_limit": "Incremental equality inertia uses authored dmin. Record the actual state-dependent coefficient mismatch; it must fit the reference uncertainty budget.",
            "egg_mass_kg": observer.masses.sum().item(),
            "robot_public_links": scene.rigid_body.n_public_bodies,
        },
    )
    np.savez_compressed(
        args.output / "rest.npz",
        vertices_world_m=observer.rest.cpu().numpy(),
        authored_vertices_m=authored_vertices,
        initial_rotation=initial_rotation,
        tetrahedra=tetrahedra,
        masses_kg=observer.masses.cpu().numpy(),
        volumes_m3=observer.volumes.cpu().numpy(),
    )
    if args.initial_contact_audit:
        initial_contact_audit(scene, robot, args.output)
        return
    completed = False
    count = 0
    invalid = 0
    pcg_cap_steps = 0
    resets = 0
    actuator_work = 0.0
    initial_total_energy = None
    motion_time, motion_q = [0.0], [initial.copy()]
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(steps):
                control_tick = step // ticks_per_control
                reset = (
                    step % ticks_per_control == 0
                    and control_tick > bank["delays"][args.case]
                    and (control_tick - bank["delays"][args.case]) % 600 == 0
                )
                if reset:
                    scene.reset()
                    with torch.cuda.stream(scene.solver.cuda_stream):
                        servo = CappedServo.allocate(robot, config.dt, equality)
                    torch.cuda.synchronize()
                    resets += 1
                    actuator_work = 0.0
                    initial_total_energy = None
                target, cap, phase = robot_target(bank, args.case, control_tick)
                with torch.cuda.stream(scene.solver.cuda_stream):
                    previous_velocity = scene.finite_element.velocities.clone()
                    servo.push(servo.q.new_tensor(target), servo.q.new_tensor(cap))
                scene.step()
                with torch.cuda.stream(scene.solver.cuda_stream):
                    servo.update()
                    field = observer.update()
                torch.cuda.synchronize()
                contact = observer.contact()
                fe = scene.finite_element
                residual = observer.masses[:, None] * ((fe.velocities - previous_velocity) / config.dt - fe.gravity)
                residual += observer.elastic_gradient(field) - contact.fem_forces_n
                contact_norm = torch.linalg.norm(contact.fem_forces_n).item()
                force_norm = torch.linalg.norm(residual).item()
                kinetic = (observer.masses[:, None] * fe.velocities**2).sum() / 2
                potential = -(observer.masses[:, None] * fe.gravity * fe.x).sum()
                elastic = (observer.volumes * field.energy_density_relative_rest_j_per_m3).sum()
                robot_kinetic, robot_potential = robot_energy(scene, servo.velocity)
                total_energy = (kinetic + potential + elastic + robot_kinetic + robot_potential).item()
                if initial_total_energy is None:
                    initial_total_energy = total_energy
                actuator_work += (servo.effective_actuator_force * (servo.q - servo.frame_q)).sum().item()
                status = scene.solver.last_step_status
                finite = bool(torch.isfinite(field.cauchy_pa).all() and torch.isfinite(servo.q).all())
                physical = finite and field.determinant.min().item() > 0
                failed = not physical or status["newton_budget_exhausted"] or status["line_search_exhausted"]
                gap = scene.solver.min_gap_ratio
                row = {
                    "step": step,
                    "time_s": (step + 1) * config.dt,
                    "control_tick": control_tick,
                    "phase": float(phase),
                    "reset_epoch": resets,
                    **solver_statistics(scene),
                    "status": status,
                    "minimum_det_F": field.determinant.min().item(),
                    "all_fields_finite": finite,
                    "raw_vm_peak_pa": field.von_mises_pa.max().item(),
                    "raw_maximum_principal_pa": field.maximum_principal_pa.max().item(),
                    "minimum_surface_height_m": fe.x[:, 2].min().item(),
                    "contact_resultant_n": contact.fem_forces_n.sum(dim=0).tolist(),
                    "contact_moment_world_origin_nm": torch.cross(fe.x, contact.fem_forces_n, dim=1)
                    .sum(dim=0)
                    .tolist(),
                    "contact_nodal_l2_n": contact_norm,
                    "force_residual_l2_n": force_norm,
                    "force_residual_relative_contact": force_norm / contact_norm if contact_norm > 1e-12 else None,
                    "force_residual_resultant_n": residual.sum(dim=0).tolist(),
                    "contact_vertices": int((torch.linalg.norm(contact.fem_forces_n, dim=1) > 1e-12).sum()),
                    "ccd_maximum_iterations": scene.solver.max_accd_iters,
                    "minimum_reported_gap_ratio": gap if np.isfinite(gap) else None,
                    "egg_kinetic_j": kinetic.item(),
                    "egg_potential_j": potential.item(),
                    "egg_elastic_j": elastic.item(),
                    "robot_kinetic_including_rotors_j": robot_kinetic.item(),
                    "robot_potential_j": robot_potential.item(),
                    "actuator_work_since_reset_j": actuator_work,
                    "total_energy_change_minus_actuator_work_j": total_energy - initial_total_energy - actuator_work,
                    "equality_inertia_coefficient_relative_difference": servo.equality_coefficient_difference.item(),
                    "finger_q": servo.q[7:].tolist(),
                    "invalid_step": failed,
                }
                count += 1
                invalid += int(failed)
                pcg_cap_steps += int(scene.solver.max_pcg_iters >= config.max_pcg)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                motion_time.append((step + 1) * config.dt)
                motion_q.append(servo.q.cpu().numpy())
                if (step + 1) % recording_stride == 0 or failed or step == steps - 1:
                    snapshot(
                        frames / f"step{step:06d}.npz",
                        scene,
                        observer,
                        field,
                        contact,
                        servo,
                        step,
                        (step + 1) * config.dt,
                    )
                if failed:
                    break
            completed = count == steps and invalid == 0
        finally:
            if not completed:
                np.savez_compressed(
                    args.output / "failure-state.npz",
                    positions_m=scene.finite_element.x.cpu().numpy(),
                    velocities_m_per_s=scene.finite_element.velocities.cpu().numpy(),
                    q=servo.q.cpu().numpy(),
                    velocity=servo.velocity.cpu().numpy(),
                )
            np.savez_compressed(args.output / "actual-robot-motion.npz", time_s=motion_time, q=motion_q)
            write_json(
                args.output / "result.json",
                {
                    "kind": "QIPC-Franka-capped-candidate",
                    "config": asdict(config),
                    "requested_steps": steps,
                    "executed_steps": count,
                    "completed_requested_history": completed,
                    "invalid_steps": invalid,
                    "pcg_cap_steps": pcg_cap_steps,
                    "resets": resets,
                    "intrusive_elapsed_seconds": perf_counter() - started,
                    "reference_qualified": False,
                },
            )
    if not completed:
        raise RuntimeError("Unqualified reference stopped on numerical/physical failure; raw state retained")


if __name__ == "__main__":
    main()
