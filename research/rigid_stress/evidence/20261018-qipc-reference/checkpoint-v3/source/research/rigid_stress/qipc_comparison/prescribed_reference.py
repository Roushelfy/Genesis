"""Original-main QIPC 3D hollow FEM under shared robot-link pose histories; actuator caps are overridden."""

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

from .artifacts import file_id, runtime_identity, write_json
from .pose_robot import PoseRobot, pose_contact_table
from .reference import ReferenceConfig, SolidObserver, preserve_contact_failure, solver_statistics
from .robot_inputs import prepare_mjcf


def save_frame(path, scene, state, target_q, sample, step, time):
    np.savez_compressed(
        path,
        time_s=time,
        step=step,
        sample=sample,
        positions_m=scene.finite_element.x.cpu().numpy(),
        velocities_m_per_s=scene.finite_element.velocities.cpu().numpy(),
        cauchy_pa=state.field.cauchy_pa.cpu().numpy(),
        det_F=state.field.determinant.cpu().numpy(),
        contact_forces_n=state.contact.fem_forces_n.cpu().numpy(),
        inertial_reaction_n=state.inertial_reaction_n.cpu().numpy(),
        target_q=target_q,
        actual_robot_com_world_m=scene.rigid_body.t.cpu().numpy(),
        actual_robot_material_quaternion_wxyz=scene.rigid_body.quat.cpu().numpy(),
        active_global_ids=state.contact.global_vertex_ids.cpu().numpy(),
        active_global_forces_n=state.contact.global_forces_n.cpu().numpy(),
        active_global_positions_m=state.contact.global_positions_m[state.contact.global_vertex_ids].cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--dbc-stiffness", type=float, default=1e8)
    parser.add_argument("--velocity-tolerance", type=float, default=1e-7)
    parser.add_argument("--linear-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-pcg", type=int, default=4096)
    parser.add_argument("--max-newton", type=int, default=256)
    parser.add_argument("--energy", choices=("snh", "hencky"), default="snh")
    parser.add_argument("--d-hat", type=float, default=2e-5)
    parser.add_argument("--friction-epsilon-velocity", type=float, default=1e-4)
    parser.add_argument("--resistance", type=float, default=1e10)
    parser.add_argument("--record-interval", type=float, default=0.01)
    parser.add_argument("--record-times", type=float, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.max_pcg > 0 and args.max_newton > 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    with np.load(args.motion) as saved:
        motion = {key: saved[key] for key in saved.files}
    sample_dt = motion["time_s"][1] - motion["time_s"][0]
    stride = round(args.dt / sample_dt)
    assert stride > 0 and np.isclose(stride * sample_dt, args.dt)
    assert args.duration <= motion["time_s"][-1]
    record_stride = round(args.record_interval / args.dt)
    assert record_stride > 0 and abs(record_stride * args.dt - args.record_interval) < 1e-12
    extra_steps = {round(time / args.dt) for time in args.record_times}
    assert all(
        0 < time <= args.duration and abs(round(time / args.dt) * args.dt - time) < 1e-12 for time in args.record_times
    )
    with np.load(args.conditions) as bank:
        position, quaternion = bank["position"][0], bank["quaternion"][0]
        coefficient = 0.6 * bank["friction"][0, 0].item()
    with np.load(args.mesh) as mesh:
        authored, tetrahedra = mesh["vertices"], mesh["tetrahedra"]
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    world = authored @ rotation.T + position
    config = ReferenceConfig(
        dt=args.dt,
        friction=coefficient,
        mas_precision="double",
        linear_solver="linear_pcg",
        velocity_tolerance=args.velocity_tolerance,
        linear_tolerance=args.linear_tolerance,
        max_pcg=args.max_pcg,
        max_newton=args.max_newton,
        energy=args.energy,
        d_hat=args.d_hat,
        friction_epsilon_velocity=args.friction_epsilon_velocity,
        resistance=args.resistance,
    )
    scene = config.scene()
    descriptor_scene = config.scene(contact=False)
    descriptor = descriptor_scene.add_mjcf(prepare_mjcf(args.mjcf, motion["q"][0], args.output), link_type="rigid")
    robot = PoseRobot.create(scene, descriptor, args.dbc_stiffness)
    geometry = tetmesh(world, tetrahedra)
    config.material().apply_to(
        geometry, youngs_modulus=config.young, poissons_ratio=config.poisson, mass_density=config.density
    )
    scene.geometries.create("egg", geometry)
    floor = ground(height=0, N=(0, 0, 1))
    scene.geometries.create("floor", floor)
    pose_contact_table(scene, robot, floor, config.resistance)
    write_json(
        args.output / "inputs.json",
        {
            "identity": runtime_identity(),
            "qipc_import": qipc.__file__,
            "native": file_id(extension.__file__),
            "motion": file_id(args.motion),
            "mesh": file_id(args.mesh),
            "conditions": file_id(args.conditions),
            "config": asdict(config),
            "dbc_stiffness_ratio": args.dbc_stiffness,
            "requested_duration_s": args.duration,
            "record_interval_s": args.record_interval,
            "extra_observation_times_s": args.record_times,
            "observer_scope": "Stress/residual/raw peak observations at every physical step. Periodic/extra full-state frames follow the explicit recording grid; sparse pilot frames do not establish complete6s common observation coverage.",
            "protocol": "shared prescribed motion; original rigid pose-DBC potentials, no articulated KKT, no capped controller",
            "tracking_contract": "Finite-strength DBC is admitted only after measured pose and stiffness-sensitivity checks; the egg is free and every contact/reaction is solved.",
        },
    )
    started = perf_counter()
    scene.init()
    with torch.cuda.stream(scene.solver.cuda_stream):
        observer = SolidObserver.allocate(scene, config)
        robot.allocate_targets(scene, motion)
    torch.cuda.synchronize()
    scene.solver.accd_iter_stats = True
    scene.solver.min_gap_stats = True
    np.savez_compressed(
        args.output / "rest.npz",
        authored_vertices_m=authored,
        vertices_world_m=world,
        tetrahedra=tetrahedra,
        masses_kg=observer.masses.cpu().numpy(),
        volumes_m3=observer.volumes.cpu().numpy(),
    )
    steps = round(args.duration / config.dt)
    completed = False
    executed = 0
    failed = False
    peak = -np.inf
    principal_peak = -np.inf
    contact_work = 0.0
    boundary_work = 0.0
    initial_potential = -(observer.masses[:, None] * scene.finite_element.gravity * scene.finite_element.x).sum().item()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step in range(steps):
                sample = (step + 1) * stride
                with torch.cuda.stream(scene.solver.cuda_stream):
                    previous = scene.finite_element.velocities.clone()
                    robot.push(sample)
                scene.step()
                state = observer.read_step(previous)
                position_error, angle_error = robot.tracking(scene, sample)
                contact_work += (state.contact.fem_forces_n * scene.finite_element.velocities).sum().item() * config.dt
                boundary_work += robot.boundary_power(scene, state.contact) * config.dt
                gap = scene.solver.min_gap_ratio
                row = {
                    "step": step,
                    "time_s": (step + 1) * config.dt,
                    **solver_statistics(scene),
                    **state.metrics(),
                    "maximum_robot_com_tracking_error_m": position_error,
                    "maximum_robot_rotation_tracking_error_rad": angle_error,
                    "newton_cap_reached": scene.solver.newton_iters >= config.max_newton,
                    "pcg_cap_reached": scene.solver.max_pcg_iters >= config.max_pcg,
                    "line_search_cap_reached": scene.solver.max_ls_iters >= 64,
                    "ccd_maximum_iterations": scene.solver.max_accd_iters,
                    "minimum_reported_gap_ratio": gap if np.isfinite(gap) else None,
                    "contact_work_on_egg_j": contact_work,
                    "moving_rigid_boundary_contact_work_j": boundary_work,
                    "egg_energy_change_minus_contact_work_j": (
                        state.kinetic_j + state.potential_j + state.elastic_j
                    ).item()
                    - initial_potential
                    - contact_work,
                }
                intersection_failed = row["edge_triangle_intersection_count"] not in (None, 0)
                failed = (
                    not row["all_fields_finite"]
                    or row["minimum_det_F"] <= 0
                    or row["newton_cap_reached"]
                    or intersection_failed
                )
                row["invalid_step"] = failed
                row["edge_triangle_forced_exit"] = intersection_failed
                executed += 1
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                if (step + 1) % record_stride == 0 or (step + 1) in extra_steps or failed or step == steps - 1:
                    save_frame(
                        args.output / "frames" / f"step{step:06d}.npz",
                        scene,
                        state,
                        motion["q"][sample],
                        sample,
                        step,
                        row["time_s"],
                    )
                if row["raw_vm_peak_pa"] > peak:
                    peak = row["raw_vm_peak_pa"]
                    save_frame(
                        args.output / "raw-vm-peak-state.npz",
                        scene,
                        state,
                        motion["q"][sample],
                        sample,
                        step,
                        row["time_s"],
                    )
                if row["raw_maximum_principal_pa"] > principal_peak:
                    principal_peak = row["raw_maximum_principal_pa"]
                    save_frame(
                        args.output / "raw-principal-peak-state.npz",
                        scene,
                        state,
                        motion["q"][sample],
                        sample,
                        step,
                        row["time_s"],
                    )
                if failed:
                    if intersection_failed:
                        preserve_contact_failure(scene, args.output)
                    break
            completed = executed == steps and not failed
        finally:
            write_json(
                args.output / "result.json",
                {
                    "kind": "QIPC-prescribed-link-motion-candidate",
                    "config": asdict(config),
                    "requested_steps": steps,
                    "executed_steps": executed,
                    "completed_requested_history": completed,
                    "intrusive_seconds": perf_counter() - started,
                    "reference_qualified": False,
                },
            )
    if not completed:
        raise RuntimeError("Prescribed reference failed; retained partial is unqualified")


if __name__ == "__main__":
    main()
