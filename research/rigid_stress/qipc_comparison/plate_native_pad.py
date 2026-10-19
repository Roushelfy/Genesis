"""Advance the native rigid egg with measured implicit physical-pad wrenches.

Optional DPR/SCR observations use the current trajectory's live pad loads;
the original collider remains disabled for this declared boundary condition.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_R
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_recovery import NativePadRecovery
from .pad_cost import arguments as cost_arguments, write_cost
from .plates import configure_native_plate, control_native_plate, equilibrium, force_history, parser, plate_xml
from .rigid_pad_coupling import RigidPadState, carry_normal, prepare_geometry, solve


def motion(link, center_local):
    position, quaternion, velocity, angular = (
        tensor_to_array(value)[0] for value in (link.get_pos(), link.get_quat(), link.get_vel(), link.get_ang())
    )
    rotation = quat_to_R(quaternion)
    offset = rotation @ center_local
    return position, rotation, position + offset, velocity + np.cross(angular, offset), angular


def history_checks(row):
    """Check actual dynamics throughout loading, independently of settled windows."""
    return {
        "native_velocity_prediction": row["native_prediction_velocity_error_m_per_s"] <= 1e-9,
        "physical_pad_law": row["normal_law_relative_error"] <= 1e-7,
        "physical_friction_law": row["friction_law_relative_error"] <= 1e-7,
        "linear_equation": row["equation_residual_relative"] <= 1e-7,
        "angular_equation": row["angular_equation_residual_nm"] <= 1e-7 * max(row["command_force_n"], 1e-3) * 0.03,
        "frozen_point_moment": row["frozen_point_moment_defect_nm"] <= 1e-7 * max(row["command_force_n"], 1e-3) * 0.03,
    }


def main():
    cli = parser()
    cli.add_argument("--assets", type=Path, required=True)
    cli.add_argument("--method", choices=("rigid", "dpr", "scr"), default="rigid")
    cli.add_argument("--radius", type=float, default=0.002)
    cli.add_argument("--source-reduction", choices=("wrench", "none"), default="wrench")
    cli.add_argument("--friction", type=float, choices=(0.0, 0.3), default=0.0)
    cli.add_argument("--gravity-x", type=float, choices=(0.0, 1.0), default=0.0)
    cli.add_argument("--gravity-start", type=float, default=0.2)
    cli.add_argument("--gravity-ramp", type=float, default=0.05)
    cli.add_argument("--integrator", choices=("Euler", "approximate_implicitfast"), default="Euler")
    cost_arguments(cli)
    args = cli.parse_args()
    assert args.stress_stride >= 0 and args.cost_warm_levels >= 0
    assert not args.empty and not args.shear_distance and args.plate_offset_x == 0
    assert args.record_stride == 1 and (not args.gravity_x or args.friction == 0.3)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    assets = json.loads((args.assets / "result.json").read_text())
    assert file_id(args.mesh)["sha256"] == assets["source"]["sha256"]
    center_local = np.array(assets["com_m"])
    inertia_local = np.array(assets["exact_volume_inertia_kg_m2"])
    with np.load(args.mesh) as saved:
        outer, faces = saved["outer_vertices"], saved["mid_faces"]
    triangle = outer[faces]
    points = triangle.mean(axis=1)
    area = np.abs(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])[:, 2]) / 2
    lower, upper = float(outer[:, 2].min() - args.clearance), float(outer[:, 2].max() + args.clearance)
    xml = plate_xml(lower, upper)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0), dt=args.dt),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=False, enable_collision=False, integrator=getattr(gs.integrator, args.integrator)
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(
        morph=gs.morphs.MJCF(file=xml, default_armature=0.0),
        material=gs.materials.Rigid(),
    )
    egg = scene.add_entity(
        morph=gs.morphs.URDF(
            file=args.assets / "egg_shell.urdf",
            align=False,
            decimate=False,
            preserve_mesh_precision=True,
        ),
        material=gs.materials.Rigid(),
    )
    started = perf_counter()
    scene.build(n_envs=1)
    configure_native_plate(plate, args)
    link = egg.base_link
    mass_kg = link.get_mass().item()
    np.testing.assert_allclose(mass_kg, assets["mass_kg"], rtol=1e-12)
    actual_inertia = tensor_to_array(scene.rigid_solver.get_links_inertia([link.idx])).reshape(3, 3)
    np.testing.assert_allclose(np.linalg.eigvalsh(actual_inertia), np.linalg.eigvalsh(inertia_local), rtol=1e-10)
    pad = RigidPadState.allocate(points, area)
    recovery = NativePadRecovery.allocate(args, outer, faces, points, xml) if args.method != "rigid" else None
    top = points[:, 2] > 0
    sign = np.where(top, -1.0, 1.0)
    stiffness = 1e9
    write_json(
        args.output / "start.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "assets": file_id(args.assets / "result.json"),
            "native_source": gs.__file__,
            "mass_kg": mass_kg,
            "setup_seconds": perf_counter() - started,
            "scope": "Native rigid integration with a separately declared implicit physical-pad boundary condition; no default collider contact or stress accuracy claim.",
        },
    )
    rows, completed = [], False
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, command, endpoint) in enumerate(force_history(args)):
                physics_started = perf_counter()
                position, rotation, center, velocity, angular = motion(link, center_local)
                q, plate_velocity = (
                    tensor_to_array(v)[0, 0].item() for v in (plate.get_dofs_position(), plate.get_dofs_velocity())
                )
                inertia = rotation @ inertia_local @ rotation.T
                gravity = args.gravity_x * min(
                    max(((step + 1) * args.dt - args.gravity_start) / args.gravity_ramp, 0), 1
                )
                body_force = np.array([mass_kg * gravity, 0.0, 0.0])
                mass = np.zeros((7, 7))
                mass[:3, :3] = mass_kg * np.eye(3)
                mass[3:6, 3:6] = inertia / 0.03**2
                mass[6, 6] = 0.05 + args.dt * args.damping
                predicted = np.r_[
                    velocity + args.dt * body_force / mass_kg,
                    0.03 * (angular - args.dt * np.linalg.solve(inertia, np.cross(angular, inertia @ angular))),
                    (0.05 * plate_velocity + args.dt * command) / mass[6, 6],
                ]
                prepare_geometry(position, rotation, center, lower, upper - q, pad)
                iterations, solve_residual = solve(pad, mass, predicted, args.dt, stiffness, args.friction, command)
                force = qd_to_numpy(pad.force, transpose=True)
                world = qd_to_numpy(pad.world, transpose=True)
                predicted_velocity = qd_to_numpy(pad.velocity, transpose=True)
                previous_normal = qd_to_numpy(pad.previous_normal, transpose=True)
                total_force = force.sum(axis=0)
                torque = np.cross(world - center, force).sum(axis=0)
                link.apply_external_wrench(
                    force=total_force + body_force, torque=torque, ref=gs.link_ref_frame.link_COM
                )
                for name, mask in (("upper", top), ("lower", ~top)):
                    plate_link = plate.get_link(name)
                    plate_center = tensor_to_array(plate_link.get_pos())[0]
                    plate_link.apply_external_wrench(
                        force=-force[mask].sum(axis=0),
                        torque=-np.cross(world[mask] - plate_center, force[mask]).sum(axis=0),
                        ref=gs.link_ref_frame.link_COM,
                    )
                control_native_plate(plate, args, command, step + 1)
                scene.step()
                scene.rigid_solver.check_errno()
                scene.rigid_solver.update_forward_pos()
                new_position, new_rotation, new_center, new_velocity, new_angular = motion(link, center_local)
                new_q, new_plate_velocity = (
                    tensor_to_array(v)[0, 0].item() for v in (plate.get_dofs_position(), plate.get_dofs_velocity())
                )
                physics_seconds = perf_counter() - physics_started
                new_world = points @ new_rotation.T + new_position
                gap = np.where(top, upper - new_q - new_world[:, 2], new_world[:, 2] - lower)
                normal = stiffness * area * np.maximum(-gap, 0)
                relative_velocity = new_velocity + np.cross(new_angular, new_world - new_center)
                relative_velocity[top, 2] += new_plate_velocity
                speed = np.linalg.norm(relative_velocity[:, :2], axis=1)
                ratio = np.minimum(speed / 1e-4, 1)
                expected_tangent = (
                    -(args.friction * previous_normal * (2 * ratio - ratio**2))[:, None]
                    * relative_velocity[:, :2]
                    / np.maximum(speed[:, None], 1e-30)
                )
                inertia_force = mass_kg * (new_velocity - velocity) / args.dt
                angular_rate = (
                    new_rotation @ inertia_local @ new_rotation.T @ new_angular - inertia @ angular
                ) / args.dt
                scale = max(np.linalg.norm(force), 1e-12)
                actual_velocity = np.r_[new_velocity, 0.03 * new_angular, new_plate_velocity]
                row = {
                    "step": step + 1,
                    "time_s": (step + 1) * args.dt,
                    "level": level,
                    "command_force_n": command,
                    "compression_coordinate_m": new_q,
                    "plate_speed_m_per_s": new_plate_velocity,
                    "upper_reaction_n": -force[top, 2].sum().item(),
                    "lower_reaction_n": force[~top, 2].sum().item(),
                    "body_force_n": body_force.tolist(),
                    "tangential_resultant_n": total_force[:2].tolist(),
                    "egg_inertia_resultant_norm_n": float(np.linalg.norm(inertia_force)),
                    "egg_kinetic_j": float(
                        0.5 * mass_kg * new_velocity @ new_velocity
                        + 0.5 * new_angular @ new_rotation @ inertia_local @ new_rotation.T @ new_angular
                    ),
                    "equation_residual_relative": float(
                        np.linalg.norm(total_force + body_force - inertia_force) / max(command, 1e-3)
                    ),
                    "angular_equation_residual_nm": float(np.linalg.norm(torque - angular_rate)),
                    "frozen_point_moment_defect_nm": float(
                        np.linalg.norm(np.cross(new_world - world, force).sum(axis=0))
                    ),
                    "native_prediction_velocity_error_m_per_s": float(
                        np.max(np.abs(actual_velocity - predicted_velocity))
                    ),
                    "normal_law_relative_error": float(np.linalg.norm(sign * force[:, 2] - normal) / scale),
                    "friction_law_relative_error": float(np.linalg.norm(force[:, :2] - expected_tangent) / scale),
                    "newton_iterations": iterations,
                    "newton_momentum_residual": solve_residual,
                    "timing": {"physics_seconds": physics_seconds},
                }
                observe = endpoint or (args.stress_stride > 0 and (step + 1) % args.stress_stride == 0)
                if observe and recovery is not None:
                    row["stress"] = recovery.sample(
                        force,
                        new_rotation,
                        new_angular,
                        args.friction,
                        command,
                        args.output / "frames" / f"stress-level{level}.npz" if endpoint else None,
                    )
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if observe and recovery is not None:
                    assert row["stress"]["native_valid"], (
                        "Native recovery rejected the current pad loads; field retained"
                    )
                carry_normal(pad)
                if endpoint:
                    np.savez_compressed(
                        args.output / "frames" / f"level{level}.npz",
                        points_world_m=new_world,
                        applied_points_world_m=world,
                        contact_force_n=force,
                        projected_area_m2=area,
                        body_gap_m=gap,
                        center_m=new_center,
                        rotation=new_rotation,
                        velocity_m_per_s=new_velocity,
                        angular_velocity_rad_per_s=new_angular,
                        previous_normal_n=previous_normal,
                    )
            completed = True
        finally:
            windows = []
            history = [history_checks(row) for row in rows]
            for level in sorted({r["level"] for r in rows}):
                data = [r for r in rows if r["level"] == level]
                window = [r for r in data if r["time_s"] >= data[-1]["time_s"] - 0.05 - 1e-10]
                checks = [equilibrium(r, data[-1]["command_force_n"]) for r in window]
                for check, row in zip(checks, window):
                    check.update(history_checks(row))
                    body = np.array(row["body_force_n"])
                    if np.linalg.norm(body) > 0:
                        check["sustained_tangential_balance"] = bool(
                            np.linalg.norm(np.array(row["tangential_resultant_n"]) + body[:2])
                            <= 0.01 * np.linalg.norm(body)
                        )
                windows.append(
                    {
                        "level": level,
                        "samples": len(window),
                        "checks": {k: all(r[k] for r in checks) for k in checks[0]},
                        "maximum_prediction_error_m_per_s": max(
                            r["native_prediction_velocity_error_m_per_s"] for r in window
                        ),
                        "maximum_normal_law_error": max(r["normal_law_relative_error"] for r in window),
                        "maximum_friction_law_error": max(r["friction_law_relative_error"] for r in window),
                        "mechanical_window_passed": bool(
                            completed
                            and window[-1]["time_s"] - window[0]["time_s"] >= 0.05 - 1e-10
                            and all(all(check.values()) for check in checks)
                        ),
                    }
                )
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "steps": len(rows),
                    "history_checks": {k: all(row[k] for row in history) for k in history[0]} if history else {},
                    "history_failed_steps": [
                        row["step"] for row, check in zip(rows, history) if not all(check.values())
                    ],
                    "mechanical_history_passed": bool(completed and history and all(all(x.values()) for x in history)),
                    "windows": windows,
                    "elapsed_seconds": perf_counter() - started,
                    "physical_accuracy_qualified": False,
                },
            )
            write_cost(
                args,
                rows,
                completed,
                admitted=bool(
                    completed
                    and history
                    and all(all(x.values()) for x in history)
                    and all(w["mechanical_window_passed"] for w in windows)
                ),
            )


if __name__ == "__main__":
    main()
