"""Observe native DPR/SCR under common-force parallel-plate compression."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.memory import ProcessMemory
from .artifacts import file_id, runtime_identity, write_json
from .plates import configure_native_plate, control_native_plate, force_history, parser, plate_xml, shear_target


def main():
    cli = parser()
    cli.add_argument("--method", choices=("dpr", "scr", "rigid"), default="dpr")
    cli.add_argument("--assets", type=Path, required=True)
    cli.add_argument("--contact-radius", type=float, default=0.006)
    cli.add_argument("--friction", type=float, default=0.01)
    cli.add_argument("--envs", type=int, default=1)
    cli.add_argument("--cost-steps", type=int, default=0)
    cli.add_argument("--cost-stride", type=int, default=100)
    cli.add_argument("--cost-repeats", type=int, default=3)
    cli.add_argument("--profile-steps", type=int, default=0)
    cli.add_argument("--memory", action="store_true")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    memory = ProcessMemory(args.memory)
    memory.start()
    with np.load(args.mesh) as mesh:
        vertices = mesh["vertices"]
    lower, upper = vertices[:, 2].min() - args.clearance, vertices[:, 2].max() + args.clearance
    xml = plate_xml(lower, upper, args.plate_offset_x, bool(args.shear_distance))
    (args.output / "plates.xml").write_text(xml)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            iterations=100,
            tolerance=1e-12,
            use_hibernation=False,
            use_gjk_collision=False,
            enable_contact_patch=False,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml,
            default_armature=0.0,
        ),
        material=gs.materials.Rigid(
            friction=args.friction,
        ),
    )
    egg, link = None, None
    if not args.empty:
        egg = scene.add_entity(
            morph=gs.morphs.URDF(
                file=args.assets / "egg_shell.urdf",
                align=False,
                decimate=False,
                preserve_mesh_precision=True,
            ),
            material=gs.materials.Rigid(
                friction=args.friction,
            ),
        )
        link = egg.base_link
        if args.method != "rigid":
            link.configure_stress_recovery(
                gs.options.RigidStressOptions(
                    mesh=args.assets / "elastic.npz",
                    contact_model="pressure" if args.method == "dpr" else "intrinsic",
                    contact_radius=args.contact_radius,
                    intrinsic_load_partition="body_source_prior",
                    output_mode="full",
                    update_mode="manual",
                )
            )
    started = perf_counter()
    scene.build(n_envs=args.envs)
    configure_native_plate(plate, args)
    indices = {link.name: link.idx for link in plate.links}
    for name, height in (("lower", lower - 0.005), ("upper", upper + 0.005)):
        actual = plate.get_link(name).get_pos()[0]
        np.testing.assert_allclose(tensor_to_array(actual), [args.plate_offset_x, 0, height], rtol=0, atol=1e-12)
    write_json(
        args.output / "start.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "identity": runtime_identity(),
            "genesis_import": gs.__file__,
            "input": file_id(args.mesh),
            "elastic_mesh": file_id(args.assets / "elastic.npz") if egg is not None else None,
            "egg_rigid_mass_kg": link.get_mass().tolist() if egg is not None else None,
            "stress_options": link.stress_options.model_dump(mode="json")
            if egg is not None and args.method != "rigid"
            else None,
            "plate_link_indices": indices,
            "plate_dofs_armature": tensor_to_array(plate.get_dofs_armature()).tolist(),
            "plate_dofs_damping": tensor_to_array(plate.get_dofs_damping()).tolist(),
            "native_friction_floor": 0.01,
            "scope": "Original native contact with measured mu=0.01 floor; not an exact frictionless model.",
            "setup_seconds": perf_counter() - started,
        },
    )
    completed, executed, samples = False, 0, []
    predicted_velocity, predicted_q = 0.0, 0.0
    entry = None
    if egg is not None and args.method != "rigid":
        entry = scene.rigid_solver.stress_recovery.links[0]
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, force, endpoint) in enumerate(force_history(args)):
                previous_velocity = egg.get_vel().clone() if egg is not None else None
                control_native_plate(plate, args, force, step + 1)
                if endpoint and entry is not None:
                    link.request_stress_update()
                scene.step()
                scene.rigid_solver.check_errno()
                executed += 1
                moving_mass = 0.051 if args.shear_distance else 0.05
                predicted_velocity = (moving_mass * predicted_velocity + args.dt * force) / (
                    moving_mass + args.dt * args.damping
                )
                predicted_q += args.dt * predicted_velocity
                if (step + 1) % args.record_stride and not endpoint:
                    continue
                velocity = plate.get_dofs_velocity()[0, 0].item()
                q = plate.get_dofs_position()[0, 0].item()
                row = {
                    "step": step + 1,
                    "time_s": (step + 1) * args.dt,
                    "level": level,
                    "within_level_step": within + 1,
                    "command_force_n": force,
                    "plate_speed_m_per_s": velocity,
                    "compression_coordinate_m": q,
                    "plate_separation_m": upper - lower - q,
                    "shear_target_m": shear_target(args, step + 1),
                    "shear_coordinate_m": plate.get_dofs_position()[0, 1].item() if args.shear_distance else 0.0,
                    "empty_analytic_position_error_m": q - predicted_q if args.empty else None,
                    "empty_analytic_velocity_error_m_per_s": velocity - predicted_velocity if args.empty else None,
                }
                if egg is not None:
                    contacts = {key: tensor_to_array(value)[0] for key, value in egg.get_contacts().items()}
                    valid = contacts["valid_mask"]
                    egg_is_a = contacts["link_a"] == link.idx
                    forces = np.where(egg_is_a[:, None], contacts["force_a"], contacts["force_b"])[valid]
                    points = contacts["position"][valid]
                    counterparts = np.where(egg_is_a, contacts["link_b"], contacts["link_a"])[valid]
                    reactions = {name: -forces[counterparts == index].sum(axis=0) for name, index in indices.items()}
                    mass = link.get_mass().item()
                    linear_velocity = egg.get_vel()[0]
                    angular_velocity = egg.get_ang()[0]
                    trace_inertia = (
                        tensor_to_array(scene.rigid_solver.get_links_inertia([link.idx]))
                        .reshape(-1, 3, 3)[0]
                        .trace()
                        .item()
                    )
                    inertia = mass * torch.linalg.norm((linear_velocity - previous_velocity[0]) / args.dt).item()
                    row.update(
                        upper_reaction_n=reactions["upper"][2].item(),
                        lower_reaction_n=-reactions["lower"][2].item(),
                        upper_tangential_reaction_n=reactions["upper"][:2].tolist(),
                        lower_tangential_reaction_n=reactions["lower"][:2].tolist(),
                        egg_inertia_resultant_norm_n=inertia,
                        egg_translational_kinetic_j=(0.5 * mass * (linear_velocity**2).sum()).item(),
                        egg_kinetic_upper_bound_j=(
                            0.5 * mass * (linear_velocity**2).sum() + 0.5 * trace_inertia * (angular_velocity**2).sum()
                        ).item(),
                        egg_angular_velocity_rad_per_s=tensor_to_array(angular_velocity).tolist(),
                        maximum_contact_friction=contacts["friction"][valid].max().item() if len(forces) else None,
                        contact_tangential_force_sum_norm_n=np.linalg.norm(forces[:, :2], axis=1).sum().item(),
                        contact_count=len(forces),
                    )
                    if endpoint:
                        payload = {
                            "link_position_m": tensor_to_array(egg.get_pos())[0],
                            "link_quaternion_wxyz": tensor_to_array(egg.get_quat())[0],
                            "contact_positions_world_m": points,
                            "contact_forces_world_n": forces,
                            "contact_counterpart_link": counterparts,
                            "contact_normals": contacts["normal"][valid],
                            "contact_friction": contacts["friction"][valid],
                            "contact_penetration_m": contacts["penetration"][valid],
                        }
                        if entry is not None:
                            stress, vm = link.get_stress_field()
                            residual = qd_to_numpy(entry.state.residual, transpose=True)[0]
                            rhs = qd_to_numpy(entry.state.rhs, transpose=True)[0]
                            row.update(
                                raw_vm_peak_pa=vm.max().item(),
                                equation_residual_relative=np.linalg.norm(residual).item()
                                / max(np.linalg.norm(rhs).item(), 1e-12),
                                invalid_stress_steps=qd_to_numpy(entry.state.invalid_steps, transpose=True)[0].item(),
                            )
                            payload.update(
                                stress_voigt_link_pa=tensor_to_array(stress)[0],
                                vm_pa=tensor_to_array(vm)[0],
                                residual_n=residual,
                                rhs_n=rhs,
                            )
                            assert torch.isfinite(stress).all()
                        np.savez_compressed(args.output / "frames" / f"level{level}.npz", **payload)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                samples.append(row)
                if args.empty and endpoint:
                    assert abs(row["empty_analytic_position_error_m"]) < 1e-12
                    assert abs(row["empty_analytic_velocity_error_m_per_s"]) < 1e-10
            if args.cost_steps:
                from .plate_cost import measure

                measure(args, scene, plate, egg, link, entry)
            completed = True
        except Exception as error:
            failure = {
                "exception": repr(error),
                "attempted_step": step + 1,
                "load_level": level,
                "command_force_n": force,
                "dump_errors": [],
            }
            payload = {}
            if egg is not None:
                payload.update(
                    {
                        "position_m": tensor_to_array(egg.get_pos()),
                        "quaternion_wxyz": tensor_to_array(egg.get_quat()),
                        "velocity_m_per_s": tensor_to_array(egg.get_vel()),
                    }
                )
                payload.update({"contact_" + key: tensor_to_array(value) for key, value in egg.get_contacts().items()})
            if entry is not None:
                from genesis.utils.array_class import DataKind

                for item in scene.rigid_solver.stress_recovery.data:
                    if item.kind in (DataKind.CONFIG, DataKind.CONSTANT, DataKind.INFO):
                        continue
                    try:
                        payload[item.name] = qd_to_numpy(item.value, transpose=True)
                    except Exception as dump_error:
                        failure["dump_errors"].append({"field": item.name, "exception": repr(dump_error)})
            np.savez_compressed(args.output / "rejected-state.npz", **payload)
            write_json(args.output / "rejected-state.json", failure)
            raise
        finally:
            memory_report = memory.finish()
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "steps_returned": executed,
                    "elapsed_seconds_including_observation_IO": perf_counter() - started,
                    "last_observation": samples[-1] if samples else None,
                    "physical_accuracy_qualified": False,
                    "process_memory": memory_report,
                },
            )


if __name__ == "__main__":
    main()
