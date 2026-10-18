"""Measure original PR22 shell compression under the shared axial force history."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.utils import array_class

from ..shell_comparison import diagnostics
from ..shell_comparison.observation import FullField
from .artifacts import file_id, runtime_identity, write_json
from .plates import configure_native_plate, control_native_plate, force_history, parser, plate_xml, shear_target


def main():
    cli = parser()
    cli.add_argument("--midpoint-mesh", type=Path, required=True)
    cli.add_argument("--iterations", type=int, default=50)
    cli.add_argument("--tolerance", type=float, default=1e-4)
    cli.add_argument("--coupling-softness", type=float, default=0.002)
    cli.add_argument("--shell-damping", type=float, default=0.0)
    cli.add_argument("--friction", type=float, default=0.0)
    args = cli.parse_args()
    assert not args.empty
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
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
            use_gjk_collision=False,
            enable_contact_patch=False,
        ),
        shell_options=gs.options.ShellOptions(
            n_pcg_iterations=args.iterations,
            pcg_threshold=args.tolerance,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml,
            default_armature=0.0,
        ),
        material=gs.materials.Rigid(
            friction=0.01,
            coup_friction=args.friction,
            coup_softness=args.coupling_softness,
        ),
    )
    egg = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=args.midpoint_mesh,
            decimate=False,
        ),
        material=gs.materials.Shell(
            rho=2000.0,
            E=1e10,
            nu=0.3,
            thickness=0.0005,
            damping=args.shell_damping,
        ),
    )
    started = perf_counter()
    scene.build(n_envs=1)
    configure_native_plate(plate, args)
    for name, height in (("lower", lower - 0.005), ("upper", upper + 0.005)):
        actual = plate.get_link(name).get_pos()[0]
        np.testing.assert_allclose(tensor_to_array(actual), [args.plate_offset_x, 0, height], rtol=0, atol=1e-12)
    solver = scene.sim.shell_solver
    diagnostics.configure(solver)
    observation = FullField.allocate(solver, "full")
    contact_increment = array_class.V(dtype=gs.qd_vec3, shape=solver.shell_state.verts_pos.shape)
    contact_product = array_class.V(dtype=gs.qd_vec3, shape=solver.shell_state.verts_pos.shape)
    observation.update(solver)
    np.savez_compressed(
        args.output / "initial-field.npz",
        cauchy_world_pa=tensor_to_array(observation.tensors)[:, 0].reshape(-1, 3, 3, 3),
        vm_pa=tensor_to_array(observation.von_mises)[:, 0].reshape(-1, 3),
    )
    rest = tensor_to_array(egg.get_verts_pos())[0]
    faces = tensor_to_array(egg.get_faces())[0]
    masses = np.bincount(faces.reshape(-1), weights=np.repeat(egg.topology.faces_mass / 3, 3), minlength=len(rest))
    np.savez_compressed(args.output / "rest.npz", vertices=rest, faces=faces, masses_kg=masses)
    write_json(
        args.output / "start.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "identity": runtime_identity(),
            "genesis_import": gs.__file__,
            "input": file_id(args.mesh),
            "midpoint_mesh": file_id(args.midpoint_mesh),
            "plate_dofs_armature": tensor_to_array(plate.get_dofs_armature()).tolist(),
            "initial_raw_vm_peak_pa": observation.von_mises.max().item(),
            "fracture_enabled": solver._static_config.has_fracture,
            "plasticity_enabled": solver._static_config.has_plasticity,
            "nominal_area_thickness_mass_kg": (egg.topology.faces_rest_area.sum() * 2000 * 0.0005).item(),
            "scope": "Original PR22 constitutive/contact kernels with read-only diagnostics and full-field observer; nominal500um wall, disclosed area-thickness mass approximation.",
            "force_measure": "Contact impulse from the last1ms scene step divided by1ms; source plate associated with nearest actual plane, with nonoverlapping contact neighborhoods.",
            "setup_seconds": perf_counter() - started,
        },
    )
    completed, executed, samples = False, 0, []
    cap_steps = 0
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, force, endpoint) in enumerate(force_history(args)):
                previous_velocity = qd_to_numpy(solver.shell_state.verts_vel, transpose=True)[0]
                control_native_plate(plate, args, force, step + 1)
                diagnostics.ACTIVE.clear()
                scene.step()
                scene.rigid_solver.check_errno()
                executed += 1
                caps = tensor_to_array(diagnostics.ACTIVE.caps)[0].item()
                cap_steps += caps > 0
                if (step + 1) % args.record_stride and not endpoint:
                    continue
                q = plate.get_dofs_position()[0, 0].item()
                speed = plate.get_dofs_velocity()[0, 0].item()
                positions = tensor_to_array(egg.get_verts_pos())[0]
                velocity = qd_to_numpy(solver.shell_state.verts_vel, transpose=True)[0]
                impulses = tensor_to_array(diagnostics.ACTIVE.impulses)[:, 0]
                points = tensor_to_array(diagnostics.ACTIVE.points)[:, 0]
                forces = impulses / args.dt
                # Only these two distant parallel plates participate in coupling.
                from_upper = np.abs(points[:, 2] - (upper - q)) < np.abs(points[:, 2] - lower)
                row = {
                    "step": step + 1,
                    "time_s": (step + 1) * args.dt,
                    "level": level,
                    "within_level_step": within + 1,
                    "command_force_n": force,
                    "plate_speed_m_per_s": speed,
                    "compression_coordinate_m": q,
                    "plate_separation_m": upper - lower - q,
                    "shear_target_m": shear_target(args, step + 1),
                    "shear_coordinate_m": plate.get_dofs_position()[0, 1].item() if args.shear_distance else 0.0,
                    "upper_reaction_n": -forces[from_upper, 2].sum().item(),
                    "lower_reaction_n": forces[~from_upper, 2].sum().item(),
                    "upper_tangential_reaction_n": (-forces[from_upper, :2].sum(axis=0)).tolist(),
                    "lower_tangential_reaction_n": (-forces[~from_upper, :2].sum(axis=0)).tolist(),
                    "egg_kinetic_j": (0.5 * masses[:, None] * velocity**2).sum().item(),
                    "egg_inertia_resultant_norm_n": np.linalg.norm(
                        (masses[:, None] * (velocity - previous_velocity) / args.dt).sum(axis=0)
                    ).item(),
                    "last_step_cap_count": caps,
                    "maximum_linear_residual_relative": tensor_to_array(diagnostics.ACTIVE.residuals)[0, 0].item(),
                    "contact_tangential_force_sum_norm_n": np.linalg.norm(forces[:, :2], axis=1).sum().item(),
                    "minimum_midpoint_height_m": positions[:, 2].min().item(),
                    "maximum_midpoint_height_m": positions[:, 2].max().item(),
                }
                if endpoint:
                    observation.update(solver)
                    tensor = tensor_to_array(observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
                    vm = tensor_to_array(observation.von_mises)[:, 0].reshape(-1, 3)
                    row["raw_vm_peak_pa"] = vm.max().item()
                    row["raw_principal_peak_pa"] = np.linalg.eigvalsh(tensor)[..., -1].max().item()
                    delta = (
                        qd_to_numpy(solver.shell_state.verts_vel, transpose=True)[0]
                        - tensor_to_array(diagnostics.ACTIVE.velocities)[:, 0]
                    )
                    delta_view = qd_to_torch(contact_increment, transpose=True, copy=False)
                    delta_view[0] = torch.as_tensor(delta, device=gs.device)
                    diagnostics.true_system_product(
                        contact_increment, contact_product, solver.shell_state, solver.shell_scratch, solver.shell_info
                    )
                    implicit_impulse = qd_to_numpy(contact_product, transpose=True)[0]
                    defect = implicit_impulse - masses[:, None] * delta
                    row["split_contact_implicit_impulse_defect_l2_ns"] = np.linalg.norm(defect).item()
                    row["split_contact_impulse_l2_ns"] = np.linalg.norm(impulses).item()
                    row["split_contact_implicit_impulse_defect_relative"] = (
                        np.linalg.norm(defect) / max(np.linalg.norm(impulses), 1e-30)
                    ).item()
                    np.savez_compressed(
                        args.output / "frames" / f"level{level}.npz",
                        vertices_world_m=positions,
                        velocities_m_per_s=velocity,
                        cauchy_world_pa=tensor,
                        vm_pa=vm,
                        contact_positions_world_m=points,
                        contact_forces_world_n=forces,
                        contact_from_upper=from_upper,
                        before_contact_velocity_m_per_s=tensor_to_array(diagnostics.ACTIVE.velocities)[:, 0],
                        contact_delta_velocity_m_per_s=delta,
                        frozen_implicit_operator_contact_impulse_ns=implicit_impulse,
                        split_contact_implicit_impulse_defect_ns=defect,
                    )
                    assert np.isfinite(tensor).all()
                assert np.isfinite(positions).all() and np.isfinite(velocity).all()
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                samples.append(row)
            completed = True
        finally:
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "steps_returned": executed,
                    "cap_exhausted_steps": cap_steps,
                    "elapsed_seconds_including_observation_IO": perf_counter() - started,
                    "last_observation": samples[-1] if samples else None,
                    "physical_accuracy_qualified": False,
                },
            )


if __name__ == "__main__":
    main()
