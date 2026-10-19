"""Measure a pinned coupled-contact Shell variant on the common force-controlled plate benchmark."""

import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd
import trimesh

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.observation import FullField
from .artifacts import file_id, runtime_identity, write_json
from .plates import configure_native_plate, control_native_plate, force_history, parser, plate_xml
from .shell_linear_capture import capture, contact_diagnostics
from .shell_pad_audit import ShellPadAudit


@qd.kernel
def observe_elastic_force(force: qd.Tensor, state: array_class.ShellState, info: array_class.ShellInfo):
    for i_v, i_b in qd.ndrange(force.shape[0], force.shape[1]):
        force[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
    for i_f, i_b in qd.ndrange(state.faces_thickness.shape[0], state.faces_thickness.shape[1]):
        i_e = info.faces_entity[i_f]
        deformation, inverse = shell.func_face_deformation(i_f, i_b, state, info)
        stress = shell.func_membrane_stress(
            deformation, info.entities_stretching_modulus[i_e] * state.faces_thickness[i_f, i_b], info.entities_nu[i_e]
        )
        product = info.faces_rest_area[i_f] * deformation @ stress @ inverse.transpose()
        first, second = -product[:, 0], -product[:, 1]
        force[state.corners_vert[3 * i_f, i_b], i_b] -= first + second
        force[state.corners_vert[3 * i_f + 1, i_b], i_b] += first
        force[state.corners_vert[3 * i_f + 2, i_b], i_b] += second
    for i_h, i_b in qd.ndrange(info.hinges_rest_angle.shape[0], state.faces_thickness.shape[1]):
        a, b, c, d, intact = shell.func_hinge_verts(i_h, i_b, state, info)
        if intact:
            edge_b, edge_c, edge_d = shell.func_hinge_edges(a, b, c, d, i_b, state)
            gradient = shell.func_hinge_angle_gradient(edge_b, edge_c, edge_d)
            delta = (
                shell.func_hinge_angle(edge_b, edge_c, edge_d)
                - info.hinges_rest_angle[i_h]
                - state.hinges_plastic_angle[i_h, i_b]
            )
            coefficient = -shell.func_hinge_stiffness(i_h, i_b, state, info) * delta
            force[a, i_b] += coefficient * gradient[:, 0]
            force[b, i_b] += coefficient * gradient[:, 1]
            force[c, i_b] += coefficient * gradient[:, 2]
            force[d, i_b] += coefficient * gradient[:, 3]


def main():
    cli = parser()
    cli.add_argument("--tolerance", type=float, default=1e-8)
    cli.add_argument("--velocity-tolerance", type=float, default=1e-9)
    cli.add_argument("--iterations", type=int, default=1000)
    cli.add_argument("--contact-stiffness", type=float, default=25.0)
    cli.add_argument("--coarse-patches", type=int, default=12)
    cli.add_argument("--friction", type=float, default=0.0)
    cli.add_argument("--midpoint-mesh", type=Path)
    cli.add_argument("--floating-sums", action="store_true")
    cli.add_argument("--capture-first-linear", action="store_true")
    cli.add_argument("--capture-linear-step", type=int, choices=(40,))
    cli.add_argument("--pad-geometry", type=Path)
    cli.add_argument("--pad-stiffness", type=float, default=1e9)
    cli.add_argument("--gravity-x", type=float, default=0.0)
    cli.add_argument("--gravity-start", type=float, default=0.2)
    cli.add_argument("--gravity-ramp", type=float, default=0.05)
    args = cli.parse_args()
    if args.gravity_x:
        assert args.pad_geometry is not None and args.friction == 0.3 and args.record_stride == 1
        assert args.gravity_x == 1.0 and args.gravity_start >= 0 and args.gravity_ramp > 0
    if args.capture_linear_step is not None:
        assert not args.capture_first_linear and args.pad_geometry is not None
        assert file_id(Path(shell.__file__))["sha256"] == (
            "5097b1c56547b907df2638ed2283797093552d8d850c893dd08e516a78010ca6"
        ), "Substep40 capture requires the isolated physical-pad diagnostic source"
    if args.capture_first_linear:
        assert (
            file_id(Path(shell.__file__))["sha256"]
            == "3f42e2d46c3d45292bba294c84e503fc4d5d04b9c6614ec5da2200f5c2bfb360"
        ), "First-linear capture requires the retained diagnostic source patch, not a complete trajectory solver"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    with np.load(args.mesh) as saved:
        vertices, mid, faces = (saved[key] for key in ("vertices", "mid_vertices", "mid_faces"))
    midpoint = args.midpoint_mesh
    if midpoint is None:
        midpoint = args.output / "midpoint.obj"
        trimesh.Trimesh(mid, faces, process=False).export(midpoint, digits=17)
    lower, upper = vertices[:, 2].min() - args.clearance, vertices[:, 2].max() + args.clearance
    xml = plate_xml(lower, upper, args.plate_offset_x, bool(args.shear_distance))
    (args.output / "plates.xml").write_text(xml)
    gs.init(
        backend=gs.cuda, precision="64", logging_level="warning", use_deterministic_algorithms=not args.floating_sums
    )
    shell_options = gs.options.ShellOptions(
        pcg_tolerance=args.tolerance,
        pcg_velocity_tolerance=args.velocity_tolerance,
        pcg_max_iterations=args.iterations,
        contact_stiffness=args.contact_stiffness,
        n_coarse_patches=args.coarse_patches,
    )
    if args.pad_geometry is not None:
        assert args.floating_sums and args.pad_stiffness == 1e9
        shell_options.contact_pad_geometry = str(args.pad_geometry)
        shell_options.contact_pad_stiffness = args.pad_stiffness
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            iterations=100,
            tolerance=1e-12,
            use_gjk_collision=False,
            enable_contact_patch=False,
        ),
        shell_options=shell_options,
        show_viewer=False,
    )
    plate = scene.add_entity(
        gs.morphs.MJCF(file=xml, default_armature=0.0),
        material=gs.materials.Rigid(friction=0.01, coup_friction=args.friction),
    )
    egg = scene.add_entity(
        gs.morphs.Mesh(file=midpoint, decimate=False),
        material=gs.materials.Shell(rho=2000.0, E=1e10, nu=0.3, thickness=0.0005, damping=0.0),
    )
    started = time.perf_counter()
    scene.build(n_envs=1)
    configure_native_plate(plate, args)
    solver = scene.sim.shell_solver
    field = FullField.allocate(solver, "full")
    internal_buffer = array_class.V(dtype=gs.qd_vec3, shape=solver.shell_state.verts_pos.shape)
    rest = tensor_to_array(egg.get_verts_pos())[0]
    faces = tensor_to_array(egg.get_faces())[0]
    masses = np.bincount(faces.reshape(-1), weights=np.repeat(egg.topology.faces_mass / 3, 3), minlength=len(rest))
    pad_audit = (
        ShellPadAudit.load(args.pad_geometry, faces, args.pad_stiffness, lower, upper) if args.pad_geometry else None
    )
    np.savez_compressed(args.output / "rest.npz", vertices=rest, faces=faces, masses_kg=masses)
    write_json(
        args.output / "start.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "identity": runtime_identity(),
            "genesis_import": gs.__file__,
            "mesh": file_id(args.mesh),
            "midpoint": file_id(midpoint),
            "setup_seconds": time.perf_counter() - started,
            "mass_kg": float(masses.sum()),
            "variant": "Captured coupled-contact Shell source, identified by source.json and source.patch; unchanged constitutive model, separate from original PR22.",
            "solver_source": file_id(Path(shell.__file__)),
            "physical_pad_geometry": file_id(args.pad_geometry) if args.pad_geometry is not None else None,
            "physical_pad_stiffness_n_per_m3": args.pad_stiffness if args.pad_geometry is not None else None,
            "force_observer": "Fresh membrane and hinge forces into observer-owned storage; no cache or velocity writes.",
        },
    )
    completed, executed, samples = False, 0, []
    started = time.perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, force, endpoint) in enumerate(force_history(args)):
                previous_velocity = tensor_to_array(egg.get_verts_vel())[0]
                gravity_x = args.gravity_x * min(
                    max(((step + 1) * args.dt - args.gravity_start) / args.gravity_ramp, 0), 1
                )
                if args.gravity_x:
                    solver.set_gravity((gravity_x, 0.0, 0.0))
                control_native_plate(plate, args, force, step + 1)
                scene.step()
                scene.rigid_solver.check_errno()
                executed += 1
                if args.capture_first_linear:
                    capture(solver, scene.rigid_solver, args.output)
                    return
                if args.capture_linear_step == step + 1:
                    assert qd_to_numpy(solver._shell_scratch.envs_n_solve_iterations, transpose=True)[0] > 0, (
                        "Capture must precede the next Newton preparation; zero iterations indicates a wrong capture point"
                    )
                    capture(solver, scene.rigid_solver, args.output, physical_pad=True, step=step + 1)
                    return
                failure_flags = tensor_to_array(solver.get_envs_solver_failure())
                failed = bool(failure_flags.any())
                if (step + 1) % args.record_stride and not endpoint and not failed:
                    continue
                positions, velocity, contact = (
                    tensor_to_array(value)[0]
                    for value in (egg.get_verts_pos(), egg.get_verts_vel(), egg.get_verts_contact_force())
                )
                observe_elastic_force(internal_buffer, solver.shell_state, solver.shell_info)
                internal = qd_to_numpy(internal_buffer, transpose=True)[0]
                inertia = masses[:, None] * (velocity - previous_velocity) / args.dt
                body_force = masses[:, None] * np.array([gravity_x, 0.0, 0.0])
                residual = internal + contact + body_force - inertia
                scale = max(np.linalg.norm(contact), abs(force))
                from_upper = positions[:, 2] > (lower + upper) / 2
                row = {
                    "step": step + 1,
                    "time_s": (step + 1) * args.dt,
                    "level": level,
                    "command_force_n": force,
                    "compression_coordinate_m": plate.get_dofs_position()[0, 0].item(),
                    "plate_speed_m_per_s": plate.get_dofs_velocity()[0, 0].item(),
                    "upper_reaction_n": -contact[from_upper, 2].sum().item(),
                    "lower_reaction_n": contact[~from_upper, 2].sum().item(),
                    "egg_inertia_resultant_norm_n": float(np.linalg.norm(inertia.sum(axis=0))),
                    "egg_kinetic_j": (0.5 * masses[:, None] * velocity**2).sum().item(),
                    "equation_residual_relative": float(np.linalg.norm(residual) / scale) if scale else None,
                    "static_residual_relative": float(np.linalg.norm(internal + contact + body_force) / scale)
                    if scale
                    else None,
                    "solver_failure_flags": failure_flags.tolist(),
                    "last_pcg_iterations": tensor_to_array(solver.get_envs_pcg_iterations()).tolist(),
                    "contact_force_norm_n": float(np.linalg.norm(contact)),
                    "residual_force_scale_n": float(scale),
                    "internal_force_resultant_n": internal.sum(axis=0).tolist(),
                    "contact_force_resultant_n": contact.sum(axis=0).tolist(),
                    "body_force_resultant_n": body_force.sum(axis=0).tolist(),
                    "mass_center_velocity_m_per_s": (masses @ velocity / masses.sum()).tolist(),
                    "external_moment_about_mass_center_nm": np.cross(
                        positions - masses @ positions / masses.sum(), contact + body_force
                    )
                    .sum(axis=0)
                    .tolist(),
                }
                pad_arrays = {}
                if pad_audit is not None:
                    row["physical_pad"], pad_arrays = pad_audit.evaluate(
                        solver,
                        positions,
                        contact,
                        row["compression_coordinate_m"],
                        args.dt,
                        velocity=velocity if args.gravity_x else None,
                        friction=args.friction,
                    )
                if failed:
                    row["failed_solve_counters"] = {
                        "contact_newton_iterations": qd_to_numpy(
                            solver._shell_contact.envs_n_newton_iterations, transpose=True
                        ).tolist(),
                        "last_linear_iterations": qd_to_numpy(
                            solver._shell_scratch.envs_n_solve_iterations, transpose=True
                        ).tolist(),
                        "linear_mode": qd_to_numpy(solver._shell_scratch.envs_pcg_mode, transpose=True).tolist(),
                        "last_checked_residual": qd_to_numpy(
                            solver._shell_scratch.envs_residual_checked, transpose=True
                        ).tolist(),
                        "relative_residual_threshold": qd_to_numpy(
                            solver._shell_scratch.envs_residual_threshold, transpose=True
                        ).tolist(),
                    }
                if endpoint or failed:
                    row["contact_iteration_audit"] = contact_diagnostics(
                        solver,
                        scene.rigid_solver,
                        args.dt,
                        args.velocity_tolerance,
                        args.contact_stiffness,
                        physical_pad=pad_audit is not None,
                    )
                    field.update(solver)
                    tensors = tensor_to_array(field.tensors)[:, 0].reshape(-1, 3, 3, 3)
                    vm = tensor_to_array(field.von_mises)[:, 0].reshape(-1, 3)
                    row["raw_vm_peak_pa"] = vm.max().item()
                    row["raw_principal_peak_pa"] = np.linalg.eigvalsh(tensors)[..., -1].max().item()
                    np.savez_compressed(
                        args.output / "frames" / (f"rejected-step{step + 1}.npz" if failed else f"level{level}.npz"),
                        vertices_world_m=positions,
                        velocities_m_per_s=velocity,
                        cauchy_world_pa=tensors,
                        vm_pa=vm,
                        contact_forces_world_n=contact,
                        internal_forces_world_n=internal,
                        body_forces_world_n=body_force,
                        inertial_forces_world_n=inertia,
                        equation_residual_n=residual,
                        membrane_n_per_m=tensor_to_array(field.membrane)[:, 0],
                        bending_n=tensor_to_array(field.bending)[:, 0],
                        **pad_arrays,
                    )
                assert np.isfinite(positions).all() and np.isfinite(velocity).all()
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                samples.append(row)
                assert not failed, "Shell solver failure: full field retained at the first failing step"
            completed = True
        finally:
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "steps_returned": executed,
                    "elapsed_seconds_including_observation_IO": time.perf_counter() - started,
                    "last_observation": samples[-1] if samples else None,
                    "physical_accuracy_qualified": False,
                    "scope": "Actual end-position elastic/contact/inertial force check with zero shell damping. "
                    "The engine linearizes elasticity each substep; this residual measures its remaining "
                    "nonlinear time discretization error as well as contact balance. No velocity clearing.",
                },
            )


if __name__ == "__main__":
    main()
