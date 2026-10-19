"""Complete admitted physical-pad Shell workflow in independent native environments."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import trimesh

import genesis as gs
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.observation import FullField
from .artifacts import file_id, runtime_identity, write_json
from .pad_primary_cost import PrimaryCost
from .plate_shell_coupled import arguments, observe_elastic_force
from .plates import configure_native_plate, equilibrium, force_history, plate_xml
from .shell_pad_audit import ShellPadAudit


def main():
    cli = arguments()
    cli.add_argument("--envs", type=int, choices=(1, 2, 4, 8), required=True)
    cli.add_argument("--env-load-scales", type=float, nargs="+")
    args = cli.parse_args()
    assert args.primary_cost and args.record_stride == 1 and args.pad_geometry is not None
    assert args.floating_sums and args.pad_stiffness == 1e9 and args.friction == args.gravity_x == 0
    assert not args.empty and not args.shear_distance and args.plate_offset_x == 0
    assert not args.paired_output_check and not args.capture_first_linear and args.capture_linear_step is None
    scales = np.ones(args.envs) if args.env_load_scales is None else np.array(args.env_load_scales)
    assert scales.shape == (args.envs,) and np.all(scales > 0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    with np.load(args.mesh) as saved:
        vertices, mid, faces = (saved[key] for key in ("vertices", "mid_vertices", "mid_faces"))
    midpoint = args.midpoint_mesh
    if midpoint is None:
        midpoint = args.output / "midpoint.obj"
        trimesh.Trimesh(mid, faces, process=False).export(midpoint, digits=17)
    lower, upper = vertices[:, 2].min() - args.clearance, vertices[:, 2].max() + args.clearance
    xml = plate_xml(lower, upper)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning", use_deterministic_algorithms=False)
    setup = perf_counter()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            iterations=100,
            tolerance=1e-12,
            use_gjk_collision=False,
            enable_contact_patch=False,
        ),
        shell_options=gs.options.ShellOptions(
            pcg_tolerance=args.tolerance,
            pcg_velocity_tolerance=args.velocity_tolerance,
            pcg_max_iterations=args.iterations,
            contact_stiffness=args.contact_stiffness,
            n_coarse_patches=args.coarse_patches,
            contact_pad_geometry=str(args.pad_geometry),
            contact_pad_stiffness=args.pad_stiffness,
        ),
        show_viewer=False,
    )
    plate = scene.add_entity(
        gs.morphs.MJCF(file=xml, default_armature=0.0), material=gs.materials.Rigid(friction=0.01, coup_friction=0.0)
    )
    egg = scene.add_entity(
        gs.morphs.Mesh(file=midpoint, decimate=False),
        material=gs.materials.Shell(rho=2000.0, E=1e10, nu=0.3, thickness=0.0005, damping=0.0),
    )
    scene.build(n_envs=args.envs)
    configure_native_plate(plate, args)
    solver = scene.sim.shell_solver
    field = FullField.allocate(solver, args.stress_output)
    internal_buffer = array_class.V(dtype=gs.qd_vec3, shape=solver.shell_state.verts_pos.shape)
    all_rest = tensor_to_array(egg.get_verts_pos())
    for rest in all_rest:
        np.testing.assert_array_equal(rest, all_rest[0])
    rest = all_rest[0]
    faces = tensor_to_array(egg.get_faces())[0]
    masses = np.bincount(faces.reshape(-1), weights=np.repeat(egg.topology.faces_mass / 3, 3), minlength=len(rest))
    audits = [ShellPadAudit.load(args.pad_geometry, faces, args.pad_stiffness, lower, upper) for _ in range(args.envs)]
    np.savez_compressed(args.output / "rest.npz", vertices=rest, faces=faces, masses_kg=masses)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "mesh": file_id(args.mesh),
            "pad_geometry": file_id(args.pad_geometry),
            "setup_seconds": perf_counter() - setup,
            "mass_kg_per_env": float(masses.sum()),
            "scope": "Normal physical-pad benchmark with independent per-environment plate commands, Shell motion, coupled contact and complete-field observation. One batched scene step; unchanged captured Shell engine and constitutive model. This driver currently measures normal loading only.",
        },
    )
    primary = PrimaryCost(args)
    histories, clock_rows, completed = [[] for _ in range(args.envs)], [], False
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, command, endpoint) in enumerate(force_history(args)):
                primary.begin(level, step)
                commands = command * scales
                previous = tensor_to_array(egg.get_verts_vel())
                plate.control_dofs_force(commands[:, None])
                scene.step()
                scene.rigid_solver.check_errno()
                failures = tensor_to_array(solver.get_envs_solver_failure())
                positions, velocity, contact = map(
                    tensor_to_array, (egg.get_verts_pos(), egg.get_verts_vel(), egg.get_verts_contact_force())
                )
                observe_elastic_force(internal_buffer, solver.shell_state, solver.shell_info)
                internal = qd_to_numpy(internal_buffer, transpose=True)
                inertial = masses[None, :, None] * (velocity - previous) / args.dt
                residual = internal + contact - inertial
                compression, plate_velocity = [
                    tensor_to_array(v)[:, 0] for v in (plate.get_dofs_position(), plate.get_dofs_velocity())
                ]
                contact_data = (
                    qd_to_numpy(solver._shell_contact.contacts_geom, transpose=True),
                    qd_to_numpy(solver._shell_contact.contacts_impulse, transpose=True),
                )
                observe = endpoint or np.any(failures) or (step + 1) % args.stress_stride == 0
                field_arrays = [{} for _ in range(args.envs)]
                if observe:
                    field.update(solver)
                    peaks = tensor_to_array(field.peaks)
                    if args.stress_output == "full":
                        tensors = tensor_to_array(field.tensors).transpose(1, 0, 2, 3).reshape(args.envs, -1, 3, 3, 3)
                        vm = tensor_to_array(field.von_mises).T.reshape(args.envs, -1, 3)
                        for env in range(args.envs):
                            field_arrays[env] = {"cauchy_world_pa": tensors[env], "vm_pa": vm[env]}
                rows = []
                for env in range(args.envs):
                    top = positions[env, :, 2] > (lower + upper) / 2
                    scale = max(np.linalg.norm(contact[env]), abs(commands[env]))
                    center = masses @ positions[env] / masses.sum()
                    moment = np.cross(positions[env] - center, contact[env] - inertial[env]).sum(axis=0)
                    row = {
                        "step": step + 1,
                        "time_s": (step + 1) * args.dt,
                        "level": level,
                        "env": env,
                        "command_force_n": float(commands[env]),
                        "compression_coordinate_m": float(compression[env]),
                        "plate_speed_m_per_s": float(plate_velocity[env]),
                        "upper_reaction_n": float(-contact[env, top, 2].sum()),
                        "lower_reaction_n": float(contact[env, ~top, 2].sum()),
                        "egg_inertia_resultant_norm_n": float(np.linalg.norm(inertial[env].sum(axis=0))),
                        "egg_kinetic_j": float(0.5 * np.einsum("v,vi,vi->", masses, velocity[env], velocity[env])),
                        "equation_residual_relative": float(np.linalg.norm(residual[env]) / max(scale, 1e-30)),
                        "solver_failure_flags": [int(failures[env])],
                        "current_moment_relative": float(np.linalg.norm(moment) / (max(commands[env], 1e-3) * 0.03)),
                    }
                    row["physical_pad"], pad_arrays = audits[env].evaluate(
                        solver,
                        positions[env],
                        contact[env],
                        compression[env],
                        args.dt,
                        env=env,
                        contact_data=contact_data,
                    )
                    if observe:
                        row["stress"] = {
                            "valid": bool(
                                np.isfinite(peaks[env])
                                and not failures[env]
                                and all(np.isfinite(a).all() for a in field_arrays[env].values())
                            ),
                            "raw_vm_peak_pa": float(peaks[env]),
                            "field_host_bytes": sum(a.nbytes for a in field_arrays[env].values()),
                            "output_mode": args.stress_output,
                        }
                    if endpoint or failures[env]:
                        np.savez_compressed(
                            args.output / "frames" / f"env{env}-level{level}.npz",
                            vertices_world_m=positions[env],
                            velocities_m_per_s=velocity[env],
                            contact_forces_world_n=contact[env],
                            internal_forces_world_n=internal[env],
                            body_forces_world_n=np.zeros_like(internal[env]),
                            inertial_forces_world_n=inertial[env],
                            equation_residual_n=residual[env],
                            raw_vm_peak_pa=peaks[env],
                            **field_arrays[env],
                            **pad_arrays,
                        )
                    rows.append(row)
                    histories[env].append(row)
                journal.write(
                    json.dumps({"step": step + 1, "level": level, "environments": rows}, allow_nan=False) + "\n"
                )
                clock_row = {"level": level}
                if observe:
                    clock_row["stress"] = {
                        "valid": all(r["stress"]["valid"] for r in rows),
                        "field_host_bytes": sum(r["stress"]["field_host_bytes"] for r in rows),
                    }
                    assert clock_row["stress"]["valid"], "Batched Shell field rejected; retain every environment"
                assert not np.any(failures), "Batched Shell solver failure; retain frames"
                clock_rows.append(clock_row)
                primary.end(level, step, endpoint)
            completed = True
        finally:
            environments = []
            for env, rows in enumerate(histories):
                bad = [
                    r["step"]
                    for r in rows
                    if any(r["solver_failure_flags"])
                    or r["equation_residual_relative"] > 1e-3
                    or not r["physical_pad"]["normal_law_and_transfer_passed"]
                    or r["current_moment_relative"] > 1e-7
                ]
                windows = []
                for level in sorted({r["level"] for r in rows}):
                    selected = [r for r in rows if r["level"] == level]
                    window = [r for r in selected if r["time_s"] >= selected[-1]["time_s"] - 0.05 - 1e-10]
                    checks = [equilibrium(row, selected[-1]["command_force_n"]) for row in window]
                    windows.append(
                        {
                            "level": level,
                            "samples": len(window),
                            "checks": {k: all(c[k] for c in checks) for k in checks[0]},
                            "mechanical_window_passed": bool(
                                completed
                                and window[-1]["time_s"] - window[0]["time_s"] >= 0.05 - 1e-10
                                and all(all(c.values()) for c in checks)
                            ),
                        }
                    )
                environments.append(
                    {
                        "env": env,
                        "steps": len(rows),
                        "history_failed_steps": bad,
                        "mechanical_history_passed": bool(completed and rows and not bad),
                        "windows": windows,
                    }
                )
            admitted = bool(
                completed
                and all(
                    e["mechanical_history_passed"] and all(w["mechanical_window_passed"] for w in e["windows"])
                    for e in environments
                )
            )
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "mechanically_admitted": admitted,
                    "envs": args.envs,
                    "environments": environments,
                    "elapsed_seconds": perf_counter() - started,
                    "physical_accuracy_qualified": False,
                },
            )
            primary.write(clock_rows, completed, admitted)
    assert admitted, "Batched Shell motion failed admission; do not use its throughput"


if __name__ == "__main__":
    main()
