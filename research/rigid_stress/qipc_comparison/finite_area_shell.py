"""Static Shell elasticity replay of independent finite-area tractions; no contact-history claim."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils import array_class
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..shell_comparison.observation import FullField
from ..shell_comparison.prescribed import add_prescribed_force
from .artifacts import file_id, runtime_identity, write_json
from .plate_shell_coupled import observe_elastic_force
from .shell_load_replay import ShellLoadTransfer, wrench
from .static_inertia_relief import StaticInertiaRelief
from .pad_equilibrium import normal_pad_equilibrium, shell_pad_boundary_gaps


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--field", type=Path, required=True)
    cli.add_argument("--dt", type=float, default=0.001)
    cli.add_argument("--outer-iterations", type=int, default=80)
    cli.add_argument("--tolerance", type=float, default=1e-4)
    cli.add_argument("--velocity-tolerance", type=float, default=1e-10)
    cli.add_argument("--force-tolerance", type=float, default=1e-7)
    cli.add_argument("--iterations", type=int, default=8000)
    cli.add_argument("--coarse-patches", type=int, default=0)
    cli.add_argument("--solve-pad-contact", action="store_true")
    cli.add_argument(
        "--inertia-relief",
        action="store_true",
        help="Apply and record the free-reference six-mode mass correction, bounded by1e-8 of load norm",
    )
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert args.dt > 0 and args.outer_iterations > 0 and args.coarse_patches >= 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    with np.load(args.mesh) as saved:
        outer, midpoint, faces = saved["outer_vertices"], saved["mid_vertices"], saved["mid_faces"]
    with np.load(args.field) as saved:
        points, point_forces = saved["traction_points_m"], saved["traction_point_forces_n"]
        load = saved["load_n"].item()
        if args.solve_pad_contact:
            pad_area = saved["traction_quadrature_area_m2"]
            pad_stiffness = saved["pad_stiffness_n_per_m3"].item()
            assert pad_stiffness == 1e9, "This case uses the predeclared physical pad"
    located = ShellLoadTransfer.allocate(outer, faces, points, 1)
    relative = midpoint[located.selected] - midpoint[located.selected].mean(axis=1, keepdims=True)
    gram = np.eye(3) * (relative**2).sum(axis=(1, 2))[:, None, None] - np.einsum("pki,pkj->pij", relative, relative)
    transfer = ShellLoadTransfer(midpoint, located.selected, located.barycentric, relative, np.linalg.inv(gram))
    force, defect = transfer.apply(points, point_forces)
    local_force = transfer.local_loads(points, point_forces)
    np.testing.assert_allclose(local_force.sum(axis=1), point_forces, rtol=1e-11, atol=1e-16)
    np.testing.assert_allclose(
        np.cross(midpoint[transfer.selected], local_force).sum(axis=1),
        np.cross(points, point_forces),
        rtol=1e-11,
        atol=1e-18,
    )
    # Rigid virtual work is preserved; nonrigid through-wall work need not agree.
    translation, rotation = np.array([0.2, -0.3, 0.7]), np.array([0.5, 0.4, -0.2])
    source_work = np.sum(point_forces * (translation + np.cross(rotation, points)))
    shell_work = np.sum(force * (translation + np.cross(rotation, midpoint)))
    np.testing.assert_allclose(source_work, shell_work, atol=1e-12)
    if not args.inertia_relief:
        np.testing.assert_allclose(wrench(midpoint, force), 0, atol=1e-12)
    pad_coordinates = None
    pad_receipt = None
    if args.solve_pad_contact:
        source_pad = json.loads((args.field.parent.parent / "result.json").read_text())
        assert source_pad["mesh"]["sha256"] == file_id(args.mesh)["sha256"]
        omitted_faces = (np.linalg.norm(outer[faces, :2], axis=-1) > source_pad["candidate_radius_m"]).all(axis=1)
        assert omitted_faces.any()
        # Force transfer and gap interpolation are exact transposes. This
        # linear reference-geometry thin-surface map preserves rigid work;
        # its nonrigid thickness kinematics remain a declared Shell reduction.
        unit_axial = np.zeros_like(points)
        unit_axial[:, 2] = 1
        normal_weights = transfer.local_loads(points, unit_axial)
        point_top = points[:, 2] > 0
        point_sign = np.where(point_top, -1.0, 1.0)
        initial_gap = np.where(point_top, outer[:, 2].max() - points[:, 2], points[:, 2] - outer[:, 2].min())
        pad_rigid = np.column_stack(
            (point_sign, point_sign * points[:, 1] / 0.03, -point_sign * points[:, 0] / 0.03, point_top)
        )
        # Check the transpose on a nonrigid virtual displacement, in addition
        # to the existing pointwise force/moment and rigid-work checks.
        virtual = np.random.default_rng(49).normal(size=midpoint.shape)
        point_virtual_z = np.sum(normal_weights * virtual[transfer.selected], axis=(1, 2))
        np.testing.assert_allclose(np.sum(force * virtual), np.sum(point_forces[:, 2] * point_virtual_z), atol=1e-12)
    surface = args.output / "midpoint.obj"
    trimesh.Trimesh(midpoint, faces, process=False).export(surface, digits=17)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning", use_deterministic_algorithms=False)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=args.dt, gravity=(0, 0, 0)),
        shell_options=gs.options.ShellOptions(
            pcg_tolerance=args.tolerance,
            pcg_velocity_tolerance=args.velocity_tolerance,
            pcg_max_iterations=args.iterations,
            n_coarse_patches=args.coarse_patches,
        ),
        show_viewer=False,
    )
    egg = scene.add_entity(
        morph=gs.morphs.Mesh(file=surface, decimate=False),
        material=gs.materials.Shell(rho=2000, E=1e10, nu=0.3, thickness=0.0005, damping=0),
    )
    scene.build(n_envs=1)
    solver = scene.sim.shell_solver
    rest, actual_faces = tensor_to_array(egg.get_verts_pos())[0], tensor_to_array(egg.get_faces())[0]
    np.testing.assert_allclose(rest, midpoint, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(actual_faces, faces)
    masses = np.bincount(faces.reshape(-1), weights=np.repeat(egg.topology.faces_mass / 3, 3), minlength=len(rest))
    np.savez_compressed(args.output / "rest.npz", vertices=rest, faces=faces, masses_kg=masses)
    relief = StaticInertiaRelief.build(rest, masses) if args.inertia_relief else None
    correction = np.zeros_like(force)
    if relief is not None:
        force, correction = relief.apply(force)
        assert np.linalg.norm(correction) <= 1e-8 * np.linalg.norm(force + correction)
        np.testing.assert_allclose(wrench(rest, force), 0, atol=1e-12)
    force_view = torch.as_tensor(force[:, None], dtype=gs.tc_float, device=gs.device)
    state, scratch, info, config = solver.shell_state, solver.shell_scratch, solver.shell_info, solver._static_config
    internal_buffer = array_class.V(dtype=gs.qd_vec3, shape=state.verts_pos.shape)
    observation = FullField.allocate(solver)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "source_field": file_id(args.field),
            "source_wrench_defect": defect.tolist(),
            "rigid_virtual_work_defect": float(shell_work - source_work),
            "mass_kg": float(masses.sum()),
            "load_n": load,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    rows = []
    started = time.perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for iteration in range(args.outer_iterations):
            if args.solve_pad_contact:
                current = tensor_to_array(egg.get_verts_pos())[0]
                elastic_gap = point_sign * np.sum(normal_weights * (current - midpoint)[transfer.selected], axis=(1, 2))
                pad_coordinates, compressive_force, pad_balance, pad_iterations = normal_pad_equilibrium(
                    initial_gap + elastic_gap, pad_rigid, pad_area, pad_stiffness, load, pad_coordinates
                )
                point_forces[:, 2] = point_sign * compressive_force
                force, defect = transfer.apply(points, point_forces)
                if relief is not None:
                    force, correction = relief.apply(force)
                    assert np.linalg.norm(correction) <= 1e-8 * np.linalg.norm(force + correction)
                    np.testing.assert_allclose(wrench(rest, force), 0, atol=1e-12)
                force_view.copy_(torch.as_tensor(force[:, None], dtype=gs.tc_float, device=gs.device))
            egg.set_verts_vel(0.0)
            shell.kernel_shell_compute_forces(args.dt, solver._gravity, state, scratch, info, config)
            add_prescribed_force(args.dt, force_view, scratch)
            shell.kernel_shell_pcg_solve(
                scratch.pcg_flag,
                state,
                scratch,
                info,
                config,
                args.iterations,
                args.tolerance,
                args.velocity_tolerance,
                solver._errno,
            )
            flags = tensor_to_array(solver.get_envs_solver_failure())
            shell.kernel_shell_apply_dv(state, scratch)
            shell.kernel_shell_integrate(args.dt, state, scratch)
            observe_elastic_force(internal_buffer, state, info)
            internal = qd_to_numpy(internal_buffer, transpose=True)[0]
            if args.solve_pad_contact:
                current = tensor_to_array(egg.get_verts_pos())[0]
                elastic_gap = point_sign * np.sum(normal_weights * (current - midpoint)[transfer.selected], axis=(1, 2))
                pad_coordinates, compressive_force, pad_balance, pad_iterations = normal_pad_equilibrium(
                    initial_gap + elastic_gap, pad_rigid, pad_area, pad_stiffness, load, pad_coordinates
                )
                point_forces[:, 2] = point_sign * compressive_force
                force, defect = transfer.apply(points, point_forces)
                if relief is not None:
                    force, correction = relief.apply(force)
                    assert np.linalg.norm(correction) <= 1e-8 * np.linalg.norm(force + correction)
                    np.testing.assert_allclose(wrench(rest, force), 0, atol=1e-12)
                pad_gap = initial_gap + elastic_gap + pad_rigid @ pad_coordinates
                pad_receipt = {
                    "force_moment_relative_error": pad_balance,
                    "rigid_iterations": pad_iterations,
                    "plate_travel_m": float(-pad_coordinates[3]),
                    "normal_forces_n": [float(compressive_force[point_top == upper].sum()) for upper in (False, True)],
                    "compressed_projected_areas_m2": [
                        float(pad_area[(point_top == upper) & (pad_gap < 0)].sum()) for upper in (False, True)
                    ],
                }
            residual = internal + force
            relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(force))
            rows.append(
                {
                    "outer_iteration": iteration + 1,
                    "static_force_relative_residual": relative_residual,
                    "solver_failure_flags": flags.tolist(),
                    "pcg_iterations": tensor_to_array(solver.get_envs_pcg_iterations()).tolist(),
                    "pad_contact": pad_receipt,
                    "inertia_relief_relative_correction": float(
                        np.linalg.norm(correction) / np.linalg.norm(force + correction)
                    ),
                }
            )
            journal.write(json.dumps(rows[-1], allow_nan=False) + "\n")
            if flags.any() or relative_residual <= args.force_tolerance:
                break
    observation.update(solver)
    current = tensor_to_array(egg.get_verts_pos())[0]
    tensors = tensor_to_array(observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
    field = args.output / "frames/level0.npz"
    np.savez_compressed(
        field,
        vertices_world_m=current,
        cauchy_world_pa=tensors,
        vm_pa=tensor_to_array(observation.von_mises)[:, 0].reshape(-1, 3),
        membrane_n_per_m=tensor_to_array(observation.membrane)[:, 0],
        bending_n=tensor_to_array(observation.bending)[:, 0],
        applied_force_n=force,
        mapped_contact_force_n=force + correction,
        inertia_relief_correction_n=correction,
        internal_forces_world_n=internal,
        equation_residual_n=residual,
    )
    admitted = not flags.any() and relative_residual <= args.force_tolerance and np.isfinite(tensors).all()
    if args.solve_pad_contact:
        bottom_gap, top_gap = shell_pad_boundary_gaps(outer, midpoint, faces, current, pad_coordinates)
        outside_gap = float(min(bottom_gap[omitted_faces].min(), top_gap[omitted_faces].min()))
        scale = load / (pad_stiffness * np.pi * 0.002**2)
        pad_admitted = bool(admitted and pad_balance <= 1e-10 and outside_gap >= -1e-8 * scale)
        pad_field = args.output / "pad-contact.npz"
        np.savez_compressed(
            pad_field,
            points_m=points,
            point_forces_n=point_forces,
            projected_area_m2=pad_area,
            body_gap_m=pad_gap,
            complete_face_bottom_gap_m=bottom_gap,
            complete_face_top_gap_m=top_gap,
            rigid_coordinates_m=pad_coordinates,
            physical_vertices_linear_m=current
            + pad_coordinates[0] * np.array([0, 0, 1])
            + np.cross(np.array([pad_coordinates[1] / 0.03, pad_coordinates[2] / 0.03, 0]), midpoint),
        )
        pad_receipt.update(
            {
                "field": file_id(pad_field),
                "pad_stiffness_n_per_m3": pad_stiffness,
                "minimum_omitted_face_gap_m": outside_gap,
                "static_pad_equations_admitted": pad_admitted,
                "scope": "Self-consistent static pad pressure and plate travel solved from the current Shell displacement. Input solid pressures are not used in the solve. Pad gaps use the transpose of the fixed reference-geometry force/couple map. Complete omitted-face gaps use the same affine map. This is a declared linear thin-surface kinematic reduction and velocity-cleared static relaxation, not a continuous dynamic contact history. Spatial qualification remains outstanding.",
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "source_field": file_id(args.field),
            "field": file_id(field),
            "load_n": load,
            "iterations": len(rows),
            "last_iteration": rows[-1],
            "static_equations_admitted": bool(admitted),
            "seconds_including_observation": time.perf_counter() - started,
            "deformed_moment_drift": transfer.deformed_moment_drift(points, point_forces, current),
            "physical_accuracy_qualified": False,
            "inertia_relief_enabled": args.inertia_relief,
            "inertia_relief_relative_correction": float(
                np.linalg.norm(correction) / np.linalg.norm(force + correction)
            ),
            "inertia_relief_scope": "Optional six-mode mass-weighted correction, identical in form to the free solid reference. Actual mapped point tractions remain unchanged and saved separately; correction is explicit body loading and must stay below1e-8 of mapped load norm. Original zero-wrench1e-12 and static residual gates remain unchanged.",
            "pad_contact": pad_receipt,
            "scope": "Explicit static relaxation with velocity cleared before each pseudo-step, using unchanged pinned Shell elastic/PCG/integration kernels and an external-force adapter. This is not original scene dynamics or a continuous plate-contact history. Actual end-position elastic forces are independently observed. Source force/couple transfer preserves rest-frame wrench and rigid work; thin-shell kinematics and deformed moment drift remain measured limitations. The optional pad branch recomputes pressure and free rigid coordinates from current displacement at each iteration and retains separate contact admission.",
        },
    )


if __name__ == "__main__":
    main()
