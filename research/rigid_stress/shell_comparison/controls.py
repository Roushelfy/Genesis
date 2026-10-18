"""Run read-only shell stress conversion and original-PR contact controls."""

import argparse
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from . import diagnostics
from .common import grid_sheet, identity, vars_json, write_json
from .energy import shell_energy
from .observation import FullField


def field_controls(args):
    path = args.output / "grid.obj"
    grid_sheet(path, args.resolution, args.resolution, 0.1, 0.1)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=path,
            decimate=False,
        ),
        material=gs.materials.Shell(
            E=1e7,
            nu=0.0,
            thickness=0.001,
        ),
    )
    scene.build(n_envs=2)
    observation = FullField.allocate(scene.sim.shell_solver)
    reference = entity.init_verts.copy()
    receipts = []
    fields = {}
    for strain in (0.0, 1e-4, -1e-4):
        positions = reference.copy()
        positions[:, 0] *= 1.0 + strain
        entity.set_verts_pos(positions)
        observation.update(scene.sim.shell_solver)
        stress = tensor_to_array(observation.tensors).copy()
        expected = 1e7 * 0.5 * ((1.0 + strain) ** 2 - 1.0) * (1.0 + strain)
        maximum_error = np.abs(stress[:, :, 0, 0] - expected).max()
        receipts.append(
            {
                "case": "affine-stretch",
                "strain": strain,
                "expected_cauchy_xx_pa": expected,
                "max_absolute_xx_error_pa": float(maximum_error),
                "min_xx_pa": float(stress[:, :, 0, 0].min()),
                "max_xx_pa": float(stress[:, :, 0, 0].max()),
                "other_components_max_pa": float(np.abs(stress[:, :, 1:]).max()),
            }
        )
        fields[f"stretch_{strain}"] = stress
    # Small cylindrical bending keeps the middle surface inextensible in the continuum.
    # Boundary faces lack a neighboring hinge, so report their estimator separately.
    curvature = 2.0
    positions = reference.copy()
    positions[:, 0] = np.sin(curvature * reference[:, 0]) / curvature
    positions[:, 2] = (1.0 - np.cos(curvature * reference[:, 0])) / curvature
    entity.set_verts_pos(positions)
    observation.update(scene.sim.shell_solver)
    kappa = tensor_to_array(observation.curvature).copy()
    stress = tensor_to_array(observation.tensors).copy()
    membrane = tensor_to_array(observation.membrane).copy()
    interior = (entity.topology.faces_hinge >= 0).all(axis=1)
    eigenvalues = np.linalg.eigvalsh(kappa[interior, 0])
    expected_fiber = 1e7 * 0.001 * curvature / 2.0
    vm = tensor_to_array(observation.von_mises).reshape(-1, 3, 2)
    receipts.append(
        {
            "case": "cylindrical-bending",
            "curvature_per_m": curvature,
            "interior_faces": int(interior.sum()),
            "interior_absolute_principal_curvature_mean_per_m": float(np.abs(eigenvalues).max(axis=1).mean()),
            "expected_outer_fiber_vm_pa": expected_fiber,
            "interior_outer_fiber_vm_mean_pa": float(vm[interior][:, (0, 2)].mean()),
            "interior_outer_fiber_vm_min_pa": float(vm[interior][:, (0, 2)].min()),
            "interior_outer_fiber_vm_max_pa": float(vm[interior][:, (0, 2)].max()),
            "boundary_outer_fiber_vm_mean_pa": float(vm[~interior][:, (0, 2)].mean()),
            "membrane_max_n_per_m": float(np.abs(membrane).max()),
        }
    )
    fields.update(
        bend_tensors=stress,
        bend_curvature=kappa,
        bend_membrane=membrane,
        rest_vertices=reference,
        bent_vertices=positions,
        interior_faces=interior,
        faces=tensor_to_array(entity.get_faces())[0],
    )
    np.savez_compressed(args.output / "fields.npz", **fields)
    write_json(
        args.output / "result.json",
        {"kind": "field-conversion", "args": vars_json(args), "identity": identity(), "cases": receipts},
    )


def contact_controls(args):
    if args.ledger:
        assert args.instrumented and args.motion == "drop" and args.substeps == 1
    path = args.output / "sheet.obj"
    grid_sheet(path, args.resolution, args.resolution)
    radius, thickness, dt = 0.01, 0.001, args.dt
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=args.substeps,
            gravity=(0.0, 0.0, -9.81 if args.motion == "drop" else 0.0),
        ),
        shell_options=gs.options.ShellOptions(
            n_pcg_iterations=args.iterations,
            pcg_threshold=args.tolerance,
            n_coarse_patches=args.patches,
            coarse_update_interval=args.coarse_interval,
        ),
        show_viewer=False,
    )
    sheet = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=path,
            decimate=False,
        ),
        material=gs.materials.Shell(
            rho=500.0,
            E=1e6,
            nu=0.3,
            thickness=thickness,
            damping=0.01,
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=radius,
            pos=(0.0, 0.0, radius + 0.5 * thickness + 1e-4),
        ),
        material=gs.materials.Rigid(
            rho=1e4,
            coup_friction=args.friction,
        ),
    )
    # Same physical locations across resolutions. Includes triangle interior,
    # central free vertex, an edge and offsets spanning the sphere footprint.
    xy = np.array(
        [
            [0.2 / 3, 0.1 / 3],
            [0.0, 0.0],
            [0.05, 0.0],
            [0.008, 0.0],
            [0.014, 0.0],
            [0.02, 0.0],
            [0.04, 0.025],
            [-0.04, -0.025],
        ]
    )
    scene.build(n_envs=len(xy))
    sheet.fix_verts(np.flatnonzero(np.abs(sheet.init_verts[:, :2]).max(axis=1) > 0.1 - 1e-6))
    sphere.set_pos(np.column_stack((xy, np.full(len(xy), radius + 0.5 * thickness + 1e-4))))
    observation = FullField.allocate(scene.sim.shell_solver)
    if args.instrumented:
        diagnostics.configure(scene.sim.shell_solver)
    heights, velocities, peaks, positions = [], [], [], []
    solver_iterations, solver_caps, solver_residuals, impulses, contact_points = [], [], [], [], []
    sphere_positions, sphere_velocities = [], []
    angular_velocities, mechanical_energy = [], []
    initial_energy = np.empty(0)
    inertia = tensor_to_array(sphere.get_links_inertia()).copy()
    energy = torch.empty((len(xy), 3), dtype=gs.tc_float, device=gs.device)
    if args.ledger:
        solver = scene.sim.shell_solver
        shell_energy(9.81, energy, solver.shell_state, solver.shell_scratch, solver.shell_info)
        initial_energy = tensor_to_array(energy).sum(axis=1) + tensor_to_array(sphere.get_mass()) * 9.81 * (
            radius + 0.5 * thickness + 1e-4
        )
    for step in range(args.steps):
        if args.motion == "press-slide":
            time = step * dt
            z = radius + 0.5 * thickness + 0.001 - 0.006 * np.clip(time / 0.05, 0, 1)
            z += 0.02 * np.clip((time - 0.15) / 0.05, 0, 1)
            x = 0.06 * np.clip((time - 0.05) / 0.1, 0, 1)
            vx = 0.6 if 0.05 <= time < 0.15 else 0.0
            vz = -0.12 if time < 0.05 else (0.4 if time >= 0.15 else 0.0)
            sphere.set_pos(np.column_stack((xy[:, 0] + x, xy[:, 1], np.full(len(xy), z))))
            sphere.set_dofs_velocity([vx, 0.0, vz, 0.0, 0.0, 0.0])
        if args.instrumented:
            diagnostics.ACTIVE.clear()
        scene.step()
        observation.update(scene.sim.shell_solver)
        heights.append(tensor_to_array(sphere.get_pos())[:, 2].copy())
        velocities.append(tensor_to_array(sphere.get_vel())[:, 2].copy())
        peaks.append(tensor_to_array(observation.peaks).copy())
        sphere_positions.append(tensor_to_array(sphere.get_pos()).copy())
        sphere_velocities.append(tensor_to_array(sphere.get_vel()).copy())
        if args.ledger:
            angular_velocities.append(tensor_to_array(sphere.get_ang()).copy())
            solver = scene.sim.shell_solver
            shell_energy(9.81, energy, solver.shell_state, solver.shell_scratch, solver.shell_info)
            mechanical_energy.append(tensor_to_array(energy).copy())
        if args.instrumented:
            solver_iterations.append(tensor_to_array(diagnostics.ACTIVE.counts).copy())
            solver_caps.append(tensor_to_array(diagnostics.ACTIVE.caps).copy())
            solver_residuals.append(tensor_to_array(diagnostics.ACTIVE.residuals).copy())
            impulses.append(tensor_to_array(diagnostics.ACTIVE.impulses).copy())
            contact_points.append(tensor_to_array(diagnostics.ACTIVE.points).copy())
        if step % 10 == 0 or step == args.steps - 1:
            positions.append(tensor_to_array(sheet.get_verts_pos()).copy())
    mass = tensor_to_array(sphere.get_mass())
    height = np.array(heights)
    velocity = np.array(velocities)
    gravity = 9.81 if args.motion == "drop" else 0.0
    reaction = (np.diff(velocity, axis=0) / dt + gravity) * mass
    np.savez_compressed(
        args.output / "trajectory.npz",
        heights=height,
        velocities=velocity,
        peak_pa=np.array(peaks),
        sheet_positions=np.array(positions),
        xy=xy,
        reaction_z_n=reaction,
        faces=tensor_to_array(sheet.get_faces())[0],
        initial_sheet_vertices=sheet.init_verts,
        final_stress=tensor_to_array(observation.tensors),
        final_vm=tensor_to_array(observation.von_mises),
        sphere_positions=np.array(sphere_positions),
        sphere_velocities=np.array(sphere_velocities),
        iterations=np.array(solver_iterations),
        caps=np.array(solver_caps),
        residuals=np.array(solver_residuals),
        impulses=np.array(impulses),
        contact_points=np.array(contact_points),
        sphere_angular_velocities=np.array(angular_velocities),
        sphere_inertia_kg_m2=inertia,
        shell_energy_j=np.array(mechanical_energy),
        initial_total_energy_j=initial_energy,
    )
    write_json(
        args.output / "result.json",
        {
            "kind": "sphere-contact",
            "args": vars_json(args),
            "identity": identity(),
            "sphere_mass_kg": float(mass.reshape(-1)[0]),
            "xy_m": xy.tolist(),
            "nominal_rest_contact_z_m": radius + thickness / 2,
            "final_height_m": height[-1].tolist(),
            "minimum_height_m": height.min(axis=0).tolist(),
            "final_velocity_z_m_per_s": velocity[-1].tolist(),
            "late_mean_reaction_z_n": reaction[-min(50, len(reaction)) :].mean(axis=0).tolist(),
            "final_center_below_initial_plane_minus_radius": (height[-1] < -radius).tolist(),
            "height_marker_limit": "A flexible sheet can sag; this height marker alone does not prove geometric passage through it.",
            "peak_vm_pa": np.array(peaks).max(axis=0).tolist(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("field", "contact"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--patches", type=int, default=6)
    parser.add_argument("--coarse-interval", type=int, default=10)
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--motion", choices=("drop", "press-slide"), default="drop")
    parser.add_argument("--friction", type=float, default=0.6)
    parser.add_argument("--ledger", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision=args.precision, logging_level="warning")
    if args.kind == "field":
        field_controls(args)
    else:
        contact_controls(args)


if __name__ == "__main__":
    main()
