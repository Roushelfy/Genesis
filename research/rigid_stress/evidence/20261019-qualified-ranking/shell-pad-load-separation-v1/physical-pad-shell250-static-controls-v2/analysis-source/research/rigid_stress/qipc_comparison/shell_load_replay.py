"""Replay resolved QIPC contact/inertia as force-and-couple loads in PR22 static relaxation.

This uses the existing prescribed-load adapter and original Shell kernels.
It is a per-frame elastic-response ablation, not a contact/dynamic rollout.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers import shell_solver as shell
from genesis.utils.misc import qd_to_torch, tensor_to_array

from ..shell_comparison.observation import FullField
from ..shell_comparison.prescribed import add_prescribed_force
from ..shell_comparison.wall_input import ShellWallInput
from .artifacts import file_id, runtime_identity, write_json
from .common_metric import CommonColumns, corotation, field_error, resultants, tangent
from .compare_fields import area_error, face_assignment
from .reference_uncertainty import frame_index
from .static_inertia_relief import StaticInertiaRelief


def wrench(points, force):
    return np.r_[force.sum(axis=0), np.cross(points, force).sum(axis=0)]


@dataclass
class ShellLoadTransfer:
    vertices: np.ndarray
    selected: np.ndarray
    barycentric: np.ndarray
    relative_corners: np.ndarray
    inverse_moment: np.ndarray

    @classmethod
    def allocate(cls, vertices, faces, ray_midpoints, layers):
        triangle = vertices[faces]
        edges = triangle[:, 1:] - triangle[:, :1]
        inverse = np.linalg.inv(np.einsum("fai,fbi->fab", edges, edges))
        selected, weights = [], []
        for point in ray_midpoints:
            free = np.einsum("fij,fj->fi", inverse, np.einsum("fai,fi->fa", edges, point - triangle[:, 0]))
            bary = np.column_stack((1 - free.sum(axis=1), free))
            distances = np.linalg.norm(np.einsum("fi,fij->fj", bary, triangle) - point, axis=1)
            distances[bary.min(axis=1) < -1e-10] = np.inf
            index = distances.argmin()
            assert distances[index] < 1e-12, "Material midpoint rays do not share the Shell domain"
            selected.append(faces[index])
            # Keep three weights, not a view retaining every candidate face.
            weights.append(bary[index].copy())
        selected = np.tile(np.array(selected), (layers, 1))
        weights = np.tile(np.array(weights), (layers, 1))
        relative = vertices[selected] - vertices[selected].mean(axis=1, keepdims=True)
        gram = np.eye(3) * (relative**2).sum(axis=(1, 2))[:, None, None]
        gram -= np.einsum("pki,pkj->pij", relative, relative)
        return cls(vertices, selected, weights, relative, np.linalg.inv(gram))

    def local_loads(self, actual_points, force):
        assert actual_points.shape == force.shape == (len(self.selected), 3)
        assert np.isfinite(actual_points).all() and np.isfinite(force).all()
        midpoint = np.einsum("pk,pkj->pj", self.barycentric, self.vertices[self.selected])
        missing_couple = np.cross(actual_points - midpoint, force)
        multiplier = np.einsum("pij,pj->pi", self.inverse_moment, missing_couple)
        correction = np.cross(multiplier[:, None, :], self.relative_corners)
        return self.barycentric[:, :, None] * force[:, None, :] + correction

    def apply(self, actual_points, force):
        local = self.local_loads(actual_points, force)
        loads = np.zeros_like(self.vertices)
        for corner in range(3):
            np.add.at(loads, self.selected[:, corner], local[:, corner])
        defect = wrench(self.vertices, loads) - wrench(actual_points, force)
        np.testing.assert_allclose(defect[:3], 0, rtol=0, atol=2e-9)
        np.testing.assert_allclose(defect[3:], 0, rtol=0, atol=2e-11)
        return loads, defect

    def deformed_moment_drift(self, actual_points, force, current):
        local = self.local_loads(actual_points, force)
        difference = np.cross(current[self.selected], local).sum(axis=1) - np.cross(actual_points, force)
        return {
            "net_moment_drift_nm": difference.sum(axis=0).tolist(),
            "sum_local_moment_drift_norms_nm": np.linalg.norm(difference, axis=1).sum().item(),
        }


def transfer_controls(transfer, points, ray_count):
    rng = np.random.default_rng(623018)
    current = points + rng.normal(scale=1e-5, size=points.shape)
    force = rng.normal(size=points.shape)
    mapped, defect = transfer.apply(current, force)
    translation, angular = rng.normal(size=(2, 3))
    source_work = (force * (translation + np.cross(angular, current))).sum()
    mapped_work = (mapped * (translation + np.cross(angular, transfer.vertices))).sum()
    np.testing.assert_allclose(source_work, mapped_work, rtol=1e-11, atol=1e-11)
    couple_force = np.zeros_like(force)
    direction = points[0] - points[-ray_count]
    tangent_force = np.cross(direction / np.linalg.norm(direction), np.array([1.0, 0.3, 0.2]))
    assert np.linalg.norm(tangent_force) > 0.1
    couple_force[0], couple_force[-ray_count] = tangent_force, -tangent_force
    mapped_couple, couple_defect = transfer.apply(points, couple_force)
    assert np.linalg.norm(wrench(points, couple_force)[3:]) > 1e-6
    np.testing.assert_allclose(mapped_couple.sum(axis=0), 0, rtol=0, atol=1e-12)
    return {
        "random_force_and_moment_defect": defect.tolist(),
        "rigid_virtual_work_difference": (mapped_work - source_work).item(),
        "pure_through_wall_couple_defect": couple_defect.tolist(),
        "scope": "Actual-source-point force/moment and arbitrary rigid virtual work transferred to the fixed undeformed Shell configuration. These controls do not establish equivalence for nonrigid through-wall virtual displacements or unchanged moment after Shell deformation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--wall-input", type=Path)
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+")
    parser.add_argument("--outer-iterations", type=int, default=6)
    parser.add_argument("--pcg-iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--static-force-tolerance", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames_output = args.output / "frames"
    frames_output.mkdir()
    inputs = json.loads((args.reference / "inputs.json").read_text())
    with np.load(args.reference / "rest.npz") as saved:
        authored, source_masses = saved["authored_vertices_m"], saved["masses_kg"]
    with np.load(inputs["mesh"]["path"]) as saved:
        outer, inner = saved["outer_vertices"], saved["inner_vertices"]
        np.testing.assert_allclose(saved["vertices"], authored, rtol=0, atol=1e-15)
    ray_count = len(outer)
    layers = len(authored) // ray_count
    assert layers * ray_count == len(authored) and layers >= 2
    expected_layers = np.concatenate([outer * (1 - t) + inner * t for t in np.linspace(0, 1, layers)])
    np.testing.assert_allclose(authored, expected_layers, rtol=0, atol=1e-15)
    with np.load(args.columns) as saved:
        columns = CommonColumns(**{key: saved[key] for key in saved.files})
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        shell_options=gs.options.ShellOptions(
            n_pcg_iterations=args.pcg_iterations, pcg_threshold=args.tolerance, coarse_update_interval=1
        ),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Mesh(file=args.inputs / f"egg-mid-level{args.level}.obj", decimate=False),
        material=gs.materials.Shell(rho=2000.0, E=1e10, nu=0.3, thickness=0.0005),
    )
    wall = ShellWallInput.bind(args.wall_input, entity, np.zeros(3)) if args.wall_input is not None else None
    scene.build(n_envs=1)
    solver = scene.sim.shell_solver
    if wall is not None:
        wall.apply_thickness(solver)
    assert not solver._static_config.has_plasticity and not solver._static_config.has_fracture
    rest, faces = entity.init_verts, tensor_to_array(entity.get_faces())[0]
    thickness = tensor_to_array(qd_to_torch(solver.shell_state.faces_thickness, copy=True))[:, 0]
    shell_mass = np.bincount(
        faces.reshape(-1), weights=np.repeat(entity.topology.faces_mass / 3, 3), minlength=len(rest)
    )
    transfer = ShellLoadTransfer.allocate(rest, faces, (outer + inner) / 2, layers)
    controls = transfer_controls(transfer, authored, ray_count)
    lookup = face_assignment(columns, rest, faces)
    relief = StaticInertiaRelief.build(rest, shell_mass)
    force_view = torch.empty((len(rest), 1, 3), dtype=gs.tc_float, device=gs.device)
    observation = FullField.allocate(solver)
    files = frame_index(args.reference)
    ticks = sorted(files) if args.times is None else sorted({round(t / 0.01) for t in args.times})
    assert set(ticks) <= files.keys()
    write_json(
        args.output / "inputs.json",
        {
            "reference": file_id(args.reference / "result.json"),
            "source_mesh": file_id(inputs["mesh"]["path"]),
            "shell_mesh": file_id(args.inputs / f"egg-mid-level{args.level}.obj"),
            "columns": file_id(args.columns),
            "transfer_controls": controls,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    np.savez_compressed(
        args.output / "rest.npz", vertices_m=rest, faces=faces, masses_kg=shell_mass, face_thickness_m=thickness
    )
    records = []
    state, scratch, info, config = solver.shell_state, solver.shell_scratch, solver.shell_info, solver._static_config
    rest_tensor = torch.as_tensor(rest, dtype=gs.tc_float, device=gs.device)
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for tick in ticks:
            with np.load(files[tick]) as saved:
                position = saved["positions_m"]
                rotation = corotation(authored, position, source_masses)
                actual = (position - source_masses @ position / source_masses.sum()) @ rotation
                actual += source_masses @ authored / source_masses.sum()
                source_contact = saved["contact_forces_n"] @ rotation
                source_inertia = saved["inertial_reaction_n"] @ rotation
                contact, contact_defect = transfer.apply(actual, source_contact)
                inertia, inertia_defect = transfer.apply(actual, source_inertia)
                reference = tangent(rotation.T @ saved["cauchy_pa"][columns.reference_elements] @ rotation, columns)
            load, balance = relief.apply(contact - inertia)
            force_view.copy_(torch.as_tensor(load[:, None], dtype=gs.tc_float, device=gs.device))
            entity.set_verts_pos(rest_tensor)
            entity.set_verts_vel(0.0)
            scratch.verts_dv.fill(0.0)
            residual = np.inf
            absolute_residual = np.inf
            iterations = 0
            cap_steps = 0
            for iteration in range(args.outer_iterations):
                entity.set_verts_vel(0.0)
                shell.kernel_shell_compute_forces(0.01, solver._gravity, state, scratch, info)
                add_prescribed_force(0.01, force_view, scratch)
                if config.has_coarse_space:
                    shell.kernel_shell_coarse_assemble(state, scratch, info)
                    shell.kernel_shell_coarse_factorize(state, scratch, info, config)
                shell.kernel_shell_system_product(scratch.verts_dv, scratch.verts_Ap, state, scratch, info)
                shell.kernel_shell_pcg_init(args.tolerance, state, scratch, info, config)
                for _ in range(args.pcg_iterations):
                    shell.kernel_shell_system_product(scratch.verts_p, scratch.verts_Ap, state, scratch, info)
                    shell.kernel_shell_pcg_update(state, scratch, info, config)
                cap_steps += int(tensor_to_array(qd_to_torch(scratch.envs_is_solving, copy=True))[0])
                shell.kernel_shell_apply_dv(state, scratch)
                shell.kernel_shell_integrate(0.01, state, scratch)
                entity.set_verts_vel(0.0)
                shell.kernel_shell_compute_forces(0.01, solver._gravity, state, scratch, info)
                add_prescribed_force(0.01, force_view, scratch)
                absolute_residual = (torch.linalg.vector_norm(qd_to_torch(scratch.verts_rhs)) / 0.01).item()
                residual = absolute_residual / max(np.linalg.norm(load).item(), 1e-12)
                iterations = iteration + 1
                if not np.isfinite(residual) or residual <= args.static_force_tolerance:
                    break
            observation.update(solver)
            tensor = tensor_to_array(observation.tensors)[:, 0].reshape(-1, 3, 3, 3)
            world_tensor = tensor.copy()
            current = tensor_to_array(entity.get_verts_pos())[0]
            finite = bool(np.isfinite(tensor).all() and np.isfinite(current).all() and np.isfinite(residual))
            errors = None
            shell_rotation = np.full((3, 3), np.nan)
            if finite:
                shell_rotation = corotation(rest, current, shell_mass)
                tensor = shell_rotation.T @ tensor @ shell_rotation
                sampled = tensor[lookup, 1] + (columns.z / thickness[lookup])[:, None, None] * (
                    tensor[lookup, 2] - tensor[lookup, 0]
                )
                sampled = tangent(sampled, columns)
                membrane, bending = resultants(sampled, columns)
                expected_membrane, expected_bending = resultants(reference, columns)
                errors = {
                    "common_tangent": field_error(sampled, reference, columns),
                    "membrane": area_error(membrane, expected_membrane, columns.area, 5e-4),
                    "bending": area_error(bending, expected_bending, columns.area, 2.5e-7),
                }
            row = {
                "time_s": tick * 0.01,
                "source": file_id(files[tick]),
                "all_fields_finite": finite,
                "static_relative_force_residual": residual if np.isfinite(residual) else None,
                "static_force_residual_l2_n": absolute_residual if np.isfinite(absolute_residual) else None,
                "balanced_load_l2_n": np.linalg.norm(load).item(),
                "outer_iterations": iterations,
                "pcg_cap_steps": cap_steps,
                "raw_vm_peak_pa": tensor_to_array(observation.peaks)[0].item() if finite else None,
                "static_numerically_admitted": finite and cap_steps == 0 and residual <= args.static_force_tolerance,
                "contact_wrench_transfer_defect": contact_defect.tolist(),
                "inertial_wrench_transfer_defect": inertia_defect.tolist(),
                "deformed_contact_moment_drift": transfer.deformed_moment_drift(actual, source_contact, current)
                if finite
                else None,
                "deformed_inertial_moment_drift": transfer.deformed_moment_drift(actual, source_inertia, current)
                if finite
                else None,
                "balance_correction_l2_n": np.linalg.norm(balance).item(),
                "balance_correction_wrench": wrench(rest, balance.reshape(-1, 3)).tolist(),
                "errors": errors,
                "reference_qualified": False,
            }
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            records.append(row)
            np.savez_compressed(
                frames_output / files[tick].name,
                time_s=tick * 0.01,
                shell_tensor_world_pa=world_tensor,
                shell_material_corotation=shell_rotation,
                shell_vertices_m=current,
                mapped_contact_n=contact,
                mapped_inertial_reaction_n=inertia,
                balanced_load_n=load,
                balance_correction_n=balance.reshape(-1, 3),
            )
    write_json(
        args.output / "result.json",
        {
            "kind": "resolved-reference-load-shell-static-replay",
            "identity": runtime_identity(),
            "reference": file_id(args.reference / "result.json"),
            "source_reference_completed": json.loads((args.reference / "result.json").read_text())[
                "completed_requested_history"
            ],
            "wall_input": wall.description() if wall is not None else None,
            "transfer_controls": controls,
            "requested_replays": len(ticks),
            "completed_requested_replay_set": len(records) == len(ticks),
            "admitted_static_replays": sum(row["static_numerically_admitted"] for row in records),
            "rejected_static_replays": sum(not row["static_numerically_admitted"] for row in records),
            "reference_qualified": False,
            "scope": "Independent static elastic-response replay at each recorded source time. Contact and measured inertial loads are separately mapped to supporting midpoint triangles; local nodal couples preserve the actual source-point wrench and rigid virtual work in the fixed undeformed Shell configuration. Deformation changes the applied moment; both its net drift and a sum of uncancelled local drift norms are retained separately. A reported six-mode mass-weighted balance correction makes the free static problem solvable. This does not preserve every nonrigid3D virtual displacement, does not fit a footprint, and is not a contact-history or capped-control rollout. Original PR22 force/stiffness/PCG/integration kernels and the existing dt*external-force adapter are used; no IPC/contact substitution. Rejected and nonfinite states remain in the journal and raw world fields; material corotation is stored instead of a redundant second tensor copy.",
        },
    )


if __name__ == "__main__":
    main()
