"""Audit every original-history global stress bound and prepare objective resolved-load replays."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanBulk
from .qualify_articulated_precision import tensor_from_f
from .reference_motion_audit import motion_moments
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def wide_volume_bounds(corner_f, mu):
    """Bernstein determinant/VM bounds on the saved wide-coordinate P2 field."""
    f = np.asarray(corner_f, dtype=np.longdouble)
    grouped = {}
    for first, second, third in product(range(4), repeat=3):
        value = np.sum(f[:, first, :, 0] * np.cross(f[:, second, :, 1], f[:, third, :, 2]), axis=-1)
        grouped.setdefault(tuple(sorted((first, second, third))), []).append(value)
    lower_j = np.stack([np.mean(values, axis=0) for values in grouped.values()]).min(axis=0)
    assert np.all(lower_j > 0)
    numerator = []
    for first in range(4):
        for second in range(first, 4):
            coefficient = (
                f[:, first] @ f[:, second].swapaxes(-1, -2) + f[:, second] @ f[:, first].swapaxes(-1, -2)
            ) / 2
            coefficient -= np.trace(coefficient, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
            numerator.append(mu * np.sqrt(1.5 * np.sum(coefficient * coefficient, axis=(-2, -1))))
    upper = np.max(numerator, axis=0) / lower_j
    upper += 128 * np.finfo(np.longdouble).eps * mu * np.max(np.sum(f * f, axis=(-2, -1)), axis=1) / lower_j
    return float(lower_j.min()), float(upper.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--force-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=8)
    args = parser.parse_args()
    assert args.sample_every > 0
    for run in [args.run, args.force_audit]:
        closed_source(run)
    assert int((args.force_audit / "returncode.txt").read_text()) == 0
    audit = json.loads((args.force_audit / "results/summary.json").read_text())
    assert Path(audit["run"]).resolve() == args.run.resolve()
    assert audit.get("recovery_run") is None
    config = json.loads((args.run / "results/configuration.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.run / "results/reference_geometry.npz") as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        permutation = saved["body_component_permutation"].copy()
        rest = saved["rest_positions"].copy()
    np.savez_compressed(
        args.output / "body_geometry.npz",
        vertices=body.mesh.p.T,
        tetrahedra=body.mesh.t.T,
        component_permutation=permutation,
    )
    n = body.basis.N
    mass = body.mass[permutation][:, permutation]
    bulk = NeoHookeanBulk(body)
    rows, input_hashes, snapshots = [], {}, []
    for step in range(1, audit["steps_checked"] + 1):
        path = args.run / f"results/step{step:05d}.npz"
        with np.load(path) as saved:
            full, velocity, acceleration, contact, applied = [
                saved[key].copy()
                for key in ["displacement", "body_velocity", "body_acceleration", "contact_force_N", "applied_force_N"]
            ]
        with np.load(args.force_audit / f"results/step{step:05d}.npz") as checked:
            np.testing.assert_array_equal(checked["contact_force_N"], contact)
            assert np.linalg.norm(checked["body_residual_N"]) <= 1e-9
            assert np.linalg.norm(checked["actual_robot_effort_residual"]) <= 1e-9
        wide_vector = np.empty(n, dtype=np.longdouble)
        wide_vector[permutation] = full[:n]
        vector = np.asarray(wide_vector, dtype=float)
        rotation, _, local = finite_frame(body, vector)
        corner_f = bulk.corner_deformation_gradients(wide_vector)
        precise_corner = tensor_from_f(corner_f, body.lame)
        dev = precise_corner - np.trace(precise_corner, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        precise_vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1)))
        _, rounded_vm, rounded_upper = bulk.stress_bounds(vector)
        lower = float(precise_vm.max())
        corner_rounding = float(abs(np.asarray(precise_vm, dtype=np.longdouble) - rounded_vm).max())
        j_lower, global_upper = wide_volume_bounds(corner_f, bulk.mu)
        assert lower <= global_upper
        cell, corner = np.unravel_index(np.argmax(precise_vm), precise_vm.shape)
        location = body.corner_positions[cell, corner]
        nonlinear_world, energy = quadrature_fields(bulk, vector)
        nonlinear_material = rotation.T @ nonlinear_world @ rotation
        positions = rest[: n // 3] + full[:n].reshape(-1, 3)
        motion = motion_moments(
            body, permutation, np.asarray(positions, dtype=float), np.asarray(velocity, dtype=float)
        )
        inertial = mass @ acceleration
        material_contact, material_rhs = np.zeros(n), np.zeros(n)
        material_contact[body.component_dofs] = contact[:n].reshape(-1, 3) @ rotation
        material_rhs[body.component_dofs] = (contact[:n] + applied[:n] - inertial).reshape(-1, 3) @ rotation
        full_replay = body.solve(material_rhs)
        omega = rotation.T @ motion["angular_velocity_world_rad_s"]
        rigid_replay = body.solve(material_contact, omega)

        def at_quadrature(displacement):
            h = np.einsum("aie,ajeq->ijeq", displacement[bulk.dofs], bulk.gradients, optimize=True).transpose(
                2, 3, 0, 1
            )
            return bulk.mu * (h + h.swapaxes(-1, -2)) + bulk.lam * np.trace(h, axis1=-2, axis2=-1)[
                ..., None, None
            ] * np.eye(3)

        linear = at_quadrature(full_replay.displacement)
        rigid_linear = at_quadrature(rigid_replay.displacement)
        denominator = float(np.linalg.norm(nonlinear_material))
        row = {
            "step": step,
            "time_s": step * config["dt"],
            "raw_global_peak_lower_Pa": lower,
            "raw_global_peak_upper_Pa": global_upper,
            "raw_global_peak_reference_position_m": location.tolist(),
            "rounded_coordinate_peak_difference_Pa": abs(lower - float(rounded_vm.max())),
            "maximum_corner_VM_rounding_difference_Pa": corner_rounding,
            "rounded_field_global_peak_upper_Pa": float(rounded_upper.max()),
            "whole_volume_determinant_lower_bound": j_lower,
            "nonlinear_energy_J": energy,
            "objective_maximum_nodal_displacement_m": float(np.linalg.norm(local[body.component_dofs], axis=1).max()),
            "full_inertia_linear_peak_Pa": full_replay.peak_pa,
            "rigid_relief_linear_peak_Pa": rigid_replay.peak_pa,
            "full_rhs_balance_correction_N": float(np.linalg.norm(full_replay.rhs - material_rhs)),
            "nonlinear_vs_full_inertia_tensor_absolute_L2_Pa": float(np.linalg.norm(linear - nonlinear_material)),
            "nonlinear_tensor_L2_Pa": denominator,
            "nonlinear_vs_full_inertia_tensor_relative_L2": float(np.linalg.norm(linear - nonlinear_material))
            / max(denominator, 1e-30),
            "full_inertia_vs_rigid_relief_tensor_relative_L2": float(np.linalg.norm(rigid_linear - linear))
            / max(float(np.linalg.norm(linear)), 1e-30),
        }
        rows.append(row)
        if step % args.sample_every == 0 or step in [1, 11, 866, 869, audit["steps_checked"]]:
            np.savez_compressed(
                args.output / f"step{step:05d}.npz",
                material_contact_force_N=material_contact,
                material_full_inertia_rhs_N=material_rhs,
                omega_material_rad_s=omega,
                objective_nonlinear_quadrature_tensor_Pa=nonlinear_material,
                full_inertia_linear_tensor_Pa=full_replay.tensor_pa,
                full_inertia_linear_displacement_m=full_replay.displacement,
                raw_corner_tensor_Pa=precise_corner,
                raw_corner_vm_Pa=precise_vm,
            )
            snapshots.append(step)
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "scope": __doc__,
        "source_run": str(args.run),
        "force_audit": str(args.force_audit),
        "states_checked": len(rows),
        "complete_requested_history": audit["complete_requested_history"],
        "saved_snapshot_steps": snapshots,
        "rows": rows,
        "input_sha256": input_hashes,
        "goal_completion": False,
        "limitations": "Every saved physical state keeps its global unaveraged nonlinear peak interval and reference hotspot position. Whole-volume Bernstein determinant/numerator bounds use saved wide coordinates; rounded FP64 field bounds and corner rounding differences are separate diagnostics. Computed-field bounds do not measure continuum error. Linear replay uses a consistent-mass fitted material frame; full-inertia versus rigid-relief differences are declared model diagnostics, not native numerical errors or matched rigid/deformable feedback. Only the audited source history is covered. Physical spatial/contact/geometry/time convergence, remaining finite-footprint models and verified warmed costs remain required.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_checked": len(rows), "complete_requested_history": audit["complete_requested_history"]}))


if __name__ == "__main__":
    main()
