"""Separate resolved-load linearization from frozen-deformation constitutive differences on matched states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .neohookean_reference import NeoHookeanBulk
from .reference_motion_audit import motion_moments
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def linear_quadrature(bulk, vector):
    h = np.einsum("aie,ajeq->ijeq", vector[bulk.dofs], bulk.gradients, optimize=True).transpose(2, 3, 0, 1)
    return bulk.mu * (h + h.swapaxes(-1, -2)) + bulk.lam * np.trace(h, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)


def relative_tensor_error(first, second, weights):
    norm = np.sqrt(np.sum(weights[..., None, None] * second**2))
    delta = np.sqrt(np.sum(weights[..., None, None] * (first - second) ** 2))
    return {"absolute_volume_L2_Pa_m_3over2": float(delta), "relative_volume_L2": float(delta / max(norm, 1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.comparison)
    assert int((args.comparison / "returncode.txt").read_text()) == 0
    comparison_summary = args.comparison / "results/summary.json"
    compared = json.loads(comparison_summary.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {str(comparison_summary): hashlib.sha256(comparison_summary.read_bytes()).hexdigest()}
    for index, row in enumerate(compared["rows"]):
        assert row["complete_body_dynamic_residual_N"] <= 1e-9 and row["complete_mapped_dynamic_residual_N"] <= 1e-9
        source = (
            Path(row["state_source"])
            if "state_source" in row
            else Path(row["run"]) / f"results/step{row['step']:04d}.npz"
        )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == compared["input_sha256"][str(source)]
        hashes[str(source)] = compared["input_sha256"][str(source)]
        fields_path = args.comparison / f"results/case{index:02d}.npz"
        with np.load(fields_path) as saved:
            body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
            contact = saved["contact_force_N"][: body.basis.N].copy()
            acceleration = saved["body_acceleration_m_s2"].copy()
        permutation = body.component_dofs.reshape(-1)
        with np.load(source) as saved:
            full = saved["displacement"].copy()
            if "velocity" in saved:
                velocity = saved["velocity"][: body.basis.N].copy()
            else:
                velocity = (full[: body.basis.N] - saved["previous_displacement"][: body.basis.N]) / compared[
                    "physical_configuration"
                ]["dt"]
        vector = np.empty(body.basis.N)
        vector[permutation] = full[: body.basis.N]
        rotation, _, local = finite_frame(body, vector)
        bulk = NeoHookeanBulk(body)
        nonlinear_world, nonlinear_energy = quadrature_fields(bulk, vector)
        nonlinear_material = rotation.T @ nonlinear_world @ rotation
        gravity_acceleration = np.empty(body.basis.N)
        gravity_acceleration[body.component_dofs] = [0, 0, -9.81]
        gravity = (body.mass @ gravity_acceleration)[permutation]
        inertial = body.mass[permutation][:, permutation] @ acceleration
        force = np.zeros(body.basis.N)
        force[body.component_dofs] = (contact + gravity - inertial).reshape(-1, 3) @ rotation
        material_contact = np.zeros(body.basis.N)
        material_contact[body.component_dofs] = contact.reshape(-1, 3) @ rotation
        positions = body.coordinates + full[: body.basis.N].reshape(-1, 3)
        motion = motion_moments(
            body, permutation, np.asarray(positions, dtype=float), np.asarray(velocity, dtype=float)
        )
        omega = rotation.T @ motion["angular_velocity_world_rad_s"]
        resolved = body.solve(force)
        relief = body.solve(material_contact, omega)
        resolved_tensor = linear_quadrature(bulk, resolved.displacement)
        frozen_tensor = linear_quadrature(bulk, local)
        relief_tensor = linear_quadrature(bulk, relief.displacement)
        result = {
            "case": index,
            "tetrahedra": body.mesh.nelements,
            "time_s": row["time_s"],
            "resolved_linear_vs_nonlinear": relative_tensor_error(resolved_tensor, nonlinear_material, bulk.basis.dx),
            "frozen_objective_displacement_linear_vs_nonlinear": relative_tensor_error(
                frozen_tensor, nonlinear_material, bulk.basis.dx
            ),
            "rigid_relief_vs_resolved_full_inertia": relative_tensor_error(
                relief_tensor, resolved_tensor, bulk.basis.dx
            ),
            "raw_nonlinear_global_peak_lower_Pa": row["raw_peak_lower_Pa"],
            "raw_resolved_linear_global_peak_Pa": resolved.peak_pa,
            "raw_relief_linear_global_peak_Pa": relief.peak_pa,
            "raw_peak_linear_vs_nonlinear_relative_difference": abs(resolved.peak_pa / row["raw_peak_lower_Pa"] - 1),
            "nonlinear_energy_J": nonlinear_energy,
            "resolved_linear_energy_J": resolved.energy_j,
            "full_rhs_reference_balance_correction_N": float(np.linalg.norm(resolved.rhs - force)),
            "linear_complete_residual_N": resolved.residual_n,
        }
        rows.append(result)
        np.savez_compressed(
            args.output / f"case{index:02d}.npz",
            vertices=body.mesh.p.T,
            tetrahedra=body.mesh.t.T,
            material_full_inertia_rhs_N=force,
            material_contact_force_N=material_contact,
            omega_material_rad_s=omega,
            independent_resolved_linear_tensor_Pa=resolved.tensor_pa,
            objective_nonlinear_quadrature_tensor_Pa=nonlinear_material,
            resolved_linear_quadrature_tensor_Pa=resolved_tensor,
            frozen_displacement_linear_quadrature_tensor_Pa=frozen_tensor,
        )
        hashes[str(fields_path)] = hashlib.sha256(fields_path.read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in hashes.items())
    report = {
        "scope": __doc__,
        "comparison": str(args.comparison),
        "rows": rows,
        "input_sha256": hashes,
        "goal_completion": False,
        "limitations": "One matched state per mesh at the explicitly recorded physical time and unresolved spatial/contact settings. Replaying resolved full-inertia loads through independent linear FEM measures a combined finite-geometry/constitutive linearization discrepancy; the reference-wrench balance correction is explicit. Frozen-objective-deformation stress comparison diagnoses the constitutive assumption without claiming a new equilibrium solve. Rigid-relief comparison is not a corresponding solved rigid/deformable feedback case. Native numerical replay, physical convergence, full histories, remaining finite-footprint models and verified warmed cost remain separate requirements.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
