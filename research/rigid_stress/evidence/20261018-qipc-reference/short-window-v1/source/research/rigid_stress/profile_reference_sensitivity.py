"""Resolve alternative finite traction profiles with byte-identical original contact wrenches."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from skfem import FacetBasis, asm

from .finite_traction_reference import (
    assemble_finite_patch,
    physical_grip_patches,
    prescribed_vector_traction,
    subdivided_triangle_quadrature,
)
from .load_model_validation import traction_observables
from .skfem_reference import SkfemElasticReference

PROFILES = {"baseline": 0.45, "narrow": 0.25, "broad": 0.65, "quartic": None}


def tensor_l2(tensor, volumes):
    """Integrate the squared norm of each exact linear corner-tensor field."""
    return float(
        np.sqrt(
            np.sum(volumes / 20 * (np.sum(tensor**2, axis=(1, 2, 3)) + np.sum(tensor.sum(axis=1) ** 2, axis=(1, 2))))
        )
    )


def alternative_patch(reference, specification, original, target, profile_name, quadrature_level):
    basis = FacetBasis(
        reference.mesh,
        reference.element,
        facets=original.boundary_facets,
        quadrature=subdivided_triangle_quadrature(quadrature_level),
    )
    positions = basis.global_coordinates().transpose(1, 2, 0)
    normals = -basis.normals.transpose(1, 2, 0)
    radius_squared = np.sum((positions - specification.centre_m) ** 2, axis=-1) / specification.radius_m**2
    profile = np.maximum(1 - radius_squared, 0) ** 2
    width = PROFILES[profile_name]
    if width is not None:
        profile *= np.exp(-radius_squared / (2 * width**2))
    weights = profile * basis.dx
    integral = float(weights.sum())
    assert integral > 0
    centroid = np.sum(positions * weights[..., None], axis=(0, 1)) / integral
    direction = specification.direction / np.linalg.norm(specification.direction)
    directions = np.broadcast_to(direction, positions.shape).copy()
    if specification.is_local_normal_profile:
        tangent = np.array([0.0, 0.0, 1.0]) - normals[..., 2, None] * normals
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1), 1e-30)[..., None]
        directions = normals + specification.shear_fraction * specification.friction * tangent
    traction = profile[..., None] * directions * specification.strength_n / integral
    point, force, moment = target
    raw_force = np.sum(traction * basis.dx[..., None], axis=(0, 1))
    raw_moment = np.sum(np.cross(positions - point, traction * basis.dx[..., None]), axis=(0, 1))
    delta_force, delta_moment = force - raw_force, moment - raw_moment
    relative = positions - centroid
    inertia = np.sum(
        weights[..., None, None]
        * (np.sum(relative**2, axis=-1)[..., None, None] * np.eye(3) - relative[..., :, None] * relative[..., None, :]),
        axis=(0, 1),
    )
    couple = np.linalg.solve(inertia, delta_moment - np.cross(centroid - point, delta_force))
    correction = profile[..., None] * (delta_force / integral + np.cross(couple, relative))
    traction += correction
    weighted = traction * basis.dx[..., None]
    actual_force = weighted.sum(axis=(0, 1))
    actual_moment = np.sum(np.cross(positions - point, weighted), axis=(0, 1))
    np.testing.assert_allclose(actual_force, force, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(actual_moment, moment, rtol=0, atol=1e-13)
    active = profile > 0
    observables = traction_observables(
        positions[active], normals[active], weighted[active], basis.dx[active], specification.friction, reference.com
    )
    scale = max(observables["maximum_pressure_Pa"], observables["maximum_shear_Pa"], 1)
    admissible = (
        observables["minimum_pressure_Pa"] >= -1e-10 * scale
        and observables["maximum_coulomb_violation_Pa"] <= 1e-10 * scale
    )
    load = asm(prescribed_vector_traction, basis, traction=traction.transpose(2, 0, 1))
    nodal_force = load[reference.component_dofs]
    np.testing.assert_allclose(nodal_force.sum(axis=0), force, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(
        np.cross(reference.coordinates - point, nodal_force).sum(axis=0), moment, rtol=0, atol=1e-13
    )
    return (
        load,
        {
            "name": specification.name,
            "profile": profile_name,
            "gaussian_width_over_radius": width,
            "profile_integral_m2": integral,
            "force_error_N": (actual_force - force).tolist(),
            "moment_error_about_original_point_Nm": (actual_moment - moment).tolist(),
            "maximum_traction_correction_Pa": float(np.linalg.norm(correction, axis=-1).max()),
            "traction_admissible": bool(admissible),
            "observables": observables,
        },
        {
            "positions_m": positions[active],
            "normals": normals[active],
            "area_weights_m2": basis.dx[active],
            "tractions_Pa": traction[active],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--quadrature-level", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        vertices, tetrahedra = saved["vertices"], saved["tetrahedra"]
    reference = SkfemElasticReference(vertices, tetrahedra, 1e10, 0.3, 2000)
    volumes = abs(np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])) / 6
    mesh_hash = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    rows = []
    for snapshot_path in sorted(args.baseline.glob("*/snapshot.json")):
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["mesh_sha256"] == mesh_hash and snapshot["geometry"] == args.geometry
        case = snapshot["case"]
        original_inputs = snapshot_path.parent / "frozen_inputs.npz"
        with np.load(original_inputs) as saved:
            inputs = {key: saved[key].copy() for key in saved.files}
        specifications = physical_grip_patches(args.geometry, case, snapshot["patch_radius_m"])
        original_patches = [assemble_finite_patch(reference, p, args.quadrature_level) for p in specifications]
        native_fields = {}
        for model in snapshot["models"]:
            if model["accepted"]:
                path = snapshot_path.parent / f"{model['model']}_field.npz"
                with np.load(path) as saved:
                    native_fields[model["model"]] = (saved["tensor_Pa"].copy(), path)
        for profile_name in PROFILES:
            output = args.output / case / profile_name
            output.mkdir(parents=True)
            shutil.copy2(original_inputs, output / "frozen_inputs.npz")
            input_hash = hashlib.sha256(original_inputs.read_bytes()).hexdigest()
            assert hashlib.sha256((output / "frozen_inputs.npz").read_bytes()).hexdigest() == input_hash
            load, patches = np.zeros(reference.basis.N), []
            for index, (specification, original) in enumerate(zip(specifications, original_patches)):
                target = (inputs["points_m"][index], inputs["forces_N"][index], inputs["free_moments_Nm"][index])
                patch_load, record, arrays = alternative_patch(
                    reference, specification, original, target, profile_name, args.quadrature_level
                )
                np.savez_compressed(output / f"patch{index}.npz", **arrays)
                load += patch_load
                patches.append(record)
            solved = reference.solve(load, inputs["omega_rad_s"])
            assert solved.residual_budget_fraction <= 1
            np.savez_compressed(
                output / "reference_field.npz",
                displacement=solved.displacement,
                rhs_N=solved.rhs,
                force_N=load,
                tensor_Pa=solved.tensor_pa,
                vm_Pa=solved.von_mises_pa,
            )
            row = {
                "geometry": args.geometry,
                "case": case,
                "profile": profile_name,
                "quadrature_level": args.quadrature_level,
                "frozen_inputs_sha256": input_hash,
                "patches": patches,
                "traction_admissible": all(p["traction_admissible"] for p in patches),
                "raw_global_peak_Pa": float(solved.peak_pa),
                "raw_global_peak_position_m": solved.peak_position_m.tolist(),
                "energy_J": float(solved.energy_j),
                "complete_budget_fraction": float(solved.residual_budget_fraction),
                "models": [],
            }
            for model in snapshot["models"]:
                compared = {
                    "model": model["model"],
                    "native_classification": model["classification"],
                    "native_accepted": model["accepted"],
                }
                if model["accepted"]:
                    tensor, path = native_fields[model["model"]]
                    compared.update(
                        {
                            "source_native_field": str(path.resolve()),
                            "source_native_field_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "volume_l2_tensor_relative_difference": tensor_l2(tensor - solved.tensor_pa, volumes)
                            / tensor_l2(solved.tensor_pa, volumes),
                            "raw_peak_relative_difference": abs(model["peak_Pa"] / solved.peak_pa - 1),
                        }
                    )
                row["models"].append(compared)
            if profile_name == "baseline" and args.quadrature_level == snapshot["reference_quadrature_level"]:
                with np.load(snapshot_path.parent / "reference_field.npz") as saved:
                    baseline_difference = tensor_l2(saved["tensor_Pa"] - solved.tensor_pa, volumes) / tensor_l2(
                        solved.tensor_pa, volumes
                    )
                assert baseline_difference < 1e-8, baseline_difference
                row["baseline_full_tensor_relative_difference"] = baseline_difference
            (output / "snapshot.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
    report = {
        "scope": "Alternative independently integrated traction profiles with exactly fixed original six-component patch wrenches and byte-identical native inputs",
        "mesh_sha256": mesh_hash,
        "rows": rows,
        "native_prediction": "Existing qualified native fields are reused unchanged: native profiles are not altered",
        "profile_correction": "Profile-weighted constant force and linear couple correction preserves each original wrench; every local compression/friction violation remains visible",
        "limitations": "Fixed-discretization model sensitivity; continuum/contact/profile convergence and warmed costs remain separate. Inadmissible corrected profiles are not counted as physical references.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "admissible": sum(r["traction_admissible"] for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
