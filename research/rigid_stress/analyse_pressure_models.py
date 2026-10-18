"""Audit current mapped stress fields, supplied-pad cones and material support."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.rigid_stress.neohookean_reference import NeoHookeanBulk
from research.rigid_stress.reconstruct_franka_packets import linear_quadrature, weighted_error
from research.rigid_stress.skfem_reference import SkfemElasticReference


@dataclass(frozen=True)
class SourcePriors:
    points: np.ndarray
    forces: np.ndarray
    moments: np.ndarray
    normals: np.ndarray
    friction: np.ndarray
    witnesses: np.ndarray


def affine_l2(tensor, volumes):
    square = np.sum(tensor**2, axis=(1, 2, 3)) + np.sum(tensor.sum(axis=1) ** 2, axis=(1, 2))
    return np.sqrt(np.sum(volumes * square / 20))


def cloud_audit(positions, normals, forces, areas, source, friction, priors, origin, geometry):
    assert len(positions) == len(areas) and (areas > 0).all()
    pad_normals = priors.normals / np.linalg.norm(priors.normals, axis=1)[:, None]
    density = forces / areas[:, None]
    magnitude = np.linalg.norm(forces, axis=1)
    scale = np.linalg.norm(priors.forces, axis=1).sum()
    allowance = 64 * np.finfo(float).eps * scale
    cone = []
    for name, directions in [("supplied_pad", pad_normals[source]), ("material", normals)]:
        normal = np.einsum("qi,qi->q", forces, directions)
        tangent = forces - normal[:, None] * directions
        violation = np.maximum(np.linalg.norm(tangent, axis=1) - friction * normal, -normal)
        cone.append(
            (
                name,
                {
                    "maximum_raw_violation_Pa": float(np.maximum(violation / areas, 0).max()),
                    "maximum_raw_violation_N": float(np.maximum(violation, 0).max()),
                    "violating_force_fraction": float(
                        magnitude[violation > allowance].sum() / max(magnitude.sum(), np.finfo(float).tiny)
                    ),
                    "sample_force_allowance_N": float(allowance),
                },
            )
        )
    total_force = forces.sum(axis=0)
    total_moment = np.cross(positions - origin, forces).sum(axis=0)
    target_moments = np.cross(priors.points - origin, priors.forces) + priors.moments
    rows = []
    for index in range(len(priors.points)):
        selected = source == index
        resultant = forces[selected].sum(axis=0)
        moment = np.cross(positions[selected] - origin, forces[selected]).sum(axis=0)
        normal = np.dot(priors.forces[index], pad_normals[index])
        shear = np.linalg.norm(priors.forces[index] - normal * pad_normals[index])
        rows.append(
            {
                "source": index,
                "target_force_N": priors.forces[index].tolist(),
                "realized_force_N": resultant.tolist(),
                "force_shift_N": (resultant - priors.forces[index]).tolist(),
                "moment_shift_Nm": (moment - target_moments[index]).tolist(),
                "prior_friction_mobilization": float(
                    shear / max(priors.friction[index] * normal, np.finfo(float).tiny)
                ),
                "loaded_area_m2": float(areas[selected & (magnitude > 0)].sum()),
                "maximum_traction_Pa": float(np.linalg.norm(density[selected], axis=1).max()) if selected.any() else 0,
            }
        )
    opposed = np.einsum("qi,qi->q", normals, pad_normals[source]) < 0
    cross_gap = (
        (positions[:, 0] * priors.witnesses[source, 0] < 0)
        if geometry == "narrow_gap"
        else np.zeros(len(source), dtype=bool)
    )
    return {
        "samples": len(areas),
        "cones": dict(cone),
        "total_force_error_N": float(np.linalg.norm(total_force - priors.forces.sum(axis=0))),
        "total_moment_error_Nm": float(np.linalg.norm(total_moment - target_moments.sum(axis=0))),
        "opposing_material_normal_force_fraction": float(
            magnitude[opposed].sum() / max(magnitude.sum(), np.finfo(float).tiny)
        ),
        "cross_gap_force_fraction": float(magnitude[cross_gap].sum() / max(magnitude.sum(), np.finfo(float).tiny))
        if geometry == "narrow_gap"
        else None,
        "sources": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    inputs = {}

    def record(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        return json.loads(record(path).read_text())

    original = read(args.packets / "results/summary.json")
    mesh = record(args.packets / "results/body_mesh.npz")
    with np.load(mesh) as saved:
        reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    bulk = NeoHookeanBulk(reference)
    egg = read(args.root / "egg-rlm-v1/results/summary.json")
    egg_rows = []
    for row in egg["rows"]:
        if row["classification"] != "accepted_conditional_snapshot":
            egg_rows.append(row)
            continue
        step, radius, model = row["step"], row["radius_m"], row["model"]
        label = f"step{step:05d}-r{round(radius * 1e6)}-{model}"
        packets = read(args.packets / f"results/step{step:05d}-packets.json")
        priors = SourcePriors(
            np.array([p["point_m"] for p in packets]),
            np.array([p["force_N"] for p in packets]),
            np.array([p["free_moment_Nm"] for p in packets]),
            np.array([p["inward_normal"] for p in packets]),
            np.array([p["friction"] for p in packets]),
            np.array([p["point_m"] for p in packets]),
        )
        with np.load(record(args.packets / f"results/step{step:05d}.npz")) as saved:
            linear = linear_quadrature(bulk, saved["resolved_linear_displacement_m"])
            nonlinear = saved["resolved_nonlinear_quadrature_tensor_Pa"]
        with np.load(record(args.root / "egg-rlm-v1/results" / (label + ".npz"))) as saved:
            assert all(np.isfinite(saved[key]).all() for key in saved.files)
            expected_tensor = saved["corner_tensor_Pa"][..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
            np.testing.assert_allclose(saved["native_tensor_Pa"], expected_tensor, rtol=1e-9, atol=1e-5)
            ml = weighted_error(saved["quadrature_tensor_Pa"], linear, bulk.basis.dx)
            mr = weighted_error(saved["quadrature_tensor_Pa"], nonlinear, bulk.basis.dx)
            np.testing.assert_allclose(ml["relative_L2"], row["M_vs_L"]["relative_L2"], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(mr["relative_L2"], row["M_vs_R"]["relative_L2"], rtol=1e-12, atol=1e-15)
            rhs = reference.balanced_rhs(saved["full_inertia_rhs_N"], np.zeros(3))
            residual = np.linalg.norm(reference.stiffness @ saved["displacement_m"] - rhs)
            budget = max(1e-11, 1e-8 * np.linalg.norm(rhs))
            assert residual <= budget
            cloud = cloud_audit(
                saved["traction_positions_m"],
                saved["traction_normals"],
                saved["traction_forces_N"],
                saved["traction_areas_m2"],
                saved["packet"],
                priors.friction[saved["packet"]],
                priors,
                reference.com,
                "egg",
            )
            peak = saved["corner_vm_Pa"].max()
            np.testing.assert_allclose(peak, row["M_raw_global_peak_Pa"], rtol=1e-4, atol=1e-3)
        egg_rows.append(
            {
                **row,
                "audit_M_vs_L": ml,
                "audit_M_vs_R": mr,
                "audit_complete_budget_fraction": float(residual / budget),
                "traction_audit": cloud,
            }
        )
    controls = []
    for directory in sorted(args.root.glob("control-*-v1")):
        summary = read(directory / "results/summary.json")
        case0 = summary["cases"][0]
        old_root = args.root.parent / "20261014-physical-validation"
        mesh = record(old_root / f"matrix-mesh-{case0['geometry']}-v1/meshes/h0.0035.npz")
        with np.load(mesh) as saved:
            control_ref = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        coordinates = control_ref.mesh.p.T[control_ref.mesh.t.T]
        volumes = abs(np.linalg.det(coordinates[:, 1:] - coordinates[:, :1])) / 6
        for case in summary["cases"]:
            folder = directory / "results" / case["case"]
            patches = case["patches"]
            priors = SourcePriors(
                np.array([p["point_m"] for p in patches]),
                np.array([p["force_N"] for p in patches]),
                np.array([p["free_moment_Nm"] for p in patches]),
                np.array([p["normal"] for p in patches]),
                np.array([p["friction"] for p in patches]),
                np.array([p["witness_m"] for p in patches]),
            )
            with np.load(record(folder / "reference_field.npz")) as saved:
                resolved = saved["tensor_Pa"].copy()
            for model in case["models"]:
                result = {
                    "geometry": case["geometry"],
                    "case": case["case"],
                    "radius_m": case["patch_radius_m"],
                    "reference_peak_Pa": case["reference_peak_Pa"],
                    "reference_peak_position_m": case["reference_peak_position_m"],
                    **model,
                }
                if model["classification"] != "numerically_qualified":
                    controls.append(result)
                    continue
                name = model["model"]
                with (
                    np.load(record(folder / f"{name}_field.npz")) as field,
                    np.load(record(folder / f"{name}_elastic_diagnostic.npz")) as diagnostic,
                ):
                    rhs = diagnostic["expected_independent_rhs_N"]
                    residual = np.linalg.norm(control_ref.stiffness @ field["raw_native_displacement"] - rhs)
                    budget = max(1e-11, 1e-8 * np.linalg.norm(rhs))
                    assert residual <= budget
                    tensor = field["tensor_Pa"]
                    np.testing.assert_allclose(
                        field["raw_native_tensor_Pa"],
                        tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                        rtol=1e-4,
                        atol=1e-3,
                    )
                    result["weighted_volume_tensor_relative_L2"] = float(
                        affine_l2(tensor - resolved, volumes) / affine_l2(resolved, volumes)
                    )
                    result["audit_complete_budget_fraction"] = float(residual / budget)
                with np.load(record(folder / f"{name}_traction.npz")) as traction:
                    result["traction_audit"] = cloud_audit(
                        traction["positions_m"],
                        traction["normals"],
                        traction["forces_N"],
                        traction["areas_m2"],
                        traction["source"],
                        traction["friction"],
                        priors,
                        control_ref.com,
                        case["geometry"],
                    )
                controls.append(result)
    expected_egg = {
        (row["step"], radius, model)
        for row in original["rows"]
        for radius in (0.003, 0.006)
        for model in ("pressure", "body_source_prior")
    }
    assert len(egg_rows) == 16 and {(row["step"], row["radius_m"], row["model"]) for row in egg_rows} == expected_egg
    expected_controls = {
        (geometry, radius, case, model)
        for geometry in ("ellipsoid", "narrow_gap")
        for radius in (0.003, 0.0024)
        for case in ("opposing", "asymmetric", "near_slip", "multiple")
        for model in ("pressure", "body_source_prior")
    }
    assert (
        len(controls) == 32
        and {(row["geometry"], row["radius_m"], row["case"], row["model"]) for row in controls} == expected_controls
    )
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "egg": egg_rows,
        "controls": controls,
        "input_sha256": inputs,
        "diagnostic_definitions": {
            "cone_force_allowance": "64*FP64_epsilon*sum(norm(original source forces)); raw violations are also retained",
            "cross_gap_force_fraction": "force norm on samples with x opposite to the witnessed source on the declared narrow-gap U arms",
            "control_tensor_error": "exact affine-corner volume-integrated Frobenius L2 relative to fixed continuous-traction reference",
            "egg_tensor_error": "independent FEM volume-quadrature Frobenius L2; source-prior is a model, not truth",
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"egg_rows": len(egg_rows), "control_rows": len(controls), "passed": True}))


if __name__ == "__main__":
    main()
