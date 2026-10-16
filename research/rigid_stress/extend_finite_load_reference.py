"""Qualify unchanged original finite-load fields, then extend their CAD/volume family without importing production numerics."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .finite_traction_reference import assemble_finite_patch, physical_grip_patches
from .qualify_reference_scaling import AMGPinFactor
from .sample_load_fields import common_physical_probes, locate_probes, sample_field
from .skfem_reference import SkfemElasticReference
from .solve_continuous_traction_axis import check_field
from .summarize_fine_references import CASES, closed_source, relative_change


def sample(reference, displacement, tensor, points):
    elements, valid, barycentric = locate_probes(reference.mesh, points)
    aligned = reference.align_rigid_gauge(displacement)
    fields = sample_field(
        reference.basis, elements, valid, barycentric, points, {"tensor_Pa": tensor, "displacement": aligned}
    )
    return {
        "points_m": points,
        "valid": valid,
        **dict(zip(["tensor_Pa", "vm_Pa", "displacement_m"], fields, strict=True)),
    }


def assemble_case(reference, geometry, case, radius, quadrature):
    patches = [
        assemble_finite_patch(reference, spec, quadrature) for spec in physical_grip_patches(geometry, case, radius)
    ]
    load = np.zeros(reference.basis.N)
    for patch in patches:
        weighted = patch.tractions_pa * patch.area_weights_m2[:, None]
        expected = np.r_[weighted.sum(axis=0), np.cross(patch.positions_m - reference.com, weighted).sum(axis=0)]
        np.testing.assert_allclose(reference.rigid_modes.T @ patch.load_vector_n, expected, rtol=1e-11, atol=1e-12)
        load += patch.load_vector_n
    return load, patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", choices=["ellipsoid", "capsule", "thin_neck", "u", "narrow_gap", "cavity"], required=True
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--baseline-mesh", type=Path, required=True)
    parser.add_argument("--baseline-probes", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return digest

    for run in [args.baseline_run, args.baseline_probes]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    mesh_sha = retain(args.mesh)
    baseline_sha = retain(args.baseline_mesh)
    for module in ["finite_traction_reference", "skfem_reference"]:
        path = Path(f"research/rigid_stress/{module}.py")
        original = args.baseline_run / "source" / path
        retain(original)
        assert original.read_bytes() == path.read_bytes()
    points = common_physical_probes(args.geometry)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.qualify_only:
        assert mesh_sha == baseline_sha and args.qualification is None
        with np.load(args.baseline_mesh) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        rows = []
        for case in CASES:
            field_path = args.baseline_run / f"results/{case}/reference_field.npz"
            snapshot_path = args.baseline_run / f"results/{case}/snapshot.json"
            probe_path = args.baseline_probes / f"results/{case}/reference_field.npz"
            for path in [field_path, snapshot_path, probe_path]:
                retain(path)
            snapshot = json.loads(snapshot_path.read_text())
            assert snapshot["mesh_sha256"] == baseline_sha and snapshot["geometry"] == args.geometry
            assert snapshot["patch_radius_m"] == 0.003 and snapshot["reference_quadrature_level"] == 3
            assert snapshot["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
            np.testing.assert_allclose(reference.total_mass, snapshot["mass_kg"], rtol=1e-12, atol=1e-18)
            force, _ = assemble_case(reference, args.geometry, case, 0.003, 3)
            with np.load(field_path) as saved:
                u, original_force, rhs, tensor, vm = [
                    saved[name].copy() for name in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]
                ]
            np.testing.assert_allclose(force, original_force, rtol=1e-11, atol=1e-13)
            checked = check_field(reference, u, force, rhs, tensor, vm, snapshot["omega_rad_s"])
            probes = sample(reference, u, tensor, points)
            with np.load(probe_path) as saved:
                np.testing.assert_array_equal(saved["points_m"], points)
                np.testing.assert_array_equal(saved["valid"], probes["valid"])
                for name, absolute in [("tensor_Pa", 1e-6), ("vm_Pa", 1e-6), ("displacement_m", 1e-13)]:
                    np.testing.assert_allclose(
                        probes[name][probes["valid"]], saved[name][probes["valid"]], rtol=1e-10, atol=absolute
                    )
            rows.append(
                {
                    "case": case,
                    "original_full_nodal_force_maximum_difference_N": float(abs(force - original_force).max()),
                    "all_original_probe_coordinates_and_validity_retained": True,
                    "outside_probe_indices": np.flatnonzero(~probes["valid"]).tolist(),
                    **checked,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        summary = {
            "scope": __doc__,
            "qualify_only": True,
            "geometry": args.geometry,
            "mesh_sha256": mesh_sha,
            "tetrahedra": reference.mesh.nelements,
            "all_original_five_loads_full_fields_and_probes_qualified": True,
            "rows": rows,
            "new_equilibrium_states_solved": 0,
            "input_sha256": inputs,
            "goal_completion": False,
            "limitations": "Fresh stiffness/mass/RHS, all original complete-force budgets, original continuous traction vectors, every corner/quadrature tensor and every original physical probe are rechecked without a new solve. This qualifies the isolated pure-FEM reference path on the existing mesh, not a physical tail or native model/cost accuracy.",
        }
    else:
        assert args.qualification is not None
        closed_source(args.qualification)
        assert int((args.qualification / "returncode.txt").read_text()) == 0
        qualification_path = args.qualification / "results/summary.json"
        retain(qualification_path)
        qualification = json.loads(qualification_path.read_text())
        assert (
            qualification["qualify_only"] and qualification["all_original_five_loads_full_fields_and_probes_qualified"]
        )
        assert qualification["geometry"] == args.geometry and qualification["mesh_sha256"] == baseline_sha
        assert len(qualification["rows"]) == 5 and all(
            row["original_budget_fraction"] <= 1 for row in qualification["rows"]
        )
        for name, expected in qualification["input_sha256"].items():
            assert retain(Path(name)) == expected
        for module in [
            "extend_finite_load_reference",
            "solve_continuous_traction_axis",
            "sample_load_fields",
            "finite_traction_reference",
            "skfem_reference",
        ]:
            path = Path(f"research/rigid_stress/{module}.py")
            assert path.read_bytes() == (args.qualification / "source" / path).read_bytes()
        metadata_path = args.mesh.with_suffix(".json")
        retain(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        assert metadata["mesh_sha256"] == mesh_sha and metadata["geometry"]["name"] == args.geometry
        with np.load(args.mesh) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        assert reference.mesh.nelements > qualification["tetrahedra"]
        operator = (reference.stiffness[reference.free_dofs][:, reference.free_dofs] / reference.young).tocsr()
        modes = reference.rigid_modes[reference.free_dofs]
        np.random.seed(20261016)
        reference.factor = AMGPinFactor(operator, modes / np.linalg.norm(modes, axis=0))
        rows = []
        for case in CASES:
            force, patches = assemble_case(reference, args.geometry, case, 0.003, 3)
            omega = np.array([0.4, -0.7, 0.9]) if case == "asymmetric" else np.zeros(3)
            solved = reference.solve(force, omega)
            checked = check_field(
                reference, solved.displacement, force, solved.rhs, solved.tensor_pa, solved.von_mises_pa, omega
            )
            probes = sample(reference, solved.displacement, solved.tensor_pa, points)
            output = args.output / case
            output.mkdir()
            np.savez_compressed(
                output / "reference_field.npz",
                displacement=solved.displacement,
                force_N=force,
                rhs_N=solved.rhs,
                tensor_Pa=solved.tensor_pa,
                vm_Pa=solved.von_mises_pa,
            )
            np.savez_compressed(output / "probe_field.npz", **probes)
            patch_rows = []
            for index, patch in enumerate(patches):
                np.savez_compressed(
                    output / f"reference_patch{index}.npz",
                    positions_m=patch.positions_m,
                    normals=patch.normals,
                    area_weights_m2=patch.area_weights_m2,
                    tractions_Pa=patch.tractions_pa,
                )
                weighted = patch.tractions_pa * patch.area_weights_m2[:, None]
                patch_rows.append(
                    {
                        "name": patch.specification.name,
                        "physical_anchor_m": patch.specification.centre_m.tolist(),
                        "force_N": patch.force_n.tolist(),
                        "witness_m": patch.witness_m.tolist(),
                        "point_m": patch.point_m.tolist(),
                        "free_moment_Nm": patch.free_moment_nm.tolist(),
                        "normal": patch.inward_normal.tolist(),
                        "friction": patch.specification.friction,
                        "profile_integral_m2": float(patch.profile_integral_m2),
                        "integrated_force_N": weighted.sum(axis=0).tolist(),
                        "integrated_moment_about_COM_Nm": np.cross(patch.positions_m - reference.com, weighted)
                        .sum(axis=0)
                        .tolist(),
                        "specification": asdict(patch.specification),
                    }
                )
            row = {
                "case": case,
                "raw_global_peak_Pa": float(solved.peak_pa),
                "raw_global_peak_position_m": solved.peak_position_m.tolist(),
                "energy_J": float(solved.energy_j),
                "amg_cg_iterations": len(reference.factor.residuals),
                "original_probe_count": len(points),
                "outside_probe_indices": np.flatnonzero(~probes["valid"]).tolist(),
                **checked,
            }
            snapshot = {
                "geometry": args.geometry,
                "case": case,
                "mesh_sha256": mesh_sha,
                "tetrahedra": reference.mesh.nelements,
                "material": {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000},
                "mass_kg": float(reference.total_mass),
                "com_m": reference.com.tolist(),
                "pose": {"position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                "omega_rad_s": omega.tolist(),
                "patch_radius_m": 0.003,
                "reference_quadrature_level": 3,
                "reference_peak_Pa": float(solved.peak_pa),
                "reference_peak_position_m": solved.peak_position_m.tolist(),
                "reference_energy_J": float(solved.energy_j),
                "reference_complete_budget_fraction": checked["original_budget_fraction"],
                "patches": patch_rows,
                "models": [],
            }
            (output / "snapshot.json").write_text(
                json.dumps(snapshot, default=lambda value: value.tolist(), indent=2) + "\n"
            )
            baseline_probes = args.baseline_probes / f"results/{case}/reference_field.npz"
            with np.load(baseline_probes) as saved:
                common = saved["valid"] & probes["valid"]
                row["comparison_with_original_finest_probes"] = {
                    "common": int(common.sum()),
                    "excluded_indices": np.flatnonzero(~common).tolist(),
                    **{
                        name: dict(
                            zip(
                                ["relative_change", "absolute_change"],
                                relative_change(saved[name][common], probes[name][common]),
                                strict=True,
                            )
                        )
                        for name in ["tensor_Pa", "vm_Pa", "displacement_m"]
                    },
                }
            rows.append(row)
            print(json.dumps(row), flush=True)
        summary = {
            "scope": __doc__,
            "qualify_only": False,
            "geometry": args.geometry,
            "mesh_sha256": mesh_sha,
            "mesh": str(args.mesh),
            "mesh_metadata": metadata,
            "tetrahedra": reference.mesh.nelements,
            "qualification": str(args.qualification),
            "rows": rows,
            "new_equilibrium_states_solved": len(rows),
            "input_sha256": inputs,
            "goal_completion": False,
            "limitations": "Every original physical center,3mm radius/profile/strength, friction, material/density and omega remains unchanged. These independently generated CAD/volume meshes change represented geometry and volume together; separate fixed-domain and integration axes remain required. All original physical probes, outside indices, complete corner fields, raw global peaks and original force budgets remain. New fields require a separately closed fresh auditor before physical interpretation; this producer alone does not establish a converged tail, native model comparison, resolved-contact or warmed accuracy/cost acceptance.",
        }
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
