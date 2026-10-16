"""Freshly audit actual native finite-load fields, including independent physical-point P2 traction scatter."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .frozen_traction_bulk_reference import assemble_samples
from .skfem_reference import SkfemElasticReference
from .solve_continuous_traction_axis import check_field
from .summarize_fine_references import CASES, closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scatter-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return inputs[str(path)]

    for run in [args.run, args.scatter_qualification]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    source = args.scatter_qualification / "source/research/rigid_stress/frozen_traction_bulk_reference.py"
    retain(source)
    assert inspect.getsource(assemble_samples) in source.read_text()
    qualification_path = args.scatter_qualification / "results/summary.json"
    retain(qualification_path)
    qualification = json.loads(qualification_path.read_text())
    assert qualification["qualify_only"] and qualification["all_original_five_cases_qualified"]
    assert qualification["original_traction_samples_unchanged"]
    summary_path = args.run / "results/summary.json"
    retain(summary_path)
    producer = json.loads(summary_path.read_text())
    assert [row["case"] for row in producer["cases"]] == list(CASES)
    mesh_hash = retain(args.mesh)
    with np.load(args.mesh) as saved:
        vertices, tets = saved["vertices"].copy(), saved["tetrahedra"].copy()
    reference = SkfemElasticReference(vertices, tets, 1e10, 0.3, 2000)
    edges = np.unique(np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=1), axis=0)
    native_coordinates = np.concatenate((vertices, vertices[edges].mean(axis=1)))
    permutation = reference.native_dof_permutation(native_coordinates)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in producer["cases"]:
        assert case["mesh_sha256"] == mesh_hash and case["patch_radius_m"] == 0.003
        assert case["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
        directory = args.run / "results" / case["case"]
        snapshot = directory / "snapshot.json"
        retain(snapshot)
        frozen_inputs = directory / "frozen_inputs.npz"
        retain(frozen_inputs)
        with np.load(frozen_inputs) as saved:
            np.testing.assert_array_equal(saved["omega_rad_s"], case["omega_rad_s"])
            for name, key in [
                ("points_m", "point_m"),
                ("forces_N", "force_N"),
                ("free_moments_Nm", "free_moment_Nm"),
                ("collision_normals", "normal"),
            ]:
                np.testing.assert_array_equal(saved[name], np.stack([patch[key] for patch in case["patches"]]))
        assert [model["model"] for model in case["models"]] == [
            "pressure",
            "componentwise",
            "pair_union",
            "body_source_prior",
        ]
        for model in case["models"]:
            name = model["model"]
            row = {
                "case": case["case"],
                "model": name,
                "accepted": model["accepted"],
                "classification": model["classification"],
                "frozen_input_sha256": inputs[str(frozen_inputs)],
            }
            if not model["accepted"]:
                assert model["classification"] in [
                    "unsupported_free_contact_moment",
                    "native_load_rejected",
                    "native_elastic_rejected",
                ]
                rows.append(row)
                continue
            assert model["classification"] == "numerically_qualified"
            mapped, diagnostic, field_path, traction_path = [
                directory / f"{name}_{suffix}.npz"
                for suffix in ["mapped_load", "elastic_diagnostic", "field", "traction"]
            ]
            for path in [mapped, diagnostic, field_path, traction_path]:
                retain(path)
            with np.load(mapped) as saved:
                assert np.all(saved["contact_status"] == 0)
                force = np.empty(reference.basis.N)
                force[permutation] = saved["force_N"]
            with np.load(diagnostic) as saved:
                u = np.empty(reference.basis.N)
                u[permutation] = saved["displacement_native_order"]
                rhs = reference.balanced_rhs(force, case["omega_rad_s"])
                np.testing.assert_allclose(saved["rhs_native_order_N"], rhs[permutation], rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(saved["expected_independent_rhs_N"], rhs, rtol=1e-10, atol=1e-13)
            aligned = reference.align_rigid_gauge(u)
            tensor, vm = reference.corner_stress(aligned)
            checked = check_field(reference, aligned, force, rhs, tensor, vm, case["omega_rad_s"])
            with np.load(field_path) as saved:
                raw = saved["raw_native_tensor_Pa"].copy()
                raw_vm = saved["vm_Pa"].copy()
                expected = tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
                np.testing.assert_allclose(raw, expected, rtol=1e-4, atol=1e-3)
                np.testing.assert_allclose(raw_vm, vm, rtol=1e-4, atol=1e-3)
                np.testing.assert_allclose(saved["tensor_Pa"], tensor, rtol=1e-4, atol=1e-3)
                np.testing.assert_allclose(saved["displacement"], aligned, rtol=1e-10, atol=1e-14)
            energy = float(0.5 * aligned @ (reference.stiffness @ aligned))
            np.testing.assert_allclose(energy, model["energy_J"], rtol=1e-8, atol=1e-20)
            np.testing.assert_allclose(raw_vm.max(), model["peak_Pa"], rtol=1e-4, atol=1e-3)
            with np.load(traction_path) as saved:
                points, tractions, area, normals = [
                    saved[key].copy() for key in ["positions_m", "forces_N", "areas_m2", "normals"]
                ]
                source_ids = saved["source"].copy()
                assert np.all(area > 0) and np.all(np.isfinite(points)) and np.all(np.isfinite(tractions))
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-12)
                scattered = assemble_samples(reference, points, tractions)
            np.testing.assert_allclose(scattered, force, rtol=1e-8, atol=1e-12)
            row.update(
                {
                    **checked,
                    "raw_native_corner_component_maximum_error_Pa": float(abs(raw - expected).max()),
                    "raw_native_corner_tensor_relative_L2_difference": float(
                        np.linalg.norm(raw - expected) / np.linalg.norm(expected)
                    ),
                    "raw_native_global_peak_Pa": float(raw_vm.max()),
                    "energy_J": energy,
                    "independent_physical_point_scatter_maximum_error_N": float(abs(scattered - force).max()),
                    "actual_traction_samples_retained": len(points),
                    "source_sample_counts": {
                        str(index): int(np.sum(source_ids == index)) for index in range(len(case["patches"]))
                    },
                    "all_original_corner_and_independent_quadrature_gates_passed": True,
                    "source_wrench_and_concentration_diagnostics": model["islands"],
                    "maximum_sampled_coulomb_violation_Pa": model["maximum_coulomb_violation_Pa"],
                }
            )
            np.savez_compressed(
                args.output / f"{case['case']}-{name}.npz",
                independently_scattered_force_N=scattered,
                complete_equilibrium_residual_N=reference.stiffness @ aligned - rhs,
                raw_native_corner_component_difference_Pa=raw - expected,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert len(rows) == 20
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "run": str(args.run),
        "mesh_sha256": mesh_hash,
        "tetrahedra": reference.mesh.nelements,
        "rows": rows,
        "accepted_fields_freshly_audited": sum(row["accepted"] for row in rows),
        "unsupported_or_rejected_inputs_retained": sum(not row["accepted"] for row in rows),
        "scatter_boundary_locator_maximum_plane_distance_m": getattr(reference, "_frozen_maximum_plane_distance", None),
        "scatter_quadratic_virtual_work_maximum_error_N_m2": getattr(
            reference, "_frozen_maximum_quadratic_work_difference", None
        ),
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Actual native force/RHS/displacements and every raw corner component/peak retain the original complete1e-11N absolute/1e-8 relative equilibrium and1e-4 relative/1e-3Pa raw-field gates. Fresh independent quadrature checks reconstruct the native displacement field, not a different exact-solve displacement. The manufactured/old-field-qualified P2 physical-point locator independently scatters every retained native traction sample without using its stored face or basis. Unsupported and rejected rows remain explicit. Source charts are per input source, not independently identified continuum contact islands. Same-mesh numerical agreement and cold runtime do not establish spatial/geometry/contact/model accuracy, feedback or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
