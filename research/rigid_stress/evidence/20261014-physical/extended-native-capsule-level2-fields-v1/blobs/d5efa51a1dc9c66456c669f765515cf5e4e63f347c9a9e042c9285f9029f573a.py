"""Reassemble the original five-level thickness FEM fields without repeating their solves."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from skfem.helpers import sym_grad
from skfem.models.elasticity import linear_stress

from .sample_load_fields import locate_probes
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import CASES, closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-audit", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, nargs=5, required=True)
    parser.add_argument("--field-run", type=Path, nargs=5, required=True)
    parser.add_argument("--probe-run", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.comparison_audit, *args.field_run, *args.probe_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    audit_path = args.comparison_audit / "results/summary.json"
    retain(audit_path)
    comparison = json.loads(audit_path.read_text())
    assert comparison["fields_verified"] == len(comparison["rows"]) == 25
    assert [row["layers"] for row in comparison["independent_boundary_and_surface_flux_audit"]["rows"]] == [
        2,
        3,
        4,
        5,
        6,
    ]
    for name, digest in comparison["independent_boundary_and_surface_flux_audit"]["unchanged_input_sha256"].items():
        assert retain(Path(name)) == digest
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for layers, mesh, run, probes in zip([2, 3, 4, 5, 6], args.mesh, args.field_run, args.probe_run, strict=True):
        mesh_hash = retain(mesh)
        with np.load(mesh) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        print(
            json.dumps(
                {"layers": layers, "tetrahedra": reference.mesh.nelements, "independent_matrices_assembled": True}
            ),
            flush=True,
        )
        for case in CASES:
            expected = next(row for row in comparison["rows"] if row["layers"] == layers and row["case"] == case)
            assert mesh_hash == expected["mesh_sha256"]
            field_path = run / f"results/{case}/reference_field.npz"
            snapshot_path = run / f"results/{case}/snapshot.json"
            assert retain(field_path) == expected["field_sha256"]
            assert retain(snapshot_path) == expected["snapshot_sha256"]
            config = json.loads(snapshot_path.read_text())
            assert config["mesh_sha256"] == mesh_hash
            assert config["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
            np.testing.assert_allclose(reference.total_mass, config["mass_kg"], rtol=1e-12, atol=1e-18)
            np.testing.assert_allclose(reference.com, config["com_m"], rtol=0, atol=1e-14)
            with np.load(field_path) as saved:
                u, force, rhs, tensor, vm = [
                    saved[key].copy() for key in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]
                ]
            reconstructed_rhs = reference.balanced_rhs(force, config["omega_rad_s"])
            np.testing.assert_allclose(reconstructed_rhs, rhs, rtol=1e-10, atol=1e-13)
            residual = reference.stiffness @ u - rhs
            norm = float(np.linalg.norm(residual))
            budget = max(1e-11, 1e-8 * float(np.linalg.norm(rhs)))
            assert norm <= budget
            rebuilt, rebuilt_vm = reference.corner_stress(u)
            np.testing.assert_allclose(rebuilt, tensor, rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(rebuilt_vm, vm, rtol=1e-10, atol=1e-6)
            quadrature = linear_stress(*reference.lame)(sym_grad(reference.basis.interpolate(u))).transpose(2, 3, 0, 1)
            barycentric = np.vstack((1 - reference.basis.X.sum(axis=0), reference.basis.X))
            corner_interpolated = np.einsum("aq,taij->tqij", barycentric, tensor)
            np.testing.assert_allclose(quadrature, corner_interpolated, rtol=1e-10, atol=1e-6)
            probe_path = probes / f"results/{case}/reference_field.npz"
            assert retain(probe_path) == expected["probe_field_sha256"]
            with np.load(probe_path) as saved:
                points = saved["points_m"]
                assert len(points) == 36 and saved["valid"].all()
                elements, valid, local = locate_probes(reference.mesh, points)
                assert valid.all()
                probe_tensor = np.einsum("na,naij->nij", local, rebuilt[elements])
                deviator = probe_tensor - np.trace(probe_tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
                probe_vm = np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1)))
                aligned = reference.align_rigid_gauge(u)
                probe_u = (reference.basis.probes(points.T) @ aligned).reshape(3, -1).T
                np.testing.assert_allclose(probe_tensor, saved["tensor_Pa"], rtol=1e-10, atol=1e-6)
                np.testing.assert_allclose(probe_vm, saved["vm_Pa"], rtol=1e-10, atol=1e-6)
                np.testing.assert_allclose(probe_u, saved["displacement_m"], rtol=1e-8, atol=1e-15)
            energy = float(0.5 * u @ (reference.stiffness @ u))
            np.testing.assert_allclose(energy, config["reference_energy_J"], rtol=1e-8, atol=1e-20)
            peak_cell, peak_corner = np.unravel_index(np.argmax(rebuilt_vm), rebuilt_vm.shape)
            peak_position = reference.corner_positions[peak_cell, peak_corner]
            np.testing.assert_array_equal(peak_position, config["reference_peak_position_m"])
            row = {
                "layers": layers,
                "case": case,
                "tetrahedra": reference.mesh.nelements,
                "complete_reassembled_residual_N": norm,
                "original_budget_N": budget,
                "complete_reassembled_budget_fraction": norm / budget,
                "raw_global_peak_Pa": float(rebuilt_vm.max()),
                "raw_peak_reference_position_m": peak_position.tolist(),
                "reassembled_energy_J": energy,
                "full_corner_tensor_max_difference_Pa": float(np.max(abs(rebuilt - tensor))),
                "full_quadrature_vs_corner_interpolation_max_difference_Pa": float(
                    np.max(abs(quadrature - corner_interpolated))
                ),
                "all_original_physical_probes_checked": 36,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del (
                u,
                force,
                rhs,
                tensor,
                vm,
                reconstructed_rhs,
                residual,
                rebuilt,
                rebuilt_vm,
                quadrature,
                corner_interpolated,
                aligned,
            )
        del reference
        gc.collect()
    assert len(rows) == 25
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "fields_independently_reassembled": len(rows),
        "rows": rows,
        "all_original_complete_force_budgets_passed": True,
        "all_full_corner_and_quadrature_tensors_and_original_probes_checked": True,
        "original_adjacent_changes": comparison["adjacent_changes"],
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "Reassembled scikit-fem stiffness, consistent mass, inertia relief and full field reconstruction extend the existing fixed-boundary/traction archive audit; no new linear solve or changed physical problem is introduced. Original36-probe changes and unaveraged global peaks remain unchanged. Quadrature stresses are compared with the complete saved affine corner field. Original force samples are retained rather than replaced by a new load law. This closes a numerical evidence scope gap, not the unresolved physical tensor tail, spatial/contact/geometry/model/feedback acceptance or verified warmed native cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
