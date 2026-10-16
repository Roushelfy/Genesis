"""Isolate bulk refinement using byte-identical physical traction quadrature samples."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .qualify_reference_scaling import AMGPinFactor
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import CASES, closed_source


def assemble_samples(reference, positions, forces):
    load = np.zeros(reference.basis.N)
    for first in range(0, len(positions), 256):
        section = slice(first, first + 256)
        evaluation = reference.basis.probes(positions[section].T)
        load += evaluation.T @ forces[section].T.reshape(-1)
    expected = np.r_[forces.sum(axis=0), np.cross(positions - reference.com, forces).sum(axis=0)]
    np.testing.assert_allclose(reference.rigid_modes.T @ load, expected, rtol=1e-11, atol=1e-12)
    return load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--parent-mesh", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-probes", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--mesh-build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for run in [args.parent_run, args.parent_probes]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    mesh_sha, parent_sha = retain(args.mesh), retain(args.parent_mesh)
    if args.qualify_only:
        assert mesh_sha == parent_sha
    else:
        assert args.qualification is not None and args.mesh_build is not None
        for run in [args.qualification, args.mesh_build]:
            closed_source(run)
            assert int((run / "returncode.txt").read_text()) == 0
        qualification_path = args.qualification / "results/summary.json"
        retain(qualification_path)
        qualification = json.loads(qualification_path.read_text())
        assert qualification["qualify_only"] and qualification["all_original_five_cases_qualified"]
        assert qualification["mesh_sha256"] == parent_sha
        assert qualification["original_traction_samples_unchanged"]
        module = Path("research/rigid_stress/frozen_traction_bulk_reference.py")
        assert (args.qualification / "source" / module).read_bytes() == Path(__file__).read_bytes()
        for path, digest in qualification["input_sha256"].items():
            assert retain(Path(path)) == digest
        metadata = json.loads(args.mesh.with_suffix(".json").read_text())
        retain(args.mesh.with_suffix(".json"))
        assert metadata["seed_sha256"] == parent_sha and metadata["mesh_sha256"] == mesh_sha
        assert metadata["local_refinement_level"] > 0 and metadata["parent_partition"]["each_parent_volume_preserved"]
        assert args.mesh.parent == args.mesh_build / "results"
    with np.load(args.mesh) as saved:
        reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    if not args.qualify_only:
        operator = (reference.stiffness[reference.free_dofs][:, reference.free_dofs] / reference.young).tocsr()
        modes = reference.rigid_modes[reference.free_dofs]
        np.random.seed(20261015)
        reference.factor = AMGPinFactor(operator, modes / np.linalg.norm(modes, axis=0))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in CASES:
        parent = args.parent_run / f"results/{case}"
        snapshot_path = parent / "snapshot.json"
        retain(snapshot_path)
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["mesh_sha256"] == parent_sha and snapshot["geometry"] == "egg"
        assert snapshot["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
        np.testing.assert_allclose(reference.total_mass, snapshot["mass_kg"], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(reference.com, snapshot["com_m"], rtol=0, atol=1e-14)
        load = np.zeros(reference.basis.N)
        patch_hashes = []
        for index in range(len(snapshot["patches"])):
            path = parent / f"reference_patch{index}.npz"
            patch_hashes.append(retain(path))
            with np.load(path) as saved:
                load += assemble_samples(
                    reference, saved["positions_m"], saved["tractions_Pa"] * saved["area_weights_m2"][:, None]
                )
        probe_path = args.parent_probes / f"results/{case}/reference_field.npz"
        retain(probe_path)
        with np.load(probe_path) as saved:
            points = saved["points_m"].copy()
            assert np.all(saved["valid"])
        output = args.output / case
        output.mkdir()
        if args.qualify_only:
            path = parent / "reference_field.npz"
            retain(path)
            with np.load(path) as saved:
                displacement, original_load, original_rhs, original_tensor, original_vm = [
                    saved[key].copy() for key in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]
                ]
            np.testing.assert_allclose(load, original_load, rtol=1e-11, atol=1e-13)
            rhs = reference.balanced_rhs(load, snapshot["omega_rad_s"])
            np.testing.assert_allclose(rhs, original_rhs, rtol=1e-10, atol=1e-13)
            tensor, vm = reference.corner_stress(displacement)
            np.testing.assert_allclose(tensor, original_tensor, rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(vm, original_vm, rtol=1e-12, atol=1e-8)
            strain = reference.basis.interpolate(displacement).grad.transpose(2, 3, 0, 1)
            lam, mu = reference.lame
            quadrature = mu * (strain + strain.swapaxes(-1, -2))
            quadrature += lam * np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
            barycentric = np.column_stack((1 - reference.scalar_basis.X.sum(axis=0), reference.scalar_basis.X.T))
            from_corners = np.einsum("qc,ecij->eqij", barycentric, original_tensor)
            np.testing.assert_allclose(quadrature, from_corners, rtol=1e-10, atol=1e-6)
            residual = float(np.linalg.norm(reference.stiffness @ displacement - rhs))
            budget = max(1e-11, 1e-8 * float(np.linalg.norm(rhs)))
            assert residual <= budget
            energy = float(0.5 * displacement @ (reference.stiffness @ displacement))
            row = {
                "case": case,
                "original_load_max_difference_N": float(np.max(abs(load - original_load))),
                "original_rhs_max_difference_N": float(np.max(abs(rhs - original_rhs))),
                "all_original_corner_vm_fields_preserved": True,
                "all_original_quadrature_fields_qualified": True,
                "quadrature_max_difference_Pa": float(np.max(abs(quadrature - from_corners))),
                "complete_residual_N": residual,
                "original_complete_budget_N": budget,
                "original_complete_budget_fraction": residual / budget,
            }
        else:
            solved = reference.solve(load, snapshot["omega_rad_s"])
            assert solved.residual_budget_fraction <= 1
            displacement, rhs, tensor, vm, energy = (
                solved.displacement,
                solved.rhs,
                solved.tensor_pa,
                solved.von_mises_pa,
                solved.energy_j,
            )
            row = {
                "case": case,
                "complete_residual_N": float(solved.residual_n),
                "original_complete_budget_fraction": float(solved.residual_budget_fraction),
                "amg_cg_iterations": len(reference.factor.residuals),
            }
        cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
        probe_tensor, probe_vm = reference.stress_at_points(displacement, points)
        probe_displacement = (reference.basis.probes(points.T) @ displacement).reshape(3, -1).T
        row.update(
            raw_global_peak_Pa=float(vm.max()),
            raw_global_peak_position_m=reference.corner_positions[cell, corner].tolist(),
            energy_J=energy,
            unchanged_traction_sample_sha256=patch_hashes,
        )
        np.savez_compressed(
            output / "reference_field.npz",
            displacement=displacement,
            force_N=load,
            rhs_N=rhs,
            tensor_Pa=tensor,
            vm_Pa=vm,
        )
        np.savez_compressed(
            output / "probe_field.npz",
            points_m=points,
            tensor_Pa=probe_tensor,
            vm_Pa=probe_vm,
            displacement_m=probe_displacement,
        )
        (output / "snapshot.json").write_text(
            json.dumps({**snapshot, **row, "mesh_sha256": mesh_sha, "tetrahedra": reference.mesh.nelements}, indent=2)
            + "\n"
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "qualify_only": args.qualify_only,
        "mesh_sha256": mesh_sha,
        "parent_mesh_sha256": parent_sha,
        "tetrahedra": reference.mesh.nelements,
        "original_traction_samples_unchanged": True,
        "all_original_five_cases_qualified": args.qualify_only and len(rows) == 5,
        "rows": rows,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "The existing complete physical traction quadrature positions/weights/tractions are held byte-identical while nodal shape integration follows the refined bulk/trace basis. Unchanged physical domain/mass requires separate mesh partition evidence; no CAD geometry improvement is inferred. Qualification checks all five original nodal loads, full RHS/force budgets and complete corner/quadrature fields before finer use, without a redundant same-matrix re-solve. Finite quadrature itself remains a separately unresolved axis. Raw complete fields and all original36 probes are retained. These reference costs do not establish warmed native cost, physical/model/feedback convergence or full application acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
