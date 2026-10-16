"""Independently compare nested fixed-domain/fixed-traction bulk fields at original force gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import CASES, closed_source, relative_change


def prolongation(coarse, fine, parents):
    assert parents.shape == (fine.mesh.nelements,)
    owners = np.full(fine.scalar_basis.N, -1, dtype=int)
    owners[fine.scalar_basis.element_dofs.reshape(-1)] = np.tile(parents, 10)
    assert np.all(owners >= 0)
    coefficients = np.empty((fine.scalar_basis.N, 10))
    for first in range(0, fine.scalar_basis.N, 2048):
        section = slice(first, first + 2048)
        cells = owners[section]
        local = coarse.scalar_basis.mapping.invF(fine.coordinates[section].T[:, :, None], tind=cells)
        barycentric = np.concatenate((1 - local.sum(axis=0, keepdims=True), local), axis=0)
        assert np.all(barycentric >= -1e-10) and np.all(barycentric <= 1 + 1e-10)
        for index in range(10):
            coefficients[section, index] = coarse.scalar_basis.elem.gbasis(
                coarse.scalar_basis.mapping, local, index, tind=cells
            )[0][:, 0]
    rows = np.repeat(np.arange(fine.scalar_basis.N), 10)
    columns = coarse.scalar_basis.element_dofs[:, owners].T.reshape(-1)
    matrix = sparse.csr_matrix(
        (coefficients.reshape(-1), (rows, columns)), shape=(fine.scalar_basis.N, coarse.scalar_basis.N)
    )
    matrix.eliminate_zeros()
    checks = [(np.ones(coarse.scalar_basis.N), np.ones(fine.scalar_basis.N))]
    checks.extend((coarse.coordinates[:, index], fine.coordinates[:, index]) for index in range(3))
    checks.extend(
        (coarse.coordinates[:, i] * coarse.coordinates[:, j], fine.coordinates[:, i] * fine.coordinates[:, j])
        for i in range(3)
        for j in range(i, 3)
    )
    errors = []
    for source, exact in checks:
        evaluated = matrix @ source
        np.testing.assert_allclose(evaluated, exact, rtol=1e-11, atol=1e-13)
        errors.append(float(np.max(abs(evaluated - exact))))
    return matrix, errors


def volume_squared(tensor, volume):
    """Exact integral of an affine tensor norm on each tetrahedron."""
    return float(
        np.sum(volume * (np.sum(tensor**2, axis=(1, 2, 3)) + np.sum(tensor.sum(axis=1) ** 2, axis=(1, 2))) / 20)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-mesh", type=Path, required=True)
    parser.add_argument("--base-probes", type=Path, required=True)
    parser.add_argument("--mesh-build", type=Path, required=True)
    parser.add_argument("--level-run", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for run in [args.base_run, args.base_probes, args.mesh_build, *args.level_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    retain(args.mesh_build / "results/summary.json")
    mesh_report = json.loads((args.mesh_build / "results/summary.json").read_text())
    assert len(mesh_report["rows"]) == 3
    base_hash = retain(args.base_mesh)
    assert all(row["seed_sha256"] == base_hash for row in mesh_report["rows"])
    meshes = [args.base_mesh, *[args.mesh_build / f"results/level{index}.npz" for index in [1, 2]]]
    runs = [args.base_run, *args.level_run]
    previous_reference, previous_fields = None, None
    rows, changes = [], []
    args.output.mkdir(parents=True, exist_ok=False)
    for level, (run, mesh) in enumerate(zip(runs, meshes)):
        retain(mesh)
        with np.load(mesh) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        volume = (
            abs(
                np.linalg.det(
                    reference.mesh.p.T[reference.mesh.t.T[:, 1:]] - reference.mesh.p.T[reference.mesh.t.T[:, :1]]
                )
            )
            / 6
        )
        assert np.all(volume > 0)
        if level:
            summary = run / "results/summary.json"
            retain(summary)
            qualified = json.loads(summary.read_text())
            assert not qualified["qualify_only"] and qualified["original_traction_samples_unchanged"]
            assert qualified["mesh_sha256"] == retain(mesh) and qualified["parent_mesh_sha256"] == base_hash
            path = args.mesh_build / f"results/level{level}.parents.npz"
            retain(path)
            with np.load(path) as saved:
                matrix, polynomial_errors = prolongation(previous_reference, reference, saved["parent_ids"])
        fields = {}
        for case in CASES:
            path = run / f"results/{case}/reference_field.npz"
            retain(path)
            snapshot = run / f"results/{case}/snapshot.json"
            retain(snapshot)
            config = json.loads(snapshot.read_text())
            assert config["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
            np.testing.assert_allclose(reference.total_mass, config["mass_kg"], rtol=1e-12, atol=1e-18)
            with np.load(path) as saved:
                u, force, rhs, tensor, vm = [
                    saved[key].copy() for key in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]
                ]
            reconstructed_rhs = reference.balanced_rhs(force, config["omega_rad_s"])
            np.testing.assert_allclose(reconstructed_rhs, rhs, rtol=1e-10, atol=1e-13)
            residual = reference.stiffness @ u - rhs
            norm, budget = float(np.linalg.norm(residual)), max(1e-11, 1e-8 * float(np.linalg.norm(rhs)))
            assert norm <= budget
            rebuilt, rebuilt_vm = reference.corner_stress(u)
            np.testing.assert_allclose(rebuilt, tensor, rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(rebuilt_vm, vm, rtol=1e-10, atol=1e-6)
            probe_path = (
                args.base_probes / f"results/{case}/reference_field.npz"
                if level == 0
                else run / f"results/{case}/probe_field.npz"
            )
            retain(probe_path)
            with np.load(probe_path) as saved:
                probe = {key: saved[key].copy() for key in ["points_m", "tensor_Pa", "vm_Pa", "displacement_m"]}
            assert len(probe["points_m"]) == 36
            probe_tensor, probe_vm = reference.stress_at_points(u, probe["points_m"])
            np.testing.assert_allclose(probe_tensor, probe["tensor_Pa"], rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(probe_vm, probe["vm_Pa"], rtol=1e-10, atol=1e-6)
            energy = float(0.5 * u @ (reference.stiffness @ u))
            row = {
                "level": level,
                "case": case,
                "tetrahedra": reference.mesh.nelements,
                "complete_residual_N": norm,
                "original_budget_N": budget,
                "raw_global_peak_Pa": float(vm.max()),
                "energy_J": energy,
                "exact_volume_tensor_L2_Pa_m1p5": np.sqrt(volume_squared(tensor, volume)),
            }
            rows.append(row)
            if level:
                old = previous_fields[case]
                np.testing.assert_array_equal(probe["points_m"], old["probe"]["points_m"])
                embedded = np.empty(reference.basis.N)
                for component in range(3):
                    embedded[reference.component_dofs[:, component]] = (
                        matrix @ old["u"][previous_reference.component_dofs[:, component]]
                    )
                embedded_tensor, _ = reference.corner_stress(embedded)
                delta = u - embedded
                squared_energy_error = float(delta @ (reference.stiffness @ delta))
                embedded_energy = float(0.5 * embedded @ (reference.stiffness @ embedded))
                np.testing.assert_allclose(embedded_energy, old["energy"], rtol=1e-8, atol=1e-20)
                force_work_difference = float(embedded @ rhs - old["u"] @ old["rhs"])
                np.testing.assert_allclose(force_work_difference, 0, rtol=0, atol=max(1e-20, 1e-8 * abs(old["energy"])))
                bound = 2 * abs(embedded_energy - old["energy"]) + 2 * abs(force_work_difference)
                bound += (
                    2 * float(np.linalg.norm(embedded)) * norm
                    + 2 * float(np.linalg.norm(old["u"])) * old["residual_norm"]
                )
                bound += 256 * np.finfo(float).eps * (abs(energy) + abs(old["energy"]) + abs(squared_energy_error))
                identity_difference = squared_energy_error - 2 * (energy - old["energy"])
                assert abs(identity_difference) <= bound and energy - old["energy"] >= -bound
                assert squared_energy_error >= -bound
                tensor_error = np.sqrt(volume_squared(tensor - embedded_tensor, volume))
                change = {
                    "from_level": level - 1,
                    "to_level": level,
                    "case": case,
                    "original_common_physical_probes": 36,
                    "exact_common_volume_tensor_absolute_L2_Pa_m1p5": tensor_error,
                    "exact_common_volume_tensor_relative_L2": tensor_error / row["exact_volume_tensor_L2_Pa_m1p5"],
                    "nested_energy_error_squared_J": squared_energy_error,
                    "nested_energy_identity_defect_J": identity_difference,
                    "nested_energy_identity_bound_J": bound,
                    "unchanged_force_functional_work_difference_J": force_work_difference,
                    "polynomial_prolongation_maximum_errors": polynomial_errors,
                    "raw_global_peak_relative_change": abs(row["raw_global_peak_Pa"] - old["peak"])
                    / row["raw_global_peak_Pa"],
                    "energy_relative_change": abs(energy - old["energy"]) / abs(energy),
                }
                for name in ["tensor_Pa", "vm_Pa", "displacement_m"]:
                    relative, absolute = relative_change(old["probe"][name], probe[name])
                    change[name] = {"relative_change": relative, "absolute_change": absolute}
                changes.append(change)
                np.savez_compressed(
                    args.output / f"pair{level - 1}{level}-{case}.npz",
                    embedded_coarse_displacement_m=embedded,
                    exact_fine_volume_m3=volume,
                    fine_minus_embedded_coarse_corner_tensor_Pa=tensor - embedded_tensor,
                )
            fields[case] = {
                "u": u,
                "rhs": rhs,
                "probe": probe,
                "energy": energy,
                "peak": row["raw_global_peak_Pa"],
                "residual_norm": norm,
            }
            print(json.dumps(row), flush=True)
        previous_reference, previous_fields = reference, fields
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "rows": rows,
        "adjacent_changes": changes,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "Independent reconstruction keeps original full force budgets, all36 physical probes and raw global maxima. Parent containment yields a nested P2 prolongation checked on every quadratic polynomial component. Common-volume affine tensor integrals are exact on the finer tetrahedra; nested energy identities include explicit residual-work bounds. These supplemental volume/energy norms do not replace the declared probe/peak criteria. Fixed quadrature and fixed polyhedral domain do not establish continuum geometry, contact integration, regularization/history, finite-footprint models, full deformable contact feedback or warmed native accuracy/cost acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
