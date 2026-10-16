"""Freshly audit complete fixed-mesh traction-integration fields and retain original probes and raw peaks."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from .audit_continuous_traction_load_axis import integrate
from .audit_frozen_traction_bulk_axis import volume_squared
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import CASES, closed_source, relative_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--load-run", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--quadrature-run", type=Path, nargs=3, required=True)
    parser.add_argument("--norm-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for run in [args.load_run, args.frozen_run, args.norm_qualification, *args.quadrature_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    qualified_module = args.norm_qualification / "source/research/rigid_stress/audit_frozen_traction_bulk_axis.py"
    retain(qualified_module)
    assert inspect.getsource(volume_squared) in qualified_module.read_text()
    norm_summary = args.norm_qualification / "results/summary.json"
    retain(norm_summary)
    qualification = json.loads(norm_summary.read_text())
    assert qualification["qualification_passed"]
    retain(args.mesh)
    load_summary = args.load_run / "results/summary.json"
    retain(load_summary)
    loads = json.loads(load_summary.read_text())
    assert loads["fixed_mesh_sha256"] == inputs[str(args.mesh)]
    assert len(loads["original_five_case_continuous_law_and_full_nodal_loads_qualified"]) == 5
    for module in ["finite_traction_reference", "audit_continuous_traction_load_axis"]:
        path = Path(f"research/rigid_stress/{module}.py")
        original = args.load_run / "source" / path
        retain(original)
        assert path.read_bytes() == original.read_bytes()
    with np.load(args.mesh) as saved:
        reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    corners = reference.mesh.p.T[reference.mesh.t.T]
    volume = abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    assert np.all(volume > 0)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, adjacent, frozen_comparisons = [], [], []
    for level, run in enumerate([args.frozen_run, *args.quadrature_run]):
        summary_path = run / "results/summary.json"
        retain(summary_path)
        report = json.loads(summary_path.read_text())
        assert report["mesh_sha256"] == inputs[str(args.mesh)] and len(report["rows"]) == 5
        if level:
            assert report["quadrature_level"] == level + 1
            assert report["all_original_five_frozen_fields_qualified_before_new_solves"]
        for case in CASES:
            path = run / f"results/{case}/reference_field.npz"
            snapshot_path = run / f"results/{case}/snapshot.json"
            probe_path = run / f"results/{case}/probe_field.npz"
            for selected in [path, snapshot_path, probe_path]:
                retain(selected)
            snapshot = json.loads(snapshot_path.read_text())
            assert snapshot["mesh_sha256"] == inputs[str(args.mesh)]
            assert snapshot["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
            np.testing.assert_allclose(reference.total_mass, snapshot["mass_kg"], rtol=1e-12, atol=1e-18)
            with np.load(path) as saved:
                u, force, rhs, tensor, vm = [
                    saved[name].copy() for name in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]
                ]
            if level:
                regenerated, _ = integrate(
                    reference,
                    reference.component_dofs,
                    reference.coordinates,
                    reference.basis.N,
                    snapshot,
                    level + 1,
                    None,
                )
                np.testing.assert_allclose(regenerated, force, rtol=1e-11, atol=1e-13)
                full_load_error = float(abs(regenerated - force).max())
            else:
                full_load_error = None
            reconstructed_rhs = reference.balanced_rhs(force, snapshot["omega_rad_s"])
            np.testing.assert_allclose(reconstructed_rhs, rhs, rtol=1e-10, atol=1e-13)
            residual = float(np.linalg.norm(reference.stiffness @ u - rhs))
            budget = max(1e-11, 1e-8 * float(np.linalg.norm(rhs)))
            assert residual <= budget
            rebuilt, rebuilt_vm = reference.corner_stress(u)
            np.testing.assert_allclose(rebuilt, tensor, rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(rebuilt_vm, vm, rtol=1e-10, atol=1e-6)
            gradient = reference.basis.interpolate(u).grad.transpose(2, 3, 0, 1)
            lam, mu = reference.lame
            quadrature = mu * (gradient + gradient.swapaxes(-1, -2))
            quadrature += lam * np.trace(gradient, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
            barycentric = np.vstack((1 - reference.basis.X.sum(axis=0), reference.basis.X))
            expected_quadrature = np.einsum("cq,ecij->eqij", barycentric, tensor)
            np.testing.assert_allclose(quadrature, expected_quadrature, rtol=1e-10, atol=1e-6)
            with np.load(probe_path) as saved:
                probe = {name: saved[name].copy() for name in ["points_m", "tensor_Pa", "vm_Pa", "displacement_m"]}
            assert len(probe["points_m"]) == 36
            probe_tensor, probe_vm = reference.stress_at_points(u, probe["points_m"])
            np.testing.assert_allclose(probe_tensor, probe["tensor_Pa"], rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(probe_vm, probe["vm_Pa"], rtol=1e-10, atol=1e-6)
            probe_u = (reference.basis.probes(probe["points_m"].T) @ u).reshape(3, -1).T
            np.testing.assert_allclose(probe_u, probe["displacement_m"], rtol=1e-10, atol=1e-13)
            energy = float(0.5 * u @ (reference.stiffness @ u))
            np.testing.assert_allclose(energy, snapshot["reference_energy_J"], rtol=1e-10, atol=1e-20)
            cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
            row = {
                "case": case,
                "run": str(run),
                "quadrature_level_zero_means_frozen_original_samples": level + 1 if level else 0,
                "complete_residual_N": residual,
                "original_budget_N": budget,
                "original_budget_fraction": residual / budget,
                "raw_global_peak_Pa": float(vm.max()),
                "raw_global_peak_position_m": reference.corner_positions[cell, corner].tolist(),
                "energy_J": energy,
                "exact_volume_tensor_L2_Pa_m1p5": float(np.sqrt(volume_squared(tensor, volume))),
                "full_nodal_load_reintegration_maximum_error_N": full_load_error,
                "complete_quadrature_tensor_components_checked": int(quadrature.size),
                "quadrature_vs_corner_maximum_tensor_error_Pa": float(abs(quadrature - expected_quadrature).max()),
            }
            rows.append(row)
            if level:
                comparison_sources = [("frozen", args.frozen_run)]
                if level > 1:
                    comparison_sources.append(("adjacent", args.quadrature_run[level - 2]))
                for kind, old_run in comparison_sources:
                    old_path = old_run / f"results/{case}/reference_field.npz"
                    old_probes = old_run / f"results/{case}/probe_field.npz"
                    for selected in [old_path, old_probes]:
                        retain(selected)
                    with np.load(old_path) as saved:
                        old_tensor, old_u, old_force, old_vm = [
                            saved[name].copy() for name in ["tensor_Pa", "displacement", "force_N", "vm_Pa"]
                        ]
                    with np.load(old_probes) as saved:
                        np.testing.assert_array_equal(saved["points_m"], probe["points_m"])
                        probe_changes = {
                            name: dict(
                                zip(["relative_change", "absolute_change"], relative_change(saved[name], probe[name]))
                            )
                            for name in ["tensor_Pa", "vm_Pa", "displacement_m"]
                        }
                    old_energy = float(0.5 * old_u @ (reference.stiffness @ old_u))
                    absolute_tensor = float(np.sqrt(volume_squared(tensor - old_tensor, volume)))
                    change = {
                        "case": case,
                        "first_run": str(old_run),
                        "second_run": str(run),
                        "exact_volume_tensor_absolute_L2_Pa_m1p5": absolute_tensor,
                        "exact_volume_tensor_relative_L2": absolute_tensor / row["exact_volume_tensor_L2_Pa_m1p5"],
                        "full_nodal_force_absolute_change_N": float(np.linalg.norm(force - old_force)),
                        "full_nodal_force_relative_change": float(
                            np.linalg.norm(force - old_force) / np.linalg.norm(force)
                        ),
                        "raw_global_peak_absolute_change_Pa": float(abs(vm.max() - old_vm.max())),
                        "raw_global_peak_relative_change": float(abs(vm.max() - old_vm.max()) / vm.max()),
                        "energy_absolute_change_J": abs(energy - old_energy),
                        "energy_relative_change": abs(energy - old_energy) / abs(energy),
                        "original_common_physical_probe_changes": probe_changes,
                    }
                    (frozen_comparisons if kind == "frozen" else adjacent).append(change)
                    np.savez_compressed(
                        args.output / f"{kind}-q{level + 1}-{case}.npz",
                        exact_volume_m3=volume,
                        complete_corner_tensor_difference_Pa=tensor - old_tensor,
                        complete_displacement_difference_m=u - old_u,
                        complete_force_difference_N=force - old_force,
                    )
            print(json.dumps(row), flush=True)
    assert len(rows) == 20 and len(adjacent) == 10 and len(frozen_comparisons) == 15
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "mesh_sha256": inputs[str(args.mesh)],
        "rows": rows,
        "adjacent_quadrature_changes": adjacent,
        "frozen_sample_functional_changes": frozen_comparisons,
        "input_sha256": inputs,
        "new_equilibrium_states_solved": 0,
        "goal_completion": False,
        "limitations": "Twenty full fixed-mesh fields retain original complete force budgets, all corner/quadrature tensors, all36 physical probes, energy and raw whole-volume extrema. Exact volume norms use the manufactured-qualified affine-tensor formula. Fresh traction integration checks every new nodal force. Frozen-sample and current-facet continuum integration are separate numerical functionals; their measured stress difference is explicit. This audit isolates boundary integration at one geometry and does not establish spatial/geometry/contact/model/feedback/application convergence or warmed native cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
