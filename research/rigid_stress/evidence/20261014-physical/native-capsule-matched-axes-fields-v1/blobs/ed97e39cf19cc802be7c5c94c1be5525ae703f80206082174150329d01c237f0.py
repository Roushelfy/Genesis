"""Solve five unchanged finite-traction cases at one declared boundary quadrature on a fixed mesh."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_continuous_traction_load_axis import integrate
from .qualify_reference_scaling import AMGPinFactor
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import CASES, closed_source


def check_field(reference, displacement, force, rhs, tensor, vm, omega):
    np.testing.assert_allclose(reference.balanced_rhs(force, omega), rhs, rtol=1e-10, atol=1e-13)
    residual = float(np.linalg.norm(reference.stiffness @ displacement - rhs))
    budget = max(1e-11, 1e-8 * float(np.linalg.norm(rhs)))
    assert residual <= budget
    corners, rebuilt_vm = reference.corner_stress(displacement)
    np.testing.assert_allclose(corners, tensor, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(rebuilt_vm, vm, rtol=1e-10, atol=1e-6)
    gradient = reference.basis.interpolate(displacement).grad.transpose(2, 3, 0, 1)
    lam, mu = reference.lame
    quadrature = mu * (gradient + gradient.swapaxes(-1, -2))
    quadrature += lam * np.trace(gradient, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
    barycentric = np.vstack((1 - reference.basis.X.sum(axis=0), reference.basis.X))
    from_corners = np.einsum("cq,ecij->eqij", barycentric, tensor)
    np.testing.assert_allclose(quadrature, from_corners, rtol=1e-10, atol=1e-6)
    return {
        "complete_residual_N": residual,
        "original_budget_N": budget,
        "original_budget_fraction": residual / budget,
        "all_quadrature_tensor_components_checked": int(quadrature.size),
        "quadrature_vs_corner_maximum_error_Pa": float(abs(quadrature - from_corners).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--load-run", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--quadrature-level", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for run in [args.load_run, args.frozen_run, args.probe_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    retain(args.mesh)
    load_summary = args.load_run / "results/summary.json"
    retain(load_summary)
    loads = json.loads(load_summary.read_text())
    assert loads["fixed_mesh_sha256"] == inputs[str(args.mesh)]
    assert loads["fixed_patch_radius_m"] == 0.003 and len(loads["fixed_mesh_rows"]) == 15
    qualification = loads["original_five_case_continuous_law_and_full_nodal_loads_qualified"]
    assert len(qualification) == 5
    assert all(row["full_original_nodal_load_maximum_error_N"] == 0 for row in qualification)
    for module in ["finite_traction_reference", "audit_continuous_traction_load_axis"]:
        path = Path(f"research/rigid_stress/{module}.py")
        original = args.load_run / "source" / path
        retain(original)
        assert path.read_bytes() == original.read_bytes()
    with np.load(args.mesh) as saved:
        reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_rows, snapshots, points_by_case = [], {}, {}
    for case in CASES:
        snapshot_path = args.frozen_run / f"results/{case}/snapshot.json"
        retain(snapshot_path)
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["mesh_sha256"] == inputs[str(args.mesh)]
        assert snapshot["material"] == {"young_Pa": 1e10, "poisson": 0.3, "density_kg_m3": 2000}
        np.testing.assert_allclose(reference.total_mass, snapshot["mass_kg"], rtol=1e-12, atol=1e-18)
        old_field = args.frozen_run / f"results/{case}/reference_field.npz"
        retain(old_field)
        with np.load(old_field) as saved:
            record = check_field(
                reference,
                *[saved[name] for name in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]],
                snapshot["omega_rad_s"],
            )
        frozen_rows.append({"case": case, **record})
        probes = args.probe_run / f"results/{case}/reference_field.npz"
        retain(probes)
        with np.load(probes) as saved:
            assert np.all(saved["valid"]) and len(saved["points_m"]) == 36
            points_by_case[case] = saved["points_m"].copy()
        snapshots[case] = snapshot
        print(json.dumps({"phase": "original_frozen_field_qualification", **frozen_rows[-1]}), flush=True)
    (args.output / "original_frozen_field_qualification.json").write_text(json.dumps(frozen_rows, indent=2) + "\n")
    operator = (reference.stiffness[reference.free_dofs][:, reference.free_dofs] / reference.young).tocsr()
    modes = reference.rigid_modes[reference.free_dofs]
    np.random.seed(20261015)
    reference.factor = AMGPinFactor(operator, modes / np.linalg.norm(modes, axis=0))
    rows = []
    for case in CASES:
        snapshot = snapshots[case]
        load_path = args.load_run / f"results/{case}/q{args.quadrature_level}/full_nodal_load.npz"
        retain(load_path)
        with np.load(load_path) as saved:
            force = saved["force_N"].copy()
        independently_reintegrated, load_row = integrate(
            reference,
            reference.component_dofs,
            reference.coordinates,
            reference.basis.N,
            snapshot,
            args.quadrature_level,
            None,
        )
        np.testing.assert_allclose(independently_reintegrated, force, rtol=1e-11, atol=1e-13)
        solved = reference.solve(force, snapshot["omega_rad_s"])
        record = check_field(
            reference,
            solved.displacement,
            force,
            solved.rhs,
            solved.tensor_pa,
            solved.von_mises_pa,
            snapshot["omega_rad_s"],
        )
        points = points_by_case[case]
        tensor, vm = reference.stress_at_points(solved.displacement, points)
        displacement = (reference.basis.probes(points.T) @ solved.displacement).reshape(3, -1).T
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
        np.savez_compressed(
            output / "probe_field.npz", points_m=points, tensor_Pa=tensor, vm_Pa=vm, displacement_m=displacement
        )
        row = {
            "case": case,
            "quadrature_level": args.quadrature_level,
            "full_load_reintegration_maximum_error_N": float(abs(independently_reintegrated - force).max()),
            "raw_global_peak_Pa": float(solved.peak_pa),
            "raw_global_peak_position_m": solved.peak_position_m.tolist(),
            "energy_J": float(solved.energy_j),
            "original_common_probes": len(points),
            "amg_cg_iterations": len(reference.factor.residuals),
            **record,
        }
        (output / "snapshot.json").write_text(
            json.dumps(
                {
                    **snapshot,
                    **row,
                    "reference_peak_Pa": float(solved.peak_pa),
                    "reference_peak_position_m": solved.peak_position_m.tolist(),
                    "reference_energy_J": float(solved.energy_j),
                    "reference_quadrature_level": args.quadrature_level,
                    "continuous_traction_integrated_on_current_mesh": True,
                    "current_load_integration": load_row,
                },
                indent=2,
            )
            + "\n"
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "mesh_sha256": inputs[str(args.mesh)],
        "tetrahedra": reference.mesh.nelements,
        "quadrature_level": args.quadrature_level,
        "load_run": str(args.load_run),
        "frozen_run": str(args.frozen_run),
        "all_original_five_frozen_fields_qualified_before_new_solves": True,
        "rows": rows,
        "new_equilibrium_states_solved": len(rows),
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Only current-facet boundary quadrature changes for the same declared continuum traction law, original3mm physical radius/profile/strengths and fixed geometry/material/mass. Full original frozen fields pass unchanged force/corner/quadrature gates before new solves. Reintegrated full nodal loads must match the qualified load-only parent. Complete independent field auditing and adjacent integration-level/frozen-functional comparisons remain required; nodal-load agreement alone does not establish stress, spatial, geometry, solved-contact, application or warmed-cost accuracy.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
