"""Check native numerical recovery separately from matched nonlinear-reference model differences."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from skfem.helpers import sym_grad
from skfem.models.elasticity import linear_stress

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--case", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.audit_run)
    assert int((args.audit_run / "returncode.txt").read_text()) == 0
    summary_path = args.audit_run / "results/summary.json"
    audit = json.loads(summary_path.read_text())
    audited = audit["rows"][args.case]
    snapshot_path = args.audit_run / f"results/case{args.case:02d}.npz"
    with np.load(snapshot_path) as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        full_rhs = saved["material_full_inertia_rhs_N"].copy()
        contact_rhs = saved["material_contact_force_N"].copy()
        angular = saved["omega_material_rad_s"].copy()
        audited_tensor = saved["independent_resolved_linear_tensor_Pa"].copy()
        nonlinear_tensor = saved["objective_nonlinear_quadrature_tensor_Pa"].copy()
    assert body.mesh.nelements == audited["tetrahedra"]
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = args.output / "body_mesh.npz"
    np.savez_compressed(
        mesh,
        vertices=body.mesh.p.T,
        tetrahedra=body.mesh.t.T,
        surface_triangles=body.mesh.facets[:, body.mesh.boundary_facets()].T,
    )
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=mesh, method="pcg", output_mode="full", max_iterations=20000)
    model = StressModel(options)
    state = model.create_state(1, "full")
    permutation = body.native_dof_permutation(qd_to_numpy(model.info.vertices, transpose=True, copy=True))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    rows = []
    for mode, force, rotation_rate in [("full_inertia", full_rhs, np.zeros(3)), ("rigid_relief", contact_rhs, angular)]:
        reference = body.solve(force, rotation_rate)
        if mode == "full_inertia":
            np.testing.assert_allclose(reference.tensor_pa, audited_tensor, rtol=1e-10, atol=1e-4)
        state.force.from_numpy(force[permutation].reshape(-1, 1, 3))
        omega.from_numpy(rotation_rate[None])
        model.recover(omega, state, options)
        displacement = np.empty(body.basis.N)
        displacement[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
        tensor, vm = body.corner_stress(displacement)
        raw = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
        peak = float(qd_to_numpy(state.peak)[0])
        native_valid = bool(qd_to_numpy(state.valid, transpose=True)[0])
        native_residual = float(np.sqrt(qd_to_numpy(state.residual_norm_squared)[0]))
        rhs = body.balanced_rhs(force, rotation_rate)
        residual = float(np.linalg.norm(body.stiffness @ displacement - rhs))
        budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
        delta = float(np.linalg.norm(tensor - reference.tensor_pa))
        norm = float(np.linalg.norm(reference.tensor_pa))
        quadrature = linear_stress(*body.lame)(sym_grad(body.basis.interpolate(displacement))).transpose(2, 3, 0, 1)
        nonlinear_norm = np.sqrt(np.sum(body.basis.dx[..., None, None] * nonlinear_tensor**2))
        model_delta = np.sqrt(np.sum(body.basis.dx[..., None, None] * (quadrature - nonlinear_tensor) ** 2))
        row = {
            "case": args.case,
            "tetrahedra": body.mesh.nelements,
            "mode": mode,
            "time_s": audited["time_s"],
            "native_valid_including_six_gauge_rows": native_valid,
            "native_complete_residual_N": native_residual,
            "independent_complete_residual_N": residual,
            "independent_complete_budget_fraction": residual / budget,
            "native_vs_independent_corner_tensor_relative_L2": delta / max(norm, 1e-30),
            "native_vs_nonlinear_quadrature_tensor_relative_volume_L2": float(model_delta / nonlinear_norm),
            "native_raw_global_peak_Pa": peak,
            "independent_raw_global_peak_Pa": reference.peak_pa,
            "nonlinear_raw_global_peak_lower_Pa": audited["raw_nonlinear_global_peak_lower_Pa"],
            "raw_peak_native_vs_nonlinear_relative_difference": abs(
                peak / audited["raw_nonlinear_global_peak_lower_Pa"] - 1
            ),
            "iterations": int(qd_to_numpy(state.iterations)[0]),
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"{mode}.npz",
            native_corner_tensor_Pa=tensor,
            independent_corner_tensor_Pa=reference.tensor_pa,
            native_displacement_m=displacement,
            native_quadrature_tensor_Pa=quadrature,
            objective_nonlinear_quadrature_tensor_Pa=nonlinear_tensor,
        )
        # Preserve complete diagnostics even if this graded mesh is outside the
        # current native iterative solver's qualified numerical range.
        (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        assert native_valid and residual <= budget, row
        np.testing.assert_allclose(raw, tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5)
        np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
        assert delta <= 1e-8 * norm + 1e-3, row
    report = {
        "scope": __doc__,
        "audit_run": str(args.audit_run),
        "rows": rows,
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [summary_path, snapshot_path]
        },
        "goal_completion": False,
        "limitations": "One matched32.5ms state per mesh at unresolved physical spatial/contact settings. This is actual production Quadrants recovery of independently resolved nodal loads. Same-discretization linear numerical error and finite-geometry/constitutive model discrepancy are reported separately. Full-inertia and rigid-relief loads define separate problems; neither establishes actual rigid/deformable feedback, finite-footprint load-model accuracy, physical convergence or warmed cost at verified accuracy.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": args.case, "replays": len(rows)}))


if __name__ == "__main__":
    main()
