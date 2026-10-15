"""Replay independently audited original-history loads through native Quadrants recovery."""

import argparse
import json
from pathlib import Path

import numpy as np

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
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.audit_run)
    assert int((args.audit_run / "returncode.txt").read_text()) == 0
    audit = json.loads((args.audit_run / "results/summary.json").read_text())
    available = set(audit["saved_snapshot_steps"])
    steps = (
        args.steps
        if args.steps is not None
        else sorted(available & {11, 80, 866, 869, 1440, 2000, 2640, audit["states_checked"]})
    )
    assert steps and set(steps) <= available
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.audit_run / "results/body_geometry.npz") as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
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
    for step in steps:
        with np.load(args.audit_run / f"results/step{step:05d}.npz") as saved:
            inputs = [
                ("full_inertia", saved["material_full_inertia_rhs_N"].copy(), np.zeros(3)),
                ("rigid_relief", saved["material_contact_force_N"].copy(), saved["omega_material_rad_s"].copy()),
            ]
        for mode, force, angular in inputs:
            reference = body.solve(force, angular)
            state.force.from_numpy(force[permutation].reshape(-1, 1, 3))
            omega.from_numpy(angular[None])
            model.recover(omega, state, options)
            assert bool(qd_to_numpy(state.valid, transpose=True)[0])
            displacement = np.empty(body.basis.N)
            displacement[permutation] = qd_to_numpy(state.displacement, transpose=True, copy=True)[0].reshape(-1)
            tensor, vm = body.corner_stress(displacement)
            raw = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
            np.testing.assert_allclose(raw, tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5)
            peak = float(qd_to_numpy(state.peak)[0])
            np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
            rhs = body.balanced_rhs(force, angular)
            residual = float(np.linalg.norm(body.stiffness @ displacement - rhs))
            budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
            assert residual <= budget
            delta = float(np.linalg.norm(tensor - reference.tensor_pa))
            norm = float(np.linalg.norm(reference.tensor_pa))
            assert delta <= 1e-8 * norm + 1e-3
            row = {
                "step": step,
                "mode": mode,
                "native_vs_independent_tensor_absolute_L2_Pa": delta,
                "native_vs_independent_tensor_relative_L2": delta / max(norm, 1e-30),
                "native_complete_budget_fraction": residual / budget,
                "native_raw_global_peak_Pa": peak,
                "independent_raw_global_peak_Pa": reference.peak_pa,
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"step{step:05d}-{mode}.npz",
                native_tensor_Pa=tensor,
                independent_tensor_Pa=reference.tensor_pa,
                native_displacement_m=displacement,
            )
    report = {
        "scope": __doc__,
        "audit_run": str(args.audit_run),
        "steps": steps,
        "rows": rows,
        "complete_original_330_macro_history": audit["complete_original_330_macro_history"],
        "goal_completion": False,
        "limitations": "This checks the actual native production solve on objective resolved nodal loads at explicit original-history times. Full-inertia and rigid-relief inputs are separate declared problems. It does not establish finite-footprint load-model accuracy, matching rigid/deformable feedback, physical spatial/contact/time convergence or a verified warmed accuracy/cost range.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"replays": len(rows), "steps": steps}))


if __name__ == "__main__":
    main()
