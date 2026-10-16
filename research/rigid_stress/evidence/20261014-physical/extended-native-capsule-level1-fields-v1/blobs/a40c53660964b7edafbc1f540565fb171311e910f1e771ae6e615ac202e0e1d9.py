"""Replay independently audited rigid/deformable body loads through actual native recovery."""

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
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.audit_run)
    assert int((args.audit_run / "returncode.txt").read_text()) == 0
    audited = json.loads((args.audit_run / "results/summary.json").read_text())
    assert len(audited["cases"]) == 2
    assert all(max(args.steps) <= c["steps_checked"] for c in audited["cases"])
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.audit_run / "results/case00/step0001.npz") as saved:
        vertices, tetrahedra = saved["body_mesh_vertices"], saved["body_mesh_tetrahedra"]
    body = SkfemElasticReference(vertices, tetrahedra, 1e10, 0.3, 2000)
    mesh = args.output / "body_mesh.npz"
    np.savez_compressed(
        mesh,
        vertices=vertices,
        tetrahedra=tetrahedra,
        surface_triangles=body.mesh.facets[:, body.mesh.boundary_facets()].T,
    )
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=mesh, method="pcg", output_mode="full", max_iterations=20000)
    model = StressModel(options)
    state = model.create_state(1, "full")
    permutation = body.native_dof_permutation(qd_to_numpy(model.info.vertices, transpose=True, copy=True))
    omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
    rows = []
    for case in range(2):
        for step in args.steps:
            with np.load(args.audit_run / f"results/case{case:02d}/step{step:04d}.npz") as saved:
                np.testing.assert_array_equal(saved["body_mesh_vertices"], vertices)
                np.testing.assert_array_equal(saved["body_mesh_tetrahedra"], tetrahedra)
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
                np.testing.assert_allclose(
                    raw, tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]], rtol=1e-9, atol=1e-5
                )
                peak = float(qd_to_numpy(state.peak)[0])
                np.testing.assert_allclose(peak, vm.max(), rtol=1e-4, atol=1e-3)
                rhs = body.balanced_rhs(force, angular)
                residual = float(np.linalg.norm(body.stiffness @ displacement - rhs))
                budget = max(options.absolute_tolerance, options.tolerance * np.linalg.norm(rhs))
                assert residual <= budget
                error = float(np.linalg.norm(tensor - reference.tensor_pa) / np.linalg.norm(reference.tensor_pa))
                assert error < 1e-8
                rows.append(
                    {
                        "case": case,
                        "step": step,
                        "mode": mode,
                        "native_vs_independent_full_tensor_relative_error": error,
                        "native_complete_budget_fraction": residual / budget,
                        "native_raw_global_peak_Pa": peak,
                        "reference_raw_global_peak_Pa": float(reference.von_mises_pa.max()),
                    }
                )
                np.savez_compressed(
                    args.output / f"case{case:02d}-step{step:04d}-{mode}.npz",
                    native_displacement_m=displacement,
                    native_corner_tensor_Pa=tensor,
                    raw_native_tensor_Pa=raw,
                    native_vm_Pa=vm,
                    independent_displacement_m=reference.displacement,
                    independent_tensor_Pa=reference.tensor_pa,
                    input_force_N=force,
                    omega_material_rad_s=angular,
                    balanced_rhs_N=rhs,
                )
    report = {
        "scope": __doc__,
        "audit_run": str(args.audit_run),
        "backend": args.backend,
        "rows": rows,
        "goal_completion": False,
        "limitations": "Actual native nodal replay isolates elastic algebra and compares full-inertia versus rigid-relief inputs at declared sampled times. It does not qualify finite-footprint mapping, contact/spatial/regularization convergence, original approach/angular history or physically verified warmed cost. Rigid/deformable feedback comes from independently solved corresponding contact histories, not native algebra agreement.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "replays": len(rows),
                "maximum_full_tensor_relative_error": max(
                    r["native_vs_independent_full_tensor_relative_error"] for r in rows
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
