"""Fixed two-point load control: raw extrema, compliance and localized test-work bounds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-run", type=Path, required=True)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for run in [args.mesh_run, args.contact_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    config = json.loads((args.contact_run / "results/configuration.json").read_text())
    with np.load(args.contact_run / "results/reference_geometry.npz") as saved:
        seed = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    field_path = args.contact_run / f"results/step{config['steps']:04d}.npz"
    with np.load(field_path) as saved:
        force = saved["contact_force_N"][: seed.basis.N].reshape(-1, 3)
        permutation = saved["body_component_permutation"]
    active = np.flatnonzero(np.linalg.norm(force, axis=1) > 1e-8)
    assert len(active) == 2
    rest_nodes = seed.coordinates[seed.scalar_basis.dofs.nodal_dofs[0]]
    # Recover the exact interleaved collision-node ordering from the archived
    # component permutation, without assuming it matches scikit-fem node order.
    collision_coordinates = seed.basis.doflocs[:, permutation[0::3]].T
    anchors, point_forces = collision_coordinates[active], force[active]
    discarded = force.copy()
    discarded[active] = 0
    assert np.linalg.norm(discarded) < 1e-12
    assert cKDTree(rest_nodes).query(anchors)[0].max() < 1e-12
    with np.load(args.probes) as saved:
        points = saved["reference_points_m"].copy()
        weights = saved["volume_weights_m3"].copy()
    rows, observed, hashes = [], [], {}
    for level in args.levels:
        mesh_path = args.mesh_run / f"results/mesh{level:02d}.npz"
        with np.load(mesh_path) as saved:
            body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        np.testing.assert_allclose(body.total_mass, seed.total_mass, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(body.com, seed.com, rtol=0, atol=1e-13)
        distance, nodes = cKDTree(body.coordinates).query(anchors)
        assert distance.max() < 1e-12
        load = np.zeros(body.basis.N)
        load[body.component_dofs[nodes]] = point_forces
        solution = body.solve(load)
        assert solution.residual_budget_fraction <= 1
        assert solution.residual_n <= 1e-9
        np.testing.assert_allclose(body.rigid_modes.T @ solution.rhs, np.zeros(6), rtol=0, atol=1e-12)
        np.testing.assert_allclose(solution.rhs @ solution.displacement, 2 * solution.energy_j, rtol=1e-8, atol=1e-15)
        tests = np.zeros((body.basis.N, len(nodes)))
        for index, node in enumerate(nodes):
            tests[body.component_dofs[node], index] = point_forces[index] / np.linalg.norm(point_forces[index])
            tests[:, index] = body.align_rigid_gauge(tests[:, index])
        gram = tests.T @ (body.stiffness @ tests)
        work = tests.T @ solution.rhs
        lower_bound = float(work @ np.linalg.solve(gram, work))
        compliance = 2 * solution.energy_j
        assert lower_bound <= compliance * (1 + 1e-8) + 1e-16
        elements = body.mesh.element_finder()(*points.T)
        local = body.basis.mapping.invF(points.T[:, :, None], tind=elements)[:, :, 0]
        barycentric = np.column_stack((1 - local.sum(axis=0), local.T))
        assert barycentric.min() > 1e-8
        tensor, vm, values = body.stress_at_points(solution.displacement, points)
        observed.append({"tensor": tensor, "vm": vm, "displacement": values})
        row = {
            "level": level,
            "tetrahedra": body.mesh.nelements,
            "body_dofs": body.basis.N,
            "complete_residual_N": solution.residual_n,
            "original_budget_fraction": solution.residual_budget_fraction,
            "raw_global_peak_Pa": float(solution.peak_pa),
            "raw_global_peak_position_m": solution.peak_position_m.tolist(),
            "elastic_energy_J": solution.energy_j,
            "compliance_J": compliance,
            "localized_test_stiffness_gram_N_per_m": gram.tolist(),
            "localized_test_load_work_N": work.tolist(),
            "localized_test_compliance_lower_bound_J": lower_bound,
            "minimum_probe_barycentric_margin": float(barycentric.min()),
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"level{level:02d}.npz",
            body_displacement_m=solution.displacement,
            raw_corner_tensor_Pa=solution.tensor_pa,
            raw_corner_vm_Pa=solution.von_mises_pa,
            point_force_N=load,
            balanced_rhs_N=solution.rhs,
            localized_tests=tests,
            probe_points_m=points,
            probe_tensor_Pa=tensor,
            probe_vm_Pa=vm,
            probe_displacement_m=values,
        )
        hashes[str(mesh_path)] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        print(json.dumps(row), flush=True)
    changes = []
    for index in range(1, len(observed)):
        old, new = observed[index - 1]["tensor"], observed[index]["tensor"]
        norm = lambda tensor: np.sqrt(np.sum(weights * np.sum(tensor * tensor, axis=(-2, -1))))
        changes.append(
            {
                "coarse_level": args.levels[index - 1],
                "fine_level": args.levels[index],
                "probe_tensor_relative_change": float(norm(old - new) / norm(new)),
                "raw_peak_relative_change": abs(
                    rows[index - 1]["raw_global_peak_Pa"] / rows[index]["raw_global_peak_Pa"] - 1
                ),
                "energy_relative_change": abs(
                    rows[index - 1]["elastic_energy_J"] / rows[index]["elastic_energy_J"] - 1
                ),
            }
        )
    for path in [field_path, args.probes]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "scope": __doc__,
        "anchors_m": anchors.tolist(),
        "fixed_point_forces_N": point_forces.tolist(),
        "common_probes": len(points),
        "rows": rows,
        "adjacent_changes": changes,
        "input_sha256": hashes,
        "goal_completion": False,
        "limitations": "Explicit concentrated loads, with unchanged force vectors and material/reference anchors across meshes; consistent free-body inertia relief is part of this declared linear control. No finite traction footprint is imposed. Localized virtual test work provides a compliance lower bound, not a fitted singular exponent. This is a point-load diagnostic, not resolved continuum-contact convergence or an accepted finite-area egg/Franka physical model. Global raw maxima remain unsmoothed.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
