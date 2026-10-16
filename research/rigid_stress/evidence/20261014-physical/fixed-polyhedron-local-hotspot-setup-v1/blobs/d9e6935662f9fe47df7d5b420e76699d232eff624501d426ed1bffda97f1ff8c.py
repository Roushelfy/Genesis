"""Qualify the original continuum traction law, then compare boundary integration on one fixed finer mesh without solving elasticity."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from skfem import ElementTetP2, ElementVector, FacetBasis, MeshTet

from .finite_traction_reference import assemble_finite_patch, physical_grip_patches
from .summarize_fine_references import CASES, closed_source


def trace_reference(mesh_path):
    with np.load(mesh_path) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
    element = ElementVector(ElementTetP2())
    # A single facet supplies global P2 DOF identities without assembling a
    # stiffness/mass matrix or allocating whole-volume gradient arrays.
    basis = FacetBasis(mesh, element, facets=mesh.boundary_facets()[:1], intorder=4)
    component_dofs = np.concatenate((basis.dofs.nodal_dofs, basis.dofs.edge_dofs), axis=1).T
    positions = np.concatenate((mesh.p.T, mesh.p.T[mesh.edges.T].mean(axis=1)))
    assert component_dofs.shape == positions.shape and np.unique(component_dofs).size == basis.N
    return SimpleNamespace(mesh=mesh, element=element), component_dofs, positions, basis.N


def integrate(reference, component_dofs, positions, count, snapshot, level, output):
    load = np.zeros(count)
    force = np.zeros(3)
    moment = np.zeros(3)
    com = np.array(snapshot["com_m"])
    patch_rows = []
    specs = physical_grip_patches("egg", snapshot["case"], radius_m=snapshot["patch_radius_m"])
    assert len(specs) == len(snapshot["patches"])
    for index, spec in enumerate(specs):
        patch = assemble_finite_patch(reference, spec, level)
        weighted = patch.tractions_pa * patch.area_weights_m2[:, None]
        force += weighted.sum(axis=0)
        moment += np.cross(patch.positions_m - com, weighted).sum(axis=0)
        nodal = patch.load_vector_n[component_dofs]
        np.testing.assert_allclose(nodal.sum(axis=0), weighted.sum(axis=0), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            np.cross(positions - com, nodal).sum(axis=0),
            np.cross(patch.positions_m - com, weighted).sum(axis=0),
            rtol=1e-11,
            atol=1e-12,
        )
        maximum_work_error = 0.0
        for component in range(3):
            for first in range(3):
                for second in range(first, 3):
                    actual = nodal[:, component] @ (positions[:, first] * positions[:, second])
                    expected = np.sum(
                        weighted[:, component] * patch.positions_m[:, first] * patch.positions_m[:, second]
                    )
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
                    maximum_work_error = max(maximum_work_error, float(abs(actual - expected)))
        load += patch.load_vector_n
        patch_rows.append(
            {
                "name": spec.name,
                "active_quadrature_samples": len(weighted),
                "profile_integral_m2": float(patch.profile_integral_m2),
                "force_N": weighted.sum(axis=0).tolist(),
                "moment_about_original_com_Nm": np.cross(patch.positions_m - com, weighted).sum(axis=0).tolist(),
                "maximum_quadratic_virtual_work_error_N_m2": maximum_work_error,
            }
        )
        if output is not None:
            np.savez_compressed(
                output / f"patch{index}.npz",
                positions_m=patch.positions_m,
                normals=patch.normals,
                area_weights_m2=patch.area_weights_m2,
                tractions_Pa=patch.tractions_pa,
                boundary_facets=patch.boundary_facets,
            )
    return load, {
        "quadrature_level": level,
        "force_N": force.tolist(),
        "moment_about_original_com_Nm": moment.tolist(),
        "patches": patch_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mesh", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--fine-mesh", type=Path, required=True)
    parser.add_argument("--fine-run", type=Path, required=True)
    parser.add_argument("--quadrature-levels", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (
        args.quadrature_levels == sorted(set(args.quadrature_levels))
        and 0 <= min(args.quadrature_levels) < max(args.quadrature_levels) <= 4
    )
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return digest

    for run in [args.base_run, args.fine_run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    module = Path("research/rigid_stress/finite_traction_reference.py")
    original_source = args.base_run / "source" / module
    assert retain(original_source) == hashlib.sha256(module.read_bytes()).hexdigest()
    base_hash, fine_hash = retain(args.base_mesh), retain(args.fine_mesh)
    original, dofs, coordinates, count = trace_reference(args.base_mesh)
    qualification = []
    snapshots = {}
    for case in CASES:
        path = args.base_run / f"results/{case}/snapshot.json"
        retain(path)
        snapshot = json.loads(path.read_text())
        assert snapshot["mesh_sha256"] == base_hash and snapshot["geometry"] == "egg"
        assert snapshot["case"] == case and snapshot["patch_radius_m"] == 0.003
        field = args.base_run / f"results/{case}/reference_field.npz"
        retain(field)
        with np.load(field) as saved:
            expected = saved["force_N"].copy()
        actual, record = integrate(
            original, dofs, coordinates, count, snapshot, snapshot["reference_quadrature_level"], None
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
        qualification.append(
            {
                "case": case,
                "original_quadrature_level": snapshot["reference_quadrature_level"],
                "full_original_nodal_load_maximum_error_N": float(abs(actual - expected).max()),
                "full_original_nodal_load_relative_L2": float(
                    np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                ),
                **record,
            }
        )
        snapshots[case] = snapshot
        print(json.dumps({"phase": "original_load_qualification", **qualification[-1]}), flush=True)
    del original, dofs, coordinates
    reference, dofs, coordinates, count = trace_reference(args.fine_mesh)
    rows, changes, frozen = [], [], []
    for case in CASES:
        snapshot = snapshots[case]
        field = args.fine_run / f"results/{case}/reference_field.npz"
        retain(field)
        fine_snapshot = args.fine_run / f"results/{case}/snapshot.json"
        retain(fine_snapshot)
        assert json.loads(fine_snapshot.read_text())["mesh_sha256"] == fine_hash
        with np.load(field) as saved:
            frozen_load = saved["force_N"].copy()
        previous = None
        for level in args.quadrature_levels:
            output = args.output / f"{case}/q{level}"
            output.mkdir(parents=True)
            load, record = integrate(reference, dofs, coordinates, count, snapshot, level, output)
            assert load.shape == frozen_load.shape
            np.savez_compressed(output / "full_nodal_load.npz", force_N=load)
            rows.append(
                {
                    "case": case,
                    "fixed_mesh_sha256": fine_hash,
                    "fixed_mesh_tetrahedra": reference.mesh.nelements,
                    **record,
                }
            )
            frozen.append(
                {
                    "case": case,
                    "quadrature_level": level,
                    "absolute_full_nodal_load_difference_from_frozen_N": float(np.linalg.norm(load - frozen_load)),
                    "relative_full_nodal_load_difference_from_frozen": float(
                        np.linalg.norm(load - frozen_load) / np.linalg.norm(load)
                    ),
                }
            )
            if previous is not None:
                changes.append(
                    {
                        "case": case,
                        "from_quadrature_level": previous[0],
                        "to_quadrature_level": level,
                        "absolute_full_nodal_load_change_N": float(np.linalg.norm(load - previous[1])),
                        "relative_full_nodal_load_change": float(
                            np.linalg.norm(load - previous[1]) / np.linalg.norm(load)
                        ),
                    }
                )
            previous = level, load
            print(json.dumps(rows[-1]), flush=True)
    assert len(qualification) == 5 and len(rows) == 15 and len(changes) == 10
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "original_five_case_continuous_law_and_full_nodal_loads_qualified": qualification,
        "fixed_mesh_rows": rows,
        "quadrature_load_changes": changes,
        "frozen_sample_load_comparisons": frozen,
        "fixed_mesh_sha256": fine_hash,
        "fixed_patch_radius_m": 0.003,
        "new_equilibrium_states_solved": 0,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "This load-only boundary-integration study keeps the existing physical centres, radius, compact profile, strengths, friction and normal/shear law unchanged. It first reproduces every original coarse nodal force at the original quadrature level using the unchanged original law source. On identical finer geometry it integrates that law on each current facet at three declared levels, retaining full nodal vectors and all physical quadrature arrays plus independent force/moment and quadratic-work checks. Frozen original-sample loading and newly integrated continuous-profile loading define distinct numerical load functionals. No stiffness/mass matrix or elasticity solve is assembled here. Small load changes or preserved total wrench do not certify stress accuracy; complete independent force/RHS/tensor/probe/peak/energy solves are still required before using this study as a physical convergence result.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
