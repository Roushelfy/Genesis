"""Test original shared U initialization coordinates on a remeshed body without moving any anchor."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

from .sample_load_fields import locate_probes
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--qualified-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.qualified_run)
    assert int((args.qualified_run / "returncode.txt").read_text()) == 0
    inputs = {}

    def retain(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for path in [args.mesh, args.anchors, args.qualified_run / "solution/initialization_anchors.json"]:
        retain(path)
    original = json.loads((args.qualified_run / "solution/initialization_anchors.json").read_text())
    assert original["sha256"] == inputs[str(args.anchors)]
    with np.load(args.anchors) as saved:
        points, components = saved["positions_m"].copy(), saved["components"].copy()
    np.testing.assert_array_equal(points, original["positions_m"])
    np.testing.assert_array_equal(components, original["components"])
    assert points.shape == (6, 3) and components.shape == (6,)
    with np.load(args.mesh) as saved:
        mesh = MeshTet(saved["vertices"].T, saved["tetrahedra"].T, sort_t=False)
    scalar = Basis(mesh, ElementTetP2(), intorder=4)
    vector = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
    distance, nearest = cKDTree(scalar.doflocs.T).query(points)
    elements, valid, barycentric = locate_probes(mesh, points)
    rows = []
    for index in range(6):
        rows.append(
            {
                "original_point_m": points[index].tolist(),
                "component": int(components[index]),
                "nearest_actual_P2_node_distance_m": float(distance[index]),
                "nearest_actual_P2_node_m": scalar.doflocs[:, nearest[index]].tolist(),
                "inside_original_new_mesh": bool(valid[index]),
                "element": int(elements[index]),
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "anchor_locations.npz",
        positions_m=points,
        components=components,
        valid=valid,
        elements=elements,
        inside_barycentric=barycentric,
    )
    manufactured_error, rank, condition = None, None, None
    if valid.all():
        evaluation = vector.probes(points.T).tocsr()
        constraints = evaluation[components * len(points) + np.arange(len(points))].tocsr()
        component_dofs = np.stack(vector.split_indices(), axis=-1)
        affine = np.array([[2.0, 7, 13], [-3, 5, -11], [17, -19, 23]])
        shift = np.array([0.2, -0.3, 0.5])
        values = scalar.doflocs.T @ affine.T + shift
        manufactured = np.empty(vector.N)
        manufactured[component_dofs] = values
        expected = (points @ affine.T + shift)[np.arange(len(points)), components]
        actual = constraints @ manufactured
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        manufactured_error = float(abs(actual - expected).max())
        modes = np.zeros((vector.N, 6))
        modes[component_dofs, np.arange(3)[None]] = 1
        relative = scalar.doflocs.T - points.mean(axis=0)
        for axis in range(3):
            modes[component_dofs, axis + 3] = np.cross(np.eye(3)[axis], relative)
        imposed = constraints @ modes
        scaled = imposed / np.linalg.norm(imposed, axis=0)
        rank, condition = int(np.linalg.matrix_rank(scaled)), float(np.linalg.cond(scaled))
        assert rank == 6
        sparse.save_npz(args.output / "exact_physical_coordinate_constraints.npz", constraints)
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "rows": rows,
        "input_sha256": inputs,
        "all_original_anchors_inside_new_polyhedral_mesh": bool(valid.all()),
        "all_original_anchors_are_existing_P2_nodes_at_original_1e_minus12_m_check": bool(np.all(distance <= 1e-12)),
        "manufactured_affine_coordinate_maximum_error": manufactured_error,
        "interpolated_constraint_rigid_mode_rank": rank,
        "interpolated_constraint_scaled_condition": condition,
        "new_equilibrium_states_solved": 0,
        "goal_completion": False,
        "limitations": "Original physical coordinates and components remain byte-exact. Outside points are reported without snapping, extrapolating, tolerance relaxation or load changes. Exact P2 coordinate rows, when geometrically valid, are manufactured algebra controls only. Their use in nonlinear initialization, sparse coordinate elimination, force/source/history audits and old-node-case reproduction remains unimplemented and unverified here. No new physical equilibrium, shared fourth-level history or convergence is accepted.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
