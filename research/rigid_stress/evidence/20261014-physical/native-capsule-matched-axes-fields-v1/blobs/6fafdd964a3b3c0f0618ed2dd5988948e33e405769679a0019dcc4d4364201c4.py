"""Capture the unchanged first paired dynamic Newton equation from its saved initialization closure."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .franka_rigid_mode_coordinates import coordinate_map
from .precise_stencil_spatial import PreciseStencilSpatial
from .summarize_fine_references import closed_source
from .wide_spatial_franka import restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.source_run)
    source_code = int((args.source_run / "returncode.txt").read_text())
    assert source_code in [0, 1]
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for name in [
        "configuration.json",
        "closure_state.npz",
        "reference_geometry.npz",
        "contact_geometry.npz",
        "contact_node_map.npz",
    ]:
        retain(args.source_run / "results" / name)
    original_config = json.loads((args.source_run / "results/configuration.json").read_text())
    closure_config_path = Path(original_config["closure_run"]) / "results/configuration.json"
    retain(closure_config_path)
    capture = Path(json.loads(closure_config_path.read_text())["capture"])
    capture_report_path = capture / "results/summary.json"
    retain(capture_report_path)
    capture_report = json.loads(capture_report_path.read_text())
    selected_frame = [row for row in capture_report["captures"] if row["mode"] == "stationary_lift_hold"][-1]
    retain(capture / "results" / selected_frame["geometry_snapshot"])
    for geometry in capture_report["geometry"]:
        if geometry["link"] in {finger["link"] for finger in selected_frame["fingers"]}:
            retain(capture / "results" / geometry["mesh"])
    retain(capture / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz")
    static, config = restore_static(args.source_run)
    assert config["mode"] in ["cad", "polyhedron"] and config["cold_closure"]
    with np.load(args.source_run / "results/closure_state.npz") as saved:
        u = np.asarray(saved["displacement"], dtype=np.longdouble)
        np.testing.assert_array_equal(saved["fixed_points_m"], static.fixed_initializer_points_m)
        np.testing.assert_array_equal(saved["fixed_components"], static.fixed_initializer_components)
    assert len(u) == static.stiffness.shape[0]
    system = PreciseStencilSpatial(static, config["dt"])
    system.q_target[:] = 0.018
    # Original first dynamic step starts with zero velocity and predicts u itself.
    system.predicted_displacement = u.copy()
    mapping = coordinate_map(system)
    gravity = system.gravity_force([0, 0, -9.81])
    print(json.dumps({"body_tetrahedra": system.body.mesh.nelements, "first_dynamic_state_restored": True}), flush=True)
    _energy, gradient, matrix, _normal, contact = system.evaluate_updated_friction(u, u, gravity, True)
    reduced_gradient = system.reduced_gradient(gradient, np.arange(len(u)), mapping)
    reduced = system.reduced_hessian(matrix, np.arange(len(u)), mapping).tocsc()
    assert reduced.shape == (mapping.shape[1],) * 2
    body_dofs = system.body.basis.N
    assert reduced.shape[0] == body_dofs + 2
    border = np.arange(body_dofs - 6, body_dofs + 2)
    graph = reduced.tocsr(copy=True)
    graph.eliminate_zeros()
    graph.data = np.ones(graph.nnz, dtype=np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    degree = np.diff(graph.indptr)
    interior = np.arange(body_dofs - 6)
    largest = np.argsort(degree)[-12:][::-1]
    args.output.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(args.output / "first_dynamic_matrix.npz", reduced)
    sparse.save_npz(args.output / "original_coordinate_map.npz", mapping)
    np.savez_compressed(
        args.output / "first_dynamic_state.npz",
        displacement=u,
        previous_displacement=u,
        predicted_displacement=u,
        applied_force_N=gravity,
        full_gradient_N=gradient,
        reduced_rhs_N=-reduced_gradient,
        contact_force_N=contact,
        explicit_rigid_and_jaw_columns=border,
    )
    residual = float(np.linalg.norm(reduced_gradient))
    if source_code == 1:
        log = args.source_run / "reference.log"
        retain(log)
        lines = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
        starts = [
            row["updated_residual_N"] for row in lines if row.get("step") == 1 and row.get("newton_iteration") == 0
        ]
        assert starts
        np.testing.assert_allclose(residual, starts[-1], rtol=1e-10, atol=1e-12)
        restored_residual_difference = abs(residual - starts[-1])
    else:
        restored_residual_difference = None
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "source_run": str(args.source_run),
        "actual_source_wrapper_code": source_code,
        "captured_newton_iteration": 0,
        "physical_step": 1,
        "dt_s": config["dt"],
        "body_tetrahedra": int(system.body.mesh.nelements),
        "body_dofs": int(body_dofs),
        "matrix_dimension": int(reduced.shape[0]),
        "matrix_nonzeros": int(reduced.nnz),
        "symmetric_graph_nonzeros": int(graph.nnz),
        "explicit_six_rigid_plus_two_jaw_columns": border.tolist(),
        "border_degrees": degree[border].tolist(),
        "interior_maximum_degree": int(degree[interior].max()),
        "largest_degree_columns": [
            {"column": int(index), "degree": int(degree[index]), "is_declared_border": bool(index in border)}
            for index in largest
        ],
        "first_original_complete_mapped_residual_N": residual,
        "difference_from_retained_original_first_residual_N": restored_residual_difference,
        "matrix_sha256": hashlib.sha256((args.output / "first_dynamic_matrix.npz").read_bytes()).hexdigest(),
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "new_physical_states_accepted": 0,
        "factorization_attempted": False,
        "goal_completion": False,
        "limitations": "Diagnostic reassembly starts from the exact saved closure with original geometry/maps, zero initial velocity, actuation, gravity, dt and current-friction equations. No initialization/load step is added or replaced, no matrix entry is dropped and no factorization/physical state is accepted. Source closure and the failed run's retained first-residual value are checked; a matrix capture alone does not qualify a new ordering, prove a memory root cause or establish physical convergence.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}), flush=True)


if __name__ == "__main__":
    main()
