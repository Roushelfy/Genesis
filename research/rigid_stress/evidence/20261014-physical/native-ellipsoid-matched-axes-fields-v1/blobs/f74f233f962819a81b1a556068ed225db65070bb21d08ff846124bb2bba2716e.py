"""Audit retained first-dynamic matrices and states after a capture metadata serialization failure."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.capture_run
    closed_source(run)
    assert int((run / "returncode.txt").read_text()) == 1
    log_path = run / "capture.log"
    log = log_path.read_text()
    assert "TypeError: Object of type int32 is not JSON serializable" in log
    parent = Path((run / "parent-source-path.txt").read_text().strip())
    closed_source(parent)
    assert (run / "parent-source.sha256").read_bytes() == (parent / "source.sha256").read_bytes()
    for line in (parent / "source.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / relative).read_bytes()).hexdigest() == digest
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
        return digest

    for path in [log_path, run / "parent-source-path.txt", run / "parent-source.sha256"]:
        retain(path)
    for name in ["first_dynamic_matrix.npz", "original_coordinate_map.npz", "first_dynamic_state.npz"]:
        retain(run / "results" / name)
    config_path = parent / "results/configuration.json"
    retain(config_path)
    config = json.loads(config_path.read_text())
    closure_path, geometry_path = [
        parent / "results" / name for name in ["closure_state.npz", "reference_geometry.npz"]
    ]
    retain(closure_path)
    retain(geometry_path)
    matrix = sparse.load_npz(run / "results/first_dynamic_matrix.npz")
    mapping = sparse.load_npz(run / "results/original_coordinate_map.npz")
    assert np.isfinite(matrix.data).all() and matrix.shape == (mapping.shape[1],) * 2
    with np.load(run / "results/first_dynamic_state.npz") as state, np.load(closure_path) as original:
        displacement = state["displacement"]
        np.testing.assert_array_equal(displacement, np.asarray(original["displacement"], dtype=displacement.dtype))
        np.testing.assert_array_equal(state["previous_displacement"], displacement)
        np.testing.assert_array_equal(state["predicted_displacement"], displacement)
        rhs = state["reduced_rhs_N"]
        np.testing.assert_allclose(mapping.T @ state["full_gradient_N"], -rhs, rtol=1e-12, atol=1e-15)
        border = state["explicit_rigid_and_jaw_columns"].copy()
    body_dofs = int(matrix.shape[0] - 2)
    np.testing.assert_array_equal(border, np.arange(body_dofs - 6, body_dofs + 2))
    graph = matrix.tocsr(copy=True)
    graph.eliminate_zeros()
    graph.data = np.ones(graph.nnz, dtype=np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.diff(graph.indptr)
    largest = np.argsort(degree)[-12:][::-1]
    with np.load(geometry_path) as saved:
        tetrahedra = len(saved["tetrahedra"])
    residual = float(np.linalg.norm(rhs))
    source_code = int((parent / "returncode.txt").read_text())
    if source_code:
        original_log = parent / "reference.log"
        retain(original_log)
        lines = [json.loads(line) for line in original_log.read_text().splitlines() if line.startswith("{")]
        first = [
            row["updated_residual_N"] for row in lines if row.get("step") == 1 and row.get("newton_iteration") == 0
        ][-1]
        np.testing.assert_allclose(residual, first, rtol=1e-10, atol=1e-12)
        difference = abs(residual - first)
    else:
        difference = None
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "capture_run": str(run),
        "capture_actual_wrapper_code": 1,
        "source_run": str(parent),
        "source_actual_wrapper_code": source_code,
        "body_tetrahedra": tetrahedra,
        "body_dofs": body_dofs,
        "matrix_dimension": int(matrix.shape[0]),
        "matrix_nonzeros": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "matrix_index_dtype": str(matrix.indices.dtype),
        "symmetric_graph_nonzeros": int(graph.nnz),
        "explicit_six_rigid_plus_two_jaw_columns": border.tolist(),
        "border_degrees": degree[border].tolist(),
        "interior_maximum_degree": int(degree[: body_dofs - 6].max()),
        "largest_degree_columns": [
            {"column": int(index), "degree": int(degree[index]), "is_declared_border": bool(index in border)}
            for index in largest
        ],
        "first_original_complete_mapped_residual_N": residual,
        "difference_from_retained_original_first_residual_N": difference,
        "dt_s": config["dt"],
        "restored_closure_and_zero_initial_velocity_checked": True,
        "original_force_and_hessian_sources_copied_unchanged": True,
        "matrix_sha256": inputs[run / "results/first_dynamic_matrix.npz"],
        "new_physical_states_accepted": 0,
        "factorization_attempted": False,
        "input_sha256": {str(path): digest for path, digest in inputs.items()},
        "goal_completion": False,
        "limitations": "The producer's real code1 metadata-serialization failure is retained. This independent postprocessing audits already-written matrix/state arrays without rerunning the expensive nonlinear assembly or inventing a successful producer exit. Declared graph columns are diagnostics, not proof of the MemoryError cause or qualification of a new factorization. No physical state, convergence or warmed-cost acceptance is inferred.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}), flush=True)


if __name__ == "__main__":
    main()
