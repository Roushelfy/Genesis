"""Test a fully qualified factorization on the exact retained fine paired Newton equation."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-audit", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--factor-kind", choices=["bordered", "schur"], default="bordered")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.matrix_audit, args.qualification]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    qualification_path = args.qualification / "results/summary.json"
    qualified = json.loads(qualification_path.read_text())
    assert (
        qualified["all_original_arrays_and_complete_fields_qualified"]
        and qualified["states_checked"] == len(qualified["rows"]) == 5
    )
    force_rows = [row for state in qualified["rows"] for row in state["independent_force_audits"]]
    assert len(force_rows) == 10
    assert all(
        max(row["complete_body_dynamic_residual_N"], row["complete_mapped_dynamic_residual_N"]) <= 1e-9
        for row in force_rows
    )
    candidate = Path(qualified["candidate"])
    closed_source(candidate)
    assert int((candidate / "returncode.txt").read_text()) == 0
    helper_name = args.factor_kind + "_metis_reference_factor"
    for name in [helper_name, "metis_reference_factor", "franka_rigid_mode_coordinates"]:
        assert (
            Path(__file__).with_name(name + ".py").read_bytes()
            == (candidate / "source/research/rigid_stress" / (name + ".py")).read_bytes()
        )
    audit_path = args.matrix_audit / "results/summary.json"
    audit = json.loads(audit_path.read_text())
    assert audit["new_physical_states_accepted"] == 0 and not audit["factorization_attempted"]
    captured = Path(audit["capture_run"])
    closed_source(captured)
    assert int((captured / "returncode.txt").read_text()) == audit["capture_actual_wrapper_code"] == 1
    for name in [
        "precise_stencil_spatial",
        "wide_spatial_franka",
        "wide_coordinate_franka",
        "contact_reference",
        "neohookean_reference",
        "skfem_reference",
        "franka_rigid_mode_coordinates",
    ]:
        relative = Path("research/rigid_stress") / (name + ".py")
        assert (captured / "source" / relative).read_bytes() == (candidate / "source" / relative).read_bytes()
    matrix_path = captured / "results/first_dynamic_matrix.npz"
    assert hashlib.sha256(matrix_path.read_bytes()).hexdigest() == audit["matrix_sha256"]
    for path, digest in audit["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    matrix = sparse.load_npz(matrix_path)
    assert matrix.shape == (audit["matrix_dimension"],) * 2 and matrix.nnz == audit["matrix_nonzeros"]
    with np.load(captured / "results/first_dynamic_state.npz") as saved:
        rhs = saved["reduced_rhs_N"].copy()
        np.testing.assert_array_equal(
            saved["explicit_rigid_and_jaw_columns"], np.arange(matrix.shape[0] - 8, matrix.shape[0])
        )
    before = [hashlib.sha256(value.tobytes()).hexdigest() for value in [matrix.data, matrix.indices, matrix.indptr]]
    started = perf_counter()
    method = importlib.import_module("." + helper_name, __package__)
    factor = method.factorize(matrix, "paired_dynamic1")
    factor_seconds = perf_counter() - started
    direction = factor.solve(rhs)
    residual = float(np.linalg.norm(matrix @ direction - rhs))
    budget = max(1e-12, 1e-11 * float(np.linalg.norm(rhs)))
    assert residual <= budget, (residual, budget)
    assert before == [
        hashlib.sha256(value.tobytes()).hexdigest() for value in [matrix.data, matrix.indices, matrix.indptr]
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "linear_direction.npz", exact_original_rhs_N=rhs, reduced_newton_direction_m=direction
    )
    result = {
        "scope": __doc__,
        "factor_kind": args.factor_kind,
        "matrix_audit": str(args.matrix_audit),
        "qualification": str(args.qualification),
        "captured_matrix_sha256": audit["matrix_sha256"],
        "all_original_five_states_and_force_field_gates_qualified_before_large_factor": True,
        "same_original_body_force_hessian_and_coordinate_sources_verified": True,
        "dimension": matrix.shape[0],
        "matrix_nonzeros": int(matrix.nnz),
        "ordering_and_factor_seconds": factor_seconds,
        "elastic_or_full_sparse_factor_nonzeros": int(factor.factor.L.nnz + factor.factor.U.nnz),
        "true_original_linear_residual_N": residual,
        "original_cached_newton_linear_budget_N": budget,
        "original_matrix_arrays_unchanged": True,
        "coefficient_pruning_or_regularization_added": False,
        "new_physical_states_accepted": 0,
        "goal_completion": False,
        "input_sha256": {
            str(qualification_path): hashlib.sha256(qualification_path.read_bytes()).hexdigest(),
            str(audit_path): hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            str(matrix_path): audit["matrix_sha256"],
        },
        "limitations": "The permutation or exact Schur block elimination changes the numerical factorization while retaining every original matrix coefficient and the same complete equation. The exact captured first dynamic equation is solved after complete original1920-tet five-state force/corner/quadrature qualification. The large linear correction is not a converged nonlinear contact state, a controlled warmed speedup, proof of the specific internal MemoryError limit or physical convergence. Full nonlinear continuation and independent original-budget state/field audits remain mandatory.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
