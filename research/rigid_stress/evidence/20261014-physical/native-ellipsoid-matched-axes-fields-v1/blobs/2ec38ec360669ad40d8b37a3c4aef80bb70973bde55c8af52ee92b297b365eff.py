"""Solve the unchanged paired Newton matrix using an elastic LU and its exact eight-coordinate Schur complement."""

import atexit
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pymetis
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

from .metis_reference_factor import factorize as original_factorize

RECORDS = []


def save_records():
    path = os.environ.get("REFERENCE_SCHUR_METIS_STATS_PATH")
    if path:
        Path(path).write_text(json.dumps({"factorizations": RECORDS}, indent=2) + "\n")


atexit.register(save_records)


class SchurReferenceFactor:
    def __init__(self, matrix, phase):
        started = perf_counter()
        self.matrix = matrix
        self.interior_size = matrix.shape[0] - 8
        assert matrix.shape[0] == matrix.shape[1] and self.interior_size > 0
        block = matrix[: self.interior_size, : self.interior_size].tocsc()
        graph = block.tocsr(copy=True)
        graph.eliminate_zeros()
        graph.data = np.ones(graph.nnz, dtype=np.int8)
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.sort_indices()
        dtype = pymetis.zero_copy_dtype()
        adjacency = pymetis.CSRAdjacency(np.asarray(graph.indptr, dtype=dtype), np.asarray(graph.indices, dtype=dtype))
        order, inverse = pymetis.nested_dissection(adjacency, options=pymetis.Options(seed=20261015))
        self.permutation, inverse = np.asarray(order, dtype=np.int64), np.asarray(inverse, dtype=np.int64)
        np.testing.assert_array_equal(self.permutation[inverse], np.arange(self.interior_size))
        ordered = perf_counter()
        self.factor = splu(block[self.permutation][:, self.permutation].tocsc(), permc_spec="NATURAL")
        factored = perf_counter()
        upper_right = matrix[: self.interior_size, self.interior_size :].toarray()
        self.lower_left = matrix[self.interior_size :, : self.interior_size].tocsr()
        self.interior_response = self.solve_interior(upper_right)
        schur = matrix[self.interior_size :, self.interior_size :].toarray() - self.lower_left @ self.interior_response
        self.schur_factor = lu_factor(schur)
        self.record = {
            "phase": phase,
            "dimension": matrix.shape[0],
            "original_matrix_nonzeros": int(matrix.nnz),
            "elastic_block_dimension": self.interior_size,
            "elastic_block_nonzeros": int(block.nnz),
            "declared_six_rigid_plus_two_jaw_columns": list(range(self.interior_size, matrix.shape[0])),
            "ordering_seconds": ordered - started,
            "elastic_factor_seconds": factored - ordered,
            "coupling_and_schur_seconds": perf_counter() - factored,
            "elastic_factor_nonzeros": int(self.factor.L.nnz + self.factor.U.nnz),
            "schur_dimension": 8,
            "schur_condition_2": float(np.linalg.cond(schur)),
            "interior_response_shape": self.interior_response.shape,
            "permutation_sha256": hashlib.sha256(self.permutation.tobytes()).hexdigest(),
            "full_matrix_values_and_equation_unchanged": True,
            "coefficient_pruning_or_physical_regularization_added": False,
            "linear_solve_records": [],
        }
        RECORDS.append(self.record)
        save_records()
        print(json.dumps({"reference_exact_schur_factor": self.record}), flush=True)

    def solve_interior(self, rhs):
        result = np.empty_like(rhs, dtype=float)
        result[self.permutation] = self.factor.solve(rhs[self.permutation])
        return result

    def solve_once(self, rhs):
        interior = self.solve_interior(rhs[: self.interior_size])
        border = lu_solve(self.schur_factor, rhs[self.interior_size :] - self.lower_left @ interior)
        result = np.empty_like(rhs, dtype=float)
        result[: self.interior_size] = interior - self.interior_response @ border
        result[self.interior_size :] = border
        return result

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        result = self.solve_once(rhs)
        budget = max(1e-12, 1e-11 * float(np.linalg.norm(rhs)))
        history = []
        for iteration in range(3):
            residual = rhs - self.matrix @ result
            norm = float(np.linalg.norm(residual))
            history.append(norm)
            if norm <= budget or iteration == 2:
                break
            result += self.solve_once(residual)
        self.record["linear_solve_records"].append(
            {"true_complete_linear_residuals_N": history, "original_cached_newton_budget_N": budget}
        )
        return result


def factorize(matrix, phase):
    if not phase.startswith(("paired_dynamic", "generic_stencil_dynamic")):
        return original_factorize(matrix)
    return SchurReferenceFactor(matrix, phase)
