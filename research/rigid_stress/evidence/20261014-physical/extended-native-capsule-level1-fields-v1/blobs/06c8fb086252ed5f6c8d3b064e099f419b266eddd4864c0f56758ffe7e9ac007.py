"""Apply a fixed seeded nested-dissection permutation before the unchanged reference LU solve."""

import atexit
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pymetis
from scipy.sparse.linalg import splu

RECORDS = []


def save_records():
    path = os.environ.get("REFERENCE_METIS_STATS_PATH")
    if path:
        Path(path).write_text(json.dumps({"factorizations": RECORDS}, indent=2) + "\n")


atexit.register(save_records)


class PermutedReferenceFactor:
    def __init__(self, matrix):
        started = perf_counter()
        graph = matrix.tocsr(copy=True)
        graph.eliminate_zeros()
        graph.data = np.ones(graph.nnz, dtype=np.int8)
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.sort_indices()
        dtype = pymetis.zero_copy_dtype()
        adjacency = pymetis.CSRAdjacency(np.asarray(graph.indptr, dtype=dtype), np.asarray(graph.indices, dtype=dtype))
        permutation, inverse = pymetis.nested_dissection(adjacency, options=pymetis.Options(seed=20261015))
        self.permutation = np.asarray(permutation, dtype=np.int64)
        inverse = np.asarray(inverse, dtype=np.int64)
        np.testing.assert_array_equal(self.permutation[inverse], np.arange(matrix.shape[0]))
        ordered_at = perf_counter()
        self.factor = splu(matrix[self.permutation][:, self.permutation].tocsc(), permc_spec="NATURAL")
        row = {
            "dimension": matrix.shape[0],
            "original_matrix_nonzeros": matrix.nnz,
            "symmetric_graph_nonzeros": graph.nnz,
            "seed": 20261015,
            "permutation_sha256": hashlib.sha256(self.permutation.tobytes()).hexdigest(),
            "ordering_seconds": ordered_at - started,
            "factor_seconds": perf_counter() - ordered_at,
            "factor_nonzeros": self.factor.L.nnz + self.factor.U.nnz,
            "matrix_values_and_equation_unchanged": True,
        }
        RECORDS.append(row)
        print(json.dumps({"reference_nested_dissection_factor": row}), flush=True)
        save_records()

    def solve(self, rhs):
        solved = np.empty_like(rhs, dtype=float)
        solved[self.permutation] = self.factor.solve(rhs[self.permutation])
        return solved


def factorize(matrix):
    return PermutedReferenceFactor(matrix)
