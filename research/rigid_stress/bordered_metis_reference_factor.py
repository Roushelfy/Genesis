"""Order declared paired rigid/jaw columns last while factorizing the complete unchanged Newton matrix."""

import atexit
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pymetis
from scipy.sparse.linalg import splu

from .metis_reference_factor import factorize as original_factorize

RECORDS = []


def save_records():
    path = os.environ.get("REFERENCE_BORDERED_METIS_STATS_PATH")
    if path:
        Path(path).write_text(json.dumps({"factorizations": RECORDS}, indent=2) + "\n")


atexit.register(save_records)


class BorderedReferenceFactor:
    def __init__(self, matrix, border, phase):
        started = perf_counter()
        border = np.asarray(border, dtype=np.int64)
        assert matrix.shape[0] == matrix.shape[1] and border.ndim == 1
        assert len(border) == len(np.unique(border)) and np.all((border >= 0) & (border < matrix.shape[0]))
        interior = np.setdiff1d(np.arange(matrix.shape[0]), border, assume_unique=True)
        graph = matrix.tocsr(copy=True)
        graph.eliminate_zeros()
        graph.data = np.ones(graph.nnz, dtype=np.int8)
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.sort_indices()
        degree = np.diff(graph.indptr)
        inside = graph[interior][:, interior].tocsr()
        dtype = pymetis.zero_copy_dtype()
        adjacency = pymetis.CSRAdjacency(
            np.asarray(inside.indptr, dtype=dtype), np.asarray(inside.indices, dtype=dtype)
        )
        order, inverse = pymetis.nested_dissection(adjacency, options=pymetis.Options(seed=20261015))
        order, inverse = np.asarray(order, dtype=np.int64), np.asarray(inverse, dtype=np.int64)
        np.testing.assert_array_equal(order[inverse], np.arange(len(interior)))
        self.permutation = np.r_[interior[order], border]
        np.testing.assert_array_equal(np.sort(self.permutation), np.arange(matrix.shape[0]))
        ordered = perf_counter()
        self.factor = splu(matrix[self.permutation][:, self.permutation].tocsc(), permc_spec="NATURAL")
        record = {
            "phase": phase,
            "dimension": matrix.shape[0],
            "original_matrix_nonzeros": matrix.nnz,
            "declared_border_columns": border.tolist(),
            "border_degrees": degree[border].tolist(),
            "interior_maximum_degree_in_full_graph": int(degree[interior].max()),
            "interior_graph_nonzeros": inside.nnz,
            "full_graph_nonzeros": graph.nnz,
            "permutation_sha256": hashlib.sha256(self.permutation.tobytes()).hexdigest(),
            "ordering_seconds": ordered - started,
            "factor_seconds": perf_counter() - ordered,
            "factor_nonzeros": self.factor.L.nnz + self.factor.U.nnz,
            "matrix_values_and_equation_unchanged": True,
            "complete_matrix_factorized": True,
            "coefficient_pruning_or_physical_regularization_added": False,
        }
        RECORDS.append(record)
        save_records()
        print(json.dumps({"reference_declared_border_factor": record}), flush=True)

    def solve(self, rhs):
        result = np.empty_like(rhs, dtype=float)
        result[self.permutation] = self.factor.solve(rhs[self.permutation])
        return result


def factorize(matrix, phase):
    if not phase.startswith(("paired_dynamic", "generic_stencil_dynamic")):
        return original_factorize(matrix)
    # These two drivers exclusively use franka_rigid_mode_coordinates.coordinate_map:
    # Nbody-6 elastic, six restored rigid, then two original jaw columns.
    assert matrix.shape[0] > 8
    return BorderedReferenceFactor(matrix, np.arange(matrix.shape[0] - 8, matrix.shape[0]), phase)
