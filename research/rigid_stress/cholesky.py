"""Optional long-index CHOLMOD FP64 oracle and public triangular-factor export."""

from contextlib import contextmanager

import numpy as np
from scipy import sparse

try:
    import cvxopt
    from cvxopt import cholmod
except ModuleNotFoundError as import_error:
    if import_error.name != "cvxopt":
        raise
    cvxopt = cholmod = None


@contextmanager
def natural_cholesky_options():
    previous = cholmod.options.copy()
    cholmod.options.update(supernodal=2, nmethods=1, postorder=False)
    try:
        yield
    finally:
        cholmod.options.clear()
        cholmod.options.update(previous)


class CholeskyDirectFactor:
    def __init__(self, matrix: sparse.csc_matrix) -> None:
        if cvxopt is None:
            raise ModuleNotFoundError("The optional long-index CHOLMOD backend requires cvxopt==1.3.2")
        lower = sparse.tril(matrix, format="coo")
        self.matrix = cvxopt.spmatrix(lower.data, lower.row.astype(np.int64), lower.col.astype(np.int64), matrix.shape)
        self.identity = cvxopt.matrix(np.arange(matrix.shape[0], dtype=np.int64), tc="i")
        with natural_cholesky_options():
            self.native = cholmod.symbolic(self.matrix, p=self.identity)
            cholmod.numeric(self.matrix, self.native)
        self.perm_r = np.arange(matrix.shape[0], dtype=np.int32)
        self.perm_c = self.perm_r.copy()
        self.lower: sparse.csc_matrix | None = None
        self.factorization_count = 1

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.matrix.size[0]:
            raise ValueError("Direct right-hand sides must match the factor rows")
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            return np.empty_like(rhs)
        dense = cvxopt.matrix(np.asfortranarray(rhs, dtype=np.float64))
        with natural_cholesky_options():
            cholmod.solve(self.native, dense)
        return np.asarray(dense).reshape(rhs.shape, order="F").copy()

    @property
    def L(self) -> sparse.csc_matrix:
        if self.lower is None:
            with natural_cholesky_options():
                # The public CVXOPT export consumes the numeric factor. Rebuild it for independent CPU direct solves.
                exported = cholmod.getfactor(self.native)
                pointers, indices, values = exported.CCS
                self.lower = sparse.csc_matrix((
                    np.asarray(values).ravel(), np.asarray(indices).ravel(), np.asarray(pointers).ravel(),
                ), shape=exported.size)
                cholmod.numeric(self.matrix, self.native)
                self.factorization_count += 1
        return self.lower

    @property
    def U(self) -> sparse.csr_matrix:
        return self.L.T
