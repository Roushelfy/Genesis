"""Use the existing CVXOPT CHOLMOD interface with 64-bit sparse indices."""

from importlib.metadata import version

import numpy as np
from scipy import sparse


class CvxoptCholeskyFactor:
    """Keep the supplied mesh ordering and factor without a diagonal shift."""

    def __init__(self, matrix):
        import cvxopt
        from cvxopt import cholmod

        assert version("cvxopt") == "1.3.2"
        self.index_bytes = np.asarray(cvxopt.matrix([0], tc="i")).dtype.itemsize
        assert self.index_bytes == 8, "This reference requires the 64-bit CVXOPT integer ABI"
        self.module_path = cholmod.__file__
        self.cvxopt, self.cholmod = cvxopt, cholmod
        lower = sparse.tril(matrix, format="coo")
        operator = cvxopt.spmatrix(
            cvxopt.matrix(lower.data),
            cvxopt.matrix(lower.row.astype(np.int64)),
            cvxopt.matrix(lower.col.astype(np.int64)),
            size=matrix.shape,
        )
        del lower
        previous = dict(cholmod.options)
        try:
            cholmod.options.update(supernodal=2, nmethods=1, postorder=True, dbound=0.0)
            order = cvxopt.matrix(np.arange(matrix.shape[0], dtype=np.int64))
            self.factor = cholmod.symbolic(operator, p=order, uplo="L")
            cholmod.numeric(operator, self.factor)
        finally:
            cholmod.options.clear()
            cholmod.options.update(previous)

    def solve(self, rhs):
        answer = self.cvxopt.matrix(np.asarray(rhs, dtype=np.float64, order="F"))
        self.cholmod.solve(self.factor, answer)
        return np.array(answer).reshape(rhs.shape, order="F")
