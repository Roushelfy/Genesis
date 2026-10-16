"""Public native cuSPARSE calls on analyzed descriptors and independently owned stream handles."""

import ctypes as ct

from cuda import pathfinder


class NativeSparseCalls:
    """Resolve the same CUDA library as CuPy and retain streams assigned once outside capture.

    These calls use public C APIs. They do not mutate third-party modules or borrowed handles. Each caller owns its
    descriptors and stream-specific handle; CUDA capture support is checked on the target device.
    """

    def __init__(self) -> None:
        loaded = pathfinder.load_nvidia_dynamic_lib("cusparse")
        if loaded.abs_path is None:
            raise RuntimeError("CUDA pathfinder did not provide the loaded cuSPARSE path")
        self.library = ct.CDLL(loaded.abs_path)
        pointer, integer = ct.c_void_p, ct.c_int
        self.library.cusparseSpSM_solve.argtypes = [
            pointer,
            integer,
            integer,
            pointer,
            pointer,
            pointer,
            pointer,
            integer,
            integer,
            pointer,
        ]
        self.library.cusparseSpMM.argtypes = [
            pointer,
            integer,
            integer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            integer,
            integer,
            pointer,
        ]

    def triangular(self, handle, operation, alpha, matrix, rhs, solution, dtype, algorithm, descriptor) -> None:
        status = self.library.cusparseSpSM_solve(
            handle,
            operation,
            operation,
            alpha,
            matrix,
            rhs,
            solution,
            dtype,
            algorithm,
            descriptor,
        )
        if status:
            raise RuntimeError(f"Native cuSPARSE SpSM returned status {status}")

    def multiply(self, handle, operation, alpha, matrix, rhs, beta, solution, dtype, algorithm, workspace) -> None:
        status = self.library.cusparseSpMM(
            handle,
            operation,
            operation,
            alpha,
            matrix,
            rhs,
            beta,
            solution,
            dtype,
            algorithm,
            workspace,
        )
        if status:
            raise RuntimeError(f"Native cuSPARSE SpMM returned status {status}")
