"""Optional public cuDSS 0.8 C API binding with one device factor and multiple right-hand-side shapes."""

import ctypes as ct
import os
import weakref
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from scipy import sparse


def checked(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"cuDSS {operation} returned status {status}")


class CuDSSLibrary:
    """Signatures and enum values follow the installed, version-checked public 0.8 header."""

    def __init__(self) -> None:
        configured = os.environ.get("CUDSS_LIBRARY_PATH")
        path = (
            Path(configured)
            if configured
            else distribution("nvidia-cudss-cu12").locate_file("nvidia/cu12/lib/libcudss.so.0")
        )
        self.library = ct.CDLL(str(path))
        pointer, integer, long, size = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_size_t
        output = ct.POINTER(pointer)
        self.library.cudssCreate.argtypes = [output]
        self.library.cudssDestroy.argtypes = [pointer]
        self.library.cudssConfigCreate.argtypes = [output]
        self.library.cudssConfigDestroy.argtypes = [pointer]
        self.library.cudssDataCreate.argtypes = [pointer, output]
        self.library.cudssDataDestroy.argtypes = [pointer, pointer]
        self.library.cudssSetStream.argtypes = [pointer, pointer]
        self.library.cudssGetProperty.argtypes = [integer, ct.POINTER(integer)]
        self.library.cudssConfigSet.argtypes = [pointer, integer, pointer, size]
        self.library.cudssDataSet.argtypes = [pointer, pointer, integer, pointer, size]
        self.library.cudssDataGet.argtypes = [pointer, pointer, integer, pointer, size, ct.POINTER(size)]
        self.library.cudssExecute.argtypes = [pointer, integer, pointer, pointer, pointer, pointer, pointer]
        self.library.cudssMatrixCreateDn.argtypes = [output, long, long, long, pointer, integer, integer]
        self.library.cudssMatrixCreateCsr.argtypes = [output, long, long, long] + [pointer] * 4 + [integer] * 6
        self.library.cudssMatrixDestroy.argtypes = [pointer]
        version = []
        for property_type in range(3):
            value = integer()
            checked(self.library.cudssGetProperty(property_type, ct.byref(value)), "version query")
            version.append(value.value)
        self.version = tuple(version)
        if self.version[:2] != (0, 8):
            raise ValueError(f"This binding requires cuDSS 0.8, found {self.version}")


@dataclass(frozen=True)
class CuDSSPlan:
    columns: int
    input: cp.ndarray
    output: cp.ndarray
    rhs: ct.c_void_p
    solution: ct.c_void_p


def release(library, handle, config, data, matrix, plans) -> None:
    for plan in plans:
        library.cudssMatrixDestroy(plan.rhs)
        library.cudssMatrixDestroy(plan.solution)
    library.cudssMatrixDestroy(matrix)
    library.cudssDataDestroy(handle, data)
    library.cudssConfigDestroy(config)
    library.cudssDestroy(handle)


class SharedCuDSSFactor:
    """Factor one SPD stiffness matrix on GPU, then solve all batches with that single opaque factor.

    Runtime right-hand sides and solutions remain on the creation stream's device. Shape-specific descriptors own only
    dense scratch, never a factor copy. Device-only execution is explicit; hybrid host execution is disabled by default.
    Error queries synchronize during preprocessing. Runtime acceptance uses the complete FP64 residual.
    """

    def __init__(
        self,
        matrix: sparse.spmatrix,
        columns: int,
        dtype=np.float64,
        natural_order: bool = False,
    ) -> None:
        if dtype not in (np.float64, np.float32) or columns < 1 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Square SPD matrix, positive RHS capacity and FP64/FP32 precision required")
        self.api = CuDSSLibrary()
        self.stream = cp.cuda.get_current_stream()
        self.dtype, self.rows = np.dtype(dtype), matrix.shape[0]
        triangle = sparse.tril(matrix, format="csr")
        triangle.sum_duplicates()
        triangle.sort_indices()
        self.offsets = cp.asarray(triangle.indptr, dtype=np.int64)
        self.indices = cp.asarray(triangle.indices, dtype=np.int64)
        self.values = cp.asarray(triangle.data, dtype=dtype)
        self.handle, self.config, self.data, self.matrix = (ct.c_void_p() for _ in range(4))
        self.plans: list[CuDSSPlan] = []
        lib = self.api.library
        checked(lib.cudssCreate(ct.byref(self.handle)), "create")
        checked(lib.cudssSetStream(self.handle, self.stream.ptr), "set stream")
        checked(lib.cudssConfigCreate(ct.byref(self.config)), "create config")
        checked(lib.cudssDataCreate(self.handle, ct.byref(self.data)), "create data")
        # CUDA_R_64I=24, CUDA_R_64F=1, CUDA_R_32F=0; SPD=3, lower view=1, zero index base=0.
        value_type = 1 if dtype == np.float64 else 0
        checked(
            lib.cudssMatrixCreateCsr(
                ct.byref(self.matrix),
                self.rows,
                self.rows,
                len(self.values),
                self.offsets.data.ptr,
                None,
                self.indices.data.ptr,
                self.values.data.ptr,
                24,
                24,
                value_type,
                3,
                1,
                0,
            ),
            "create sparse matrix",
        )
        # No numerical diagonal shift. The unchanged complete stiffness determines residual acceptance.
        epsilon = ct.c_double(0)
        checked(lib.cudssConfigSet(self.config, 9, ct.byref(epsilon), ct.sizeof(epsilon)), "set pivot epsilon")
        if natural_order:
            algorithm = ct.c_int(5)
            checked(lib.cudssConfigSet(self.config, 0, ct.byref(algorithm), ct.sizeof(algorithm)), "natural order")
        self.prepare(columns)
        self.finalizer = weakref.finalize(
            self,
            release,
            lib,
            self.handle,
            self.config,
            self.data,
            self.matrix,
            self.plans,
        )
        start = perf_counter()
        self.execute(3, self.plans[0])
        self.stream.synchronize()
        self.analysis_s = perf_counter() - start
        estimate = (ct.c_int64 * 16)()
        self.get_data(13, estimate)
        self.memory_estimates_bytes = tuple(estimate)
        available, _ = cp.cuda.runtime.memGetInfo()
        if estimate[1] > available:
            raise MemoryError(f"cuDSS estimates {estimate[1]} peak device bytes with {available} currently available")
        start = perf_counter()
        self.execute(4, self.plans[0])
        self.stream.synchronize()
        self.factorization_s = perf_counter() - start
        info, nnz = ct.c_int(), ct.c_int64()
        self.get_data(0, info)
        self.get_data(1, nnz)
        if info.value:
            raise ArithmeticError(f"cuDSS asynchronous factorization error at pivot {info.value}")
        self.factor_nnz = nnz.value

    def get_data(self, parameter: int, value) -> None:
        written = ct.c_size_t()
        checked(
            self.api.library.cudssDataGet(
                self.handle,
                self.data,
                parameter,
                ct.byref(value),
                ct.sizeof(value),
                ct.byref(written),
            ),
            f"data query {parameter}",
        )
        if written.value != ct.sizeof(value):
            raise RuntimeError("cuDSS returned an unexpected query size")

    def prepare(self, columns: int) -> None:
        if columns < 1:
            raise ValueError("Positive RHS column count required")
        if any(plan.columns == columns for plan in self.plans):
            return
        rhs_array = cp.zeros((self.rows, columns), dtype=self.dtype, order="F")
        solution_array = cp.zeros_like(rhs_array, order="F")
        rhs, solution = ct.c_void_p(), ct.c_void_p()
        value_type = 1 if self.dtype == np.float64 else 0
        checked(
            self.api.library.cudssMatrixCreateDn(
                ct.byref(rhs),
                self.rows,
                columns,
                self.rows,
                rhs_array.data.ptr,
                value_type,
                0,
            ),
            "create RHS descriptor",
        )
        checked(
            self.api.library.cudssMatrixCreateDn(
                ct.byref(solution),
                self.rows,
                columns,
                self.rows,
                solution_array.data.ptr,
                value_type,
                0,
            ),
            "create solution descriptor",
        )
        self.plans.append(CuDSSPlan(columns, rhs_array, solution_array, rhs, solution))

    def execute(self, phase: int, plan: CuDSSPlan) -> None:
        checked(
            self.api.library.cudssExecute(
                self.handle,
                phase,
                self.config,
                self.data,
                self.matrix,
                plan.solution,
                plan.rhs,
            ),
            f"execute phase {phase}",
        )

    def solve(self, rhs: cp.ndarray) -> cp.ndarray:
        if rhs.shape[0] != self.rows or rhs.ndim != 2 or rhs.dtype != self.dtype:
            raise ValueError("Right-hand sides require the factor's precision and [row, column] shape")
        if cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise ValueError("cuDSS factor must execute on its creation stream")
        self.prepare(rhs.shape[1])
        plan = next(plan for plan in self.plans if plan.columns == rhs.shape[1])
        plan.input[:] = rhs
        plan.output.fill(0)
        self.execute(1008, plan)
        return plan.output

    def close(self) -> None:
        self.finalizer()
