"""Shared sparse GPU recovery with an offline CPU factorization.

The runtime uses cached cuSPARSE multi-column triangular solves. All operators, right-hand sides, displacements,
complete residuals and global peaks stay on the device. This module needs CuPy; CPU reference use does not import it.
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cusparse
from cupyx import cusparse as descriptors
from cupyx.scipy import sparse

from .cudss import SharedCuDSSFactor
from .native_sparse import NativeSparseCalls


class SparseMultiplyGPU:
    """Persistent public cuSPARSE SpMM descriptors, including capture-safe use of the owned native handle."""

    def __init__(self, matrix: sparse.csr_matrix, columns: int) -> None:
        self.matrix, self.stream = matrix, cp.cuda.get_current_stream()
        self.native = NativeSparseCalls()
        self.handle = descriptors.BaseDescriptor(cusparse.create(), destroyer=cusparse.destroy)
        cusparse.setStream(self.handle.desc, self.stream.ptr)
        self.input = cp.zeros((matrix.shape[1], columns), dtype=matrix.dtype, order="F")
        self.output = cp.zeros((matrix.shape[0], columns), dtype=matrix.dtype, order="F")
        self.matrix_descriptor = descriptors.SpMatDescriptor.create(matrix)
        self.input_descriptor = descriptors.DnMatDescriptor.create(self.input)
        self.output_descriptor = descriptors.DnMatDescriptor.create(self.output)
        self.alpha, self.beta = np.array(1, dtype=matrix.dtype), np.array(0, dtype=matrix.dtype)
        self.cuda_dtype = cp.cuda.runtime.CUDA_R_64F if matrix.dtype == np.float64 else cp.cuda.runtime.CUDA_R_32F
        self.operation, self.algorithm = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE, cusparse.CUSPARSE_MM_ALG_DEFAULT
        self.workspace = cp.empty(
            cusparse.spMM_bufferSize(
                self.handle.desc,
                self.operation,
                self.operation,
                self.alpha.ctypes.data,
                self.matrix_descriptor.desc,
                self.input_descriptor.desc,
                self.beta.ctypes.data,
                self.output_descriptor.desc,
                self.cuda_dtype,
                self.algorithm,
            ),
            dtype=np.int8,
        )

    def __call__(self, value: cp.ndarray) -> cp.ndarray:
        if value.shape != self.input.shape or value.dtype != self.input.dtype:
            raise ValueError("Sparse product requires its analyzed matrix shape and precision")
        if cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise ValueError("Sparse product must execute on its creation stream")
        self.input[:] = value
        self.native.multiply(
            self.handle.desc,
            self.operation,
            self.alpha.ctypes.data,
            self.matrix_descriptor.desc,
            self.input_descriptor.desc,
            self.beta.ctypes.data,
            self.output_descriptor.desc,
            self.cuda_dtype,
            self.algorithm,
            self.workspace.data.ptr,
        )
        return self.output


class SparseTriangularSolve:
    """Analyze a triangular factor once for an exact right-hand-side shape.

    The persistent input/output buffers are column-major. A solve returns its owned output buffer, which the next solve
    overwrites. Instances belong to their creation stream; concurrent streams need independent analysis and scratch.
    """

    def __init__(self, matrix: sparse.csr_matrix, columns: int, is_lower: bool, is_unit_diagonal: bool):
        self.matrix = matrix
        self.native = NativeSparseCalls()
        self.stream = cp.cuda.get_current_stream()
        self.handle = descriptors.BaseDescriptor(cusparse.create(), destroyer=cusparse.destroy)
        cusparse.setStream(self.handle.desc, self.stream.ptr)
        self.input = cp.zeros((matrix.shape[0], columns), dtype=matrix.dtype, order="F")
        self.output = cp.zeros_like(self.input, order="F")
        self.matrix_descriptor = descriptors.SpMatDescriptor.create(matrix)
        self.input_descriptor = descriptors.DnMatDescriptor.create(self.input)
        self.output_descriptor = descriptors.DnMatDescriptor.create(self.output)
        self.solve_descriptor = descriptors.BaseDescriptor(
            cusparse.spSM_createDescr(), destroyer=cusparse.spSM_destroyDescr
        )
        self.matrix_descriptor.set_attribute(
            cusparse.CUSPARSE_SPMAT_FILL_MODE,
            cusparse.CUSPARSE_FILL_MODE_LOWER if is_lower else cusparse.CUSPARSE_FILL_MODE_UPPER,
        )
        self.matrix_descriptor.set_attribute(
            cusparse.CUSPARSE_SPMAT_DIAG_TYPE,
            cusparse.CUSPARSE_DIAG_TYPE_UNIT if is_unit_diagonal else cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT,
        )
        self.alpha = np.array(1, dtype=matrix.dtype)
        self.cuda_dtype = cp.cuda.runtime.CUDA_R_64F if matrix.dtype == np.float64 else cp.cuda.runtime.CUDA_R_32F
        self.operation = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        self.algorithm = cusparse.CUSPARSE_SPSM_ALG_DEFAULT
        arguments = (
            self.handle.desc,
            self.operation,
            self.operation,
            self.alpha.ctypes.data,
            self.matrix_descriptor.desc,
            self.input_descriptor.desc,
            self.output_descriptor.desc,
            self.cuda_dtype,
            self.algorithm,
            self.solve_descriptor.desc,
        )
        self.workspace = cp.empty(cusparse.spSM_bufferSize(*arguments), dtype=np.int8)
        cusparse.spSM_analysis(*arguments, self.workspace.data.ptr)

    def solve(self, rhs: cp.ndarray) -> cp.ndarray:
        if rhs.shape != self.input.shape or rhs.dtype != self.input.dtype:
            raise ValueError("The analyzed triangular solve requires its exact shape and precision")
        if cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise ValueError("The triangular solve must run on its creation stream")
        self.input[...] = rhs
        self.output.fill(0)
        self.native.triangular(
            self.handle.desc,
            self.operation,
            self.alpha.ctypes.data,
            self.matrix_descriptor.desc,
            self.input_descriptor.desc,
            self.output_descriptor.desc,
            self.cuda_dtype,
            self.algorithm,
            self.solve_descriptor.desc,
        )
        return self.output


@dataclass(frozen=True)
class FactorPlan:
    columns: int
    forward: SparseTriangularSolve
    backward: SparseTriangularSolve


class SharedSparseFactor:
    """Upload one SuperLU factor pair and solve many columns without refactorization.

    SuperLU permutations satisfy Pr A Pc = L U. Gather the RHS using inverse perm_r, then gather the solution
    using perm_c. Factors are shared across column counts; analyzed descriptors and scratch vary.
    """

    def __init__(self, factor, dtype=np.float64, unit_lower=True):
        if dtype not in (np.float64, np.float32):
            raise ValueError("Sparse recovery supports explicit FP64 or FP32 factors")
        self.lower = sparse.csr_matrix(factor.L.astype(dtype))
        self.upper = sparse.csr_matrix(factor.U.astype(dtype))
        self.row_gather = cp.asarray(np.argsort(factor.perm_r))
        self.column_gather = cp.asarray(factor.perm_c)
        self.is_unit_lower = unit_lower
        self.plans: list[FactorPlan] = []

    def prepare(self, columns: int) -> None:
        if columns < 1:
            raise ValueError("Analyze at least one right-hand-side column")
        if all(plan.columns != columns for plan in self.plans):
            self.plans.append(
                FactorPlan(
                    columns,
                    SparseTriangularSolve(self.lower, columns, is_lower=True, is_unit_diagonal=self.is_unit_lower),
                    SparseTriangularSolve(self.upper, columns, is_lower=False, is_unit_diagonal=False),
                )
            )

    def solve(self, rhs: cp.ndarray) -> cp.ndarray:
        if rhs.ndim != 2 or rhs.shape[0] != self.lower.shape[0]:
            raise ValueError("Right-hand sides must have shape [free_dof, environments]")
        if rhs.shape[1] < 1:
            return cp.empty_like(rhs)
        self.prepare(rhs.shape[1])
        plan = next(plan for plan in self.plans if plan.columns == rhs.shape[1])
        intermediate = plan.forward.solve(rhs[self.row_gather])
        return plan.backward.solve(intermediate)[self.column_gather]


PEAK_SOURCE = r"""
extern "C" __global__ void peak(
    const double* displacement, const long long* elements, const double* gradients,
    double* partial, int dofs, int tetrahedra, int tiles) {
    __shared__ double maximum[128];
    int environment = blockIdx.y;
    int element = blockIdx.x * blockDim.x + threadIdx.x;
    double best = 0;
    if (element < tetrahedra) {
        double local[10][3];
        for (int node = 0; node < 10; ++node) {
            long long global = elements[10 * element + node];
            long long origin = elements[10 * element];
            for (int axis = 0; axis < 3; ++axis) {
                local[node][axis] = displacement[environment * dofs + 3 * global + axis]
                    - displacement[environment * dofs + 3 * origin + axis];
            }
        }
        const int incident[16] = {0,4,5,6,4,1,7,8,5,7,2,9,6,8,9,3};
        double base[9] = {0};
        for (int axis = 0; axis < 3; ++axis) {
            for (int derivative = 0; derivative < 3; ++derivative) {
                for (int vertex = 0; vertex < 4; ++vertex) {
                    base[3 * axis + derivative] -= local[vertex][axis]
                        * gradients[12 * element + 3 * vertex + derivative];
                }
            }
        }
        for (int corner = 0; corner < 4; ++corner) {
            double strain[9];
            for (int axis = 0; axis < 3; ++axis) {
                for (int derivative = 0; derivative < 3; ++derivative) {
                    int entry = 3 * axis + derivative;
                    strain[entry] = base[entry];
                    for (int vertex = 0; vertex < 4; ++vertex) {
                        strain[entry] += 4 * local[incident[4 * corner + vertex]][axis]
                            * gradients[12 * element + 3 * vertex + derivative];
                    }
                }
            }
            double xy = strain[1] + strain[3];
            double yz = strain[5] + strain[7];
            double xz = strain[2] + strain[6];
            double xx_yy = strain[0] - strain[4];
            double yy_zz = strain[4] - strain[8];
            double zz_xx = strain[8] - strain[0];
            double squared = 2 * (xx_yy * xx_yy + yy_zz * yy_zz + zz_xx * zz_xx)
                + 3 * (xy * xy + yz * yz + xz * xz);
            best = fmax(best, squared);
        }
    }
    maximum[threadIdx.x] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) maximum[threadIdx.x] = fmax(maximum[threadIdx.x], maximum[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[environment * tiles + blockIdx.x] = maximum[0];
}
"""


class P2PeakGPU:
    """Reduce all four corners of every affine quadratic tetrahedron on GPU.

    The output is the exact discrete unaveraged isotropic von Mises maximum in Pa. Only one partial scalar per tile and
    environment is stored; no full-domain stress tensor is allocated. Arithmetic is FP64, including for FP32 factors.
    """

    def __init__(self, gradients: np.ndarray, elements: np.ndarray, shear_pa: float):
        self.gradients = cp.asarray(gradients, dtype=np.float64)
        self.elements = cp.asarray(elements, dtype=np.int64)
        self.shear_pa = shear_pa
        self.kernel = cp.RawKernel(PEAK_SOURCE, "peak")

    def __call__(self, displacement: cp.ndarray) -> cp.ndarray:
        displacement = cp.asfortranarray(displacement, dtype=np.float64)
        tiles = (len(self.elements) + 127) // 128
        partial = cp.empty((displacement.shape[1], tiles), dtype=np.float64)
        self.kernel(
            (tiles, displacement.shape[1]),
            (128,),
            (displacement, self.elements, self.gradients, partial, displacement.shape[0], len(self.elements), tiles),
        )
        return self.shear_pa * cp.sqrt(partial.max(axis=1))


@dataclass(frozen=True)
class GPURecoveryResult:
    peak_pa: cp.ndarray
    displacement_m: cp.ndarray
    relative_residual: cp.ndarray
    absolute_residual_n: cp.ndarray
    is_accepted: cp.ndarray
    rhs_norm_n: cp.ndarray


@dataclass(frozen=True)
class GPUWorkStatistics:
    failed: cp.ndarray
    refinement_count: cp.ndarray
    used_fp64_fallback: cp.ndarray
    rejected_direction: cp.ndarray
    solved_columns: int


class EggRecoveryGPU:
    """Recover stress with shared sparse factors and a complete FP64 residual.

    Construction accepts immutable CPU oracle operators. Factorization and upload happen once. Runtime inputs and
    Outputs are CuPy device arrays. Check is_accepted before using a result as a validated observation.
    """

    def __init__(
        self,
        fem,
        environments: int,
        rtol: float = 1e-6,
        atol_n: float = 1e-11,
        factor_backend: str = "spsm",
        shared_factor: SharedSparseFactor | SharedCuDSSFactor | None = None,
    ):
        if environments < 1 or rtol <= 0 or atol_n <= 0:
            raise ValueError("Positive environment count and tolerances are required")
        self.environments = environments
        self.statistics: GPUWorkStatistics | None = None
        self.rtol = rtol
        self.atol_n = atol_n
        self.stiffness = sparse.csr_matrix(fem.k)
        self.mass = sparse.csr_matrix(fem.m)
        self.free = cp.asarray(fem.free)
        self.rigid_modes = cp.asarray(fem.r)
        self.mass_modes = cp.asarray(fem.mr)
        self.gram_inverse = cp.asarray(np.linalg.inv(fem.gram))
        self.relative_position = cp.asarray(fem.xyz - fem.com)
        if shared_factor is not None:
            if factor_backend == "cudss" and not isinstance(shared_factor, SharedCuDSSFactor):
                raise ValueError("A reused cuDSS factor requires the cuDSS backend")
            if factor_backend == "spsm" and not isinstance(shared_factor, SharedSparseFactor):
                raise ValueError("A reused exported factor requires the SpSM backend")
            self.factor = shared_factor
        elif factor_backend == "spsm":
            self.factor = SharedSparseFactor(fem.factor, unit_lower=fem.is_unit_lower)
        elif factor_backend == "cudss":
            self.factor = SharedCuDSSFactor(fem.k[fem.free][:, fem.free], environments)
        else:
            raise ValueError("Select exported SpSM or native cuDSS device factors")
        self.factor.prepare(environments)
        self.stiffness_product = SparseMultiplyGPU(self.stiffness, environments)
        self.peak = P2PeakGPU(fem.glambda, fem.elements, fem.young / (2 * (1 + fem.poisson)))

    def apply_stiffness(self, value: cp.ndarray) -> cp.ndarray:
        return self.stiffness_product(value)

    def compatible_rhs(self, external_n: cp.ndarray, omega_rad_s: cp.ndarray | None = None) -> cp.ndarray:
        if external_n.shape != (self.stiffness.shape[0], self.environments):
            raise ValueError("Complete external nodal loads must have shape [dof, environments]")
        force = external_n
        if omega_rad_s is not None:
            if omega_rad_s.shape != (self.environments, 3):
                raise ValueError("Angular velocities must have shape [environments, 3]")
            centrifugal = cp.cross(omega_rad_s[:, None], cp.cross(omega_rad_s[:, None], self.relative_position))
            force = external_n - self.mass @ centrifugal.reshape(self.environments, -1).T
        return force - self.mass_modes @ (self.gram_inverse @ (self.rigid_modes.T @ force))

    def reset(self, environments: cp.ndarray | None = None) -> None:
        """The direct baseline has no temporal state; retain the same lifecycle API as temporal recovery."""

    def recover(self, rhs_n: cp.ndarray, dt: float | cp.ndarray = 1.0) -> GPURecoveryResult:
        if rhs_n.shape != (self.stiffness.shape[0], self.environments) or rhs_n.dtype != np.float64:
            raise ValueError("The direct baseline requires FP64 [dof, environments] right-hand sides")
        displacement = cp.zeros_like(rhs_n, order="F")
        displacement[self.free] = self.factor.solve(rhs_n[self.free])
        residual = rhs_n - self.apply_stiffness(displacement)
        absolute = cp.linalg.norm(residual, axis=0)
        norm = cp.linalg.norm(rhs_n, axis=0)
        relative = absolute / cp.maximum(norm, self.atol_n)
        peaks = self.peak(displacement)
        is_accepted = (
            (absolute <= cp.maximum(self.atol_n, self.rtol * norm))
            & cp.isfinite(absolute)
            & cp.isfinite(norm)
            & cp.isfinite(peaks)
        )
        self.statistics = GPUWorkStatistics(
            cp.ones(self.environments, dtype=bool),
            cp.zeros(self.environments, dtype=np.int32),
            cp.zeros(self.environments, dtype=bool),
            cp.zeros(self.environments, dtype=bool),
            self.environments,
        )
        return GPURecoveryResult(peaks, displacement, relative, absolute, is_accepted, norm)
