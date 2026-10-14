"""Exact GPU products for fixed geometry fields and complete nodal force/moment reduction."""

import cupy as cp
import numpy as np

BODY_SOURCE = r"""
extern "C" __global__ void field_product(
    const double* field, const double* coefficients, double* output, int rows, int fields, int environments) {
    int row = blockIdx.x*blockDim.x+threadIdx.x, environment = blockIdx.y;
    if (row < rows) {
        double value = 0;
        for (int column = 0; column < fields; ++column)
            value += field[row*fields+column]*coefficients[column*environments+environment];
        output[environment*rows+row] = value;
    }
}

extern "C" __global__ void wrench(
    const double* position, const double* force, double* partial, int nodes, int tiles) {
    int node = blockIdx.x*blockDim.x+threadIdx.x, environment = blockIdx.y;
    __shared__ double scratch[6*128];
    double value[6] = {0};
    if (node < nodes) {
        double x = position[3*node], y = position[3*node+1], z = position[3*node+2];
        double fx = force[environment*3*nodes+3*node];
        double fy = force[environment*3*nodes+3*node+1];
        double fz = force[environment*3*nodes+3*node+2];
        value[0] = fx; value[1] = fy; value[2] = fz;
        value[3] = y*fz-z*fy; value[4] = z*fx-x*fz; value[5] = x*fy-y*fx;
    }
    for (int axis = 0; axis < 6; ++axis) scratch[axis*128+threadIdx.x] = value[axis];
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride) for (int axis = 0; axis < 6; ++axis)
            scratch[axis*128+threadIdx.x] += scratch[axis*128+threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) for (int axis = 0; axis < 6; ++axis)
        partial[(environment*6+axis)*tiles+blockIdx.x] = scratch[axis*128];
}
"""


class FixedFieldProductGPU:
    """Multiply all rows of a fixed geometry field matrix; never restrict the elastic/contact load space."""

    def __init__(self, fields: cp.ndarray) -> None:
        self.fields = cp.ascontiguousarray(fields, dtype=np.float64)
        self.kernel = cp.RawKernel(BODY_SOURCE, "field_product")

    def __call__(self, coefficients: cp.ndarray) -> cp.ndarray:
        if coefficients.ndim != 2 or coefficients.shape[0] != self.fields.shape[1]:
            raise ValueError("Coefficients must match the complete fixed geometry fields")
        coefficients = cp.ascontiguousarray(coefficients, dtype=np.float64)
        rows, fields = self.fields.shape
        environments = coefficients.shape[1]
        output = cp.empty((rows, environments), dtype=np.float64, order="F")
        self.kernel(
            ((rows + 127) // 128, environments),
            (128,),
            (self.fields, coefficients, output, rows, fields, environments),
        )
        return output


class NodalWrenchGPU:
    """Reduce translations and COM-centered rotations over every nodal load, including gauge rows."""

    def __init__(self, relative_position_m: cp.ndarray) -> None:
        self.position = cp.ascontiguousarray(relative_position_m, dtype=np.float64)
        self.kernel = cp.RawKernel(BODY_SOURCE, "wrench")

    def __call__(self, force_n: cp.ndarray) -> cp.ndarray:
        nodes = len(self.position)
        if force_n.ndim != 2 or force_n.shape[0] != 3 * nodes:
            raise ValueError("Complete nodal loads require all fixed-geometry DOFs")
        force = cp.asfortranarray(force_n, dtype=np.float64)
        tiles = (nodes + 127) // 128
        partial = cp.empty((force.shape[1], 6, tiles), dtype=np.float64)
        self.kernel((tiles, force.shape[1]), (128,), (self.position, force, partial, nodes, tiles))
        return partial.sum(axis=2).T
