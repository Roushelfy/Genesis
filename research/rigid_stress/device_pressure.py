"""Map complete rigid contact wrenches to consistent quadratic shell loads on GPU.

The constant-ratio compliant pad law matches PadPressureFit. A block solves each active pressure footprint, scanning
the full quadrature domain. Arbitrary changing centers, radii and counts are supported. Inputs, acceptance flags and
loads remain on device. Mapping failures produce an explicit per-contact status for the caller to reject.
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np

PRESSURE_SOURCE = r"""
__device__ bool solve3(const double* g, const double* b, double* x) {
    double trace = g[0] + g[3] + g[5];
    if (!(g[0] > 1e-14 * trace)) return false;
    double l10 = g[1] / g[0], l20 = g[2] / g[0];
    double d1 = g[3] - l10 * g[1];
    if (!(d1 > 1e-14 * trace)) return false;
    double l21 = (g[4] - l20 * g[1]) / d1;
    double d2 = g[5] - l20 * g[2] - l21 * l21 * d1;
    if (!(d2 > 1e-14 * trace)) return false;
    double y0 = b[0], y1 = b[1] - l10 * y0, y2 = b[2] - l20 * y0 - l21 * y1;
    x[2] = y2 / d2;
    x[1] = y1 / d1 - l21 * x[2];
    x[0] = y0 / g[0] - l10 * x[1] - l20 * x[2];
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}

__device__ void footprint(
    const double* positions, const double* weights, int samples, const double* center,
    const double* tangent, const double* bitangent, double radius, const double* multiplier,
    double* scratch) {
    double total[11] = {0};
    for (int q = threadIdx.x; q < samples; q += blockDim.x) {
        double relative[3], distance = 0;
        for (int a = 0; a < 3; ++a) {
            relative[a] = (positions[3*q+a] - center[a]) / radius;
            distance += relative[a] * relative[a];
        }
        if (distance < 1) {
            double h1 = 0, h2 = 0;
            for (int a = 0; a < 3; ++a) {
                h1 += relative[a] * tangent[a];
                h2 += relative[a] * bitangent[a];
            }
            double weight = exp(-0.5 * distance / (0.45 * 0.45)) * (1-distance) * (1-distance) * weights[q];
            double profile = fmax(0., 1 - multiplier[0] - h1*multiplier[1] - h2*multiplier[2]);
            double active = profile > 0 ? weight : 0;
            total[0] += active;
            total[1] += active * h1;
            total[2] += active * h2;
            total[3] += active * h1 * h1;
            total[4] += active * h1 * h2;
            total[5] += active * h2 * h2;
            total[6] += weight * profile;
            total[7] += weight * profile * h1;
            total[8] += weight * profile * h2;
            total[9] += 0.5 * weight * profile * profile;
            total[10] += weight;
        }
    }
    for (int entry = 0; entry < 11; ++entry) scratch[entry*128+threadIdx.x] = total[entry];
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride)
            for (int entry = 0; entry < 11; ++entry)
                scratch[entry*128+threadIdx.x] += scratch[entry*128+threadIdx.x+stride];
        __syncthreads();
    }
}

extern "C" __global__ void pressure(
    const double* position, const double* force, const double* radius, const double* normal,
    const double* friction, const bool* valid, double source_epsilon,
    const double* samples, const double* weights, const double* shape, const long long* nodes,
    int quadrature, int sample_count, int contacts, int dofs, double* loads,
    int* status, double* diagnostics) {
    int contact = blockIdx.x, environment = blockIdx.y, index = environment*contacts + contact;
    if (!valid[index]) return;
    __shared__ double scratch[11*128];
    __shared__ double center[3], complete_force[3], tangent[3], bitangent[3];
    __shared__ double multiplier[3], candidate[3], step[3], reduced[11];
    __shared__ double r, magnitude, normal_force, excess, allowance, objective, fraction;
    __shared__ int flag, evaluations, is_accepted;
    if (threadIdx.x == 0) {
        flag = 0; evaluations = 0; magnitude = 0; double normal_norm = 0;
        r = radius[index];
        for (int a = 0; a < 3; ++a) {
            center[a] = position[3*index+a];
            complete_force[a] = force[3*index+a];
            magnitude += complete_force[a] * complete_force[a];
            normal_norm += normal[3*index+a] * normal[3*index+a];
            multiplier[a] = 0;
            if (!isfinite(center[a]) || !isfinite(complete_force[a])) flag = 1;
        }
        magnitude = sqrt(magnitude); normal_norm = sqrt(normal_norm);
        if (!(r > 0) || !isfinite(r) || !(normal_norm > 1e-12) || !isfinite(normal_norm)
            || !(friction[index] >= 0) || !isfinite(friction[index])) flag = 1;
        normal_force = 0;
        for (int a = 0; a < 3; ++a) normal_force += complete_force[a] * normal[3*index+a] / normal_norm;
        double tangent_squared = 0;
        for (int a = 0; a < 3; ++a) {
            double value = complete_force[a] - normal_force * normal[3*index+a] / normal_norm;
            tangent_squared += value * value;
        }
        excess = sqrt(tangent_squared) - friction[index] * normal_force;
        allowance = 16 * fmax(source_epsilon, 2.220446049250313e-16) * magnitude;
        if (normal_force < -allowance || excess > allowance) flag = 1;
        if (!flag && magnitude <= 1e-30) flag = 5;
        int axis = 0;
        for (int a = 1; a < 3; ++a) if (fabs(complete_force[a]) < fabs(complete_force[axis])) axis = a;
        for (int a = 0; a < 3; ++a) tangent[a] = 0;
        tangent[(axis+1)%3] = complete_force[(axis+2)%3] / magnitude;
        tangent[(axis+2)%3] = -complete_force[(axis+1)%3] / magnitude;
        double tangent_norm = sqrt(tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]);
        for (int a = 0; a < 3; ++a) tangent[a] /= tangent_norm;
        for (int a = 0; a < 3; ++a)
            bitangent[a] = (complete_force[(a+1)%3]*tangent[(a+2)%3]
                - complete_force[(a+2)%3]*tangent[(a+1)%3]) / magnitude;
    }
    __syncthreads();
    if (flag) {
        if (threadIdx.x == 0) status[index] = flag == 5 ? 0 : flag;
        return;
    }
    footprint(samples, weights, sample_count, center, tangent, bitangent, r, multiplier, scratch);
    if (threadIdx.x == 0) {
        for (int entry = 0; entry < 11; ++entry) reduced[entry] = scratch[entry*128];
        double right[3] = {reduced[6]-reduced[10], reduced[7], reduced[8]};
        if (!(reduced[10] > 0)) flag = 2;
        else if (!solve3(reduced, right, multiplier)) flag = 3;
    }
    __syncthreads();
    for (int iteration = 0; iteration < 80 && !flag; ++iteration) {
        footprint(samples, weights, sample_count, center, tangent, bitangent, r, multiplier, scratch);
        if (threadIdx.x == 0) {
            for (int entry = 0; entry < 11; ++entry) reduced[entry] = scratch[entry*128];
            double right[3] = {reduced[6]-reduced[10], reduced[7], reduced[8]};
            double mismatch = sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]) / reduced[10];
            evaluations = iteration + 1;
            if (mismatch <= 2e-12) flag = 5;
            else if (!solve3(reduced, right, step)) flag = 3;
            objective = reduced[9] + reduced[10] * multiplier[0];
            fraction = 1; is_accepted = 0;
        }
        __syncthreads();
        if (flag) break;
        for (int backtrack = 0; backtrack < 40; ++backtrack) {
            if (threadIdx.x == 0)
                for (int a = 0; a < 3; ++a) candidate[a] = multiplier[a] + fraction * step[a];
            __syncthreads();
            footprint(samples, weights, sample_count, center, tangent, bitangent, r, candidate, scratch);
            if (threadIdx.x == 0) {
                double gradient_dot_step = (reduced[10]-reduced[6])*step[0] - reduced[7]*step[1] - reduced[8]*step[2];
                double value = scratch[9*128] + reduced[10] * candidate[0];
                if (value <= objective + 1e-4 * fraction * gradient_dot_step + 1e-15 * reduced[10]) {
                    for (int a = 0; a < 3; ++a) multiplier[a] = candidate[a];
                    is_accepted = 1;
                } else fraction *= 0.5;
            }
            __syncthreads();
            if (is_accepted) break;
        }
        if (!is_accepted && threadIdx.x == 0) flag = 4;
        __syncthreads();
    }
    footprint(samples, weights, sample_count, center, tangent, bitangent, r, multiplier, scratch);
    if (threadIdx.x == 0) {
        double force_error = fabs(scratch[6*128] / scratch[10*128] - 1) * magnitude;
        double moment_error = sqrt(scratch[7*128]*scratch[7*128] + scratch[8*128]*scratch[8*128])
            / scratch[10*128] * magnitude * r;
        if (force_error > 1e-8*magnitude || moment_error > 1e-8*magnitude*r) flag = 4;
        diagnostics[5*index] = force_error;
        diagnostics[5*index+1] = moment_error;
        diagnostics[5*index+2] = excess;
        diagnostics[5*index+3] = allowance;
        diagnostics[5*index+4] = evaluations;
        status[index] = flag == 5 ? 0 : flag;
    }
    __syncthreads();
    if (flag && flag != 5) return;
    double integral = scratch[10*128];
    for (int q = threadIdx.x; q < sample_count; q += blockDim.x) {
        double relative[3], distance = 0, h1 = 0, h2 = 0;
        for (int a = 0; a < 3; ++a) {
            relative[a] = (samples[3*q+a] - center[a]) / r;
            distance += relative[a] * relative[a];
            h1 += relative[a] * tangent[a];
            h2 += relative[a] * bitangent[a];
        }
        if (distance < 1) {
            double weight = exp(-0.5 * distance / (0.45*0.45)) * (1-distance) * (1-distance) * weights[q];
            double pressure = weight / integral * fmax(0., 1-multiplier[0]-h1*multiplier[1]-h2*multiplier[2]);
            int face = q / quadrature, point = q % quadrature;
            for (int node = 0; node < 6; ++node)
                for (int a = 0; a < 3; ++a)
                    atomicAdd(loads + environment*dofs + 3*nodes[6*face+node]+a,
                        pressure * shape[6*point+node] * complete_force[a]);
        }
    }
}
"""


@dataclass(frozen=True)
class DeviceMappedLoads:
    nodal_force_n: cp.ndarray
    contact_status: cp.ndarray
    contact_diagnostics: cp.ndarray
    is_accepted: cp.ndarray


class PadPressureGPU:
    """Integrate changing complete contacts with one shared full-shell surface operator.

    Status codes distinguish invalid force/cone data (1), empty footprint (2), deficient quadrature rank (3) and failed
    wrench conservation (4). Inactive and exactly zero-force contacts contribute zero. Acceptance must gate observations.
    """

    def __init__(self, geometry):
        self.positions = cp.asarray(geometry.coords.reshape(-1, 3))
        self.weights = cp.asarray(geometry.integration_weights.reshape(-1))
        self.shape = cp.asarray(geometry.shape)
        self.nodes = cp.asarray(geometry.nodes, dtype=np.int64)
        self.dofs = geometry.f.ndof
        self.quadrature = len(geometry.shape)
        self.kernel = cp.RawKernel(PRESSURE_SOURCE, "pressure")

    def map(self, position, force, radius, normal, friction, valid, source_epsilon=0.0) -> DeviceMappedLoads:
        environments, contacts = valid.shape
        if (
            position.shape != (environments, contacts, 3)
            or force.shape != position.shape
            or normal.shape != position.shape
        ):
            raise ValueError("Positions, complete forces and inward normals require [environment, contact, 3]")
        if radius.shape != valid.shape or friction.shape != valid.shape or valid.dtype != np.bool_:
            raise ValueError("Radii, friction and boolean validity require matching contact shapes")
        if source_epsilon < 0 or not np.isfinite(source_epsilon):
            raise ValueError("Source roundoff epsilon must be finite and nonnegative")
        arrays = [cp.ascontiguousarray(a, dtype=np.float64) for a in (position, force, radius, normal, friction)]
        loads = cp.zeros((self.dofs, environments), dtype=np.float64, order="F")
        status = cp.zeros(valid.shape, dtype=np.int32)
        diagnostics = cp.zeros((*valid.shape, 5), dtype=np.float64)
        if contacts:
            self.kernel(
                (contacts, environments),
                (128,),
                (
                    *arrays[:3],
                    arrays[3],
                    arrays[4],
                    cp.ascontiguousarray(valid),
                    np.float64(source_epsilon),
                    self.positions,
                    self.weights,
                    self.shape,
                    self.nodes,
                    self.quadrature,
                    len(self.positions),
                    contacts,
                    self.dofs,
                    loads,
                    status,
                    diagnostics,
                ),
            )
        return DeviceMappedLoads(loads, status, diagnostics, cp.all(status == 0, axis=1))
