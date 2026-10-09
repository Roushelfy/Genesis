"""Validate the real shared sparse GPU baseline against identical CPU FP64 solves.

Run with --output pointing to the project data directory. This checks synthetic mechanics and GPU arithmetic; it does
not claim live grasp acceptance, physical mesh convergence or environment transition throughput.
"""

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .cpu import EggConfig, EggRecoveryCPU, reference
from .sparse_gpu import EggRecoveryGPU, SharedSparseFactor


@dataclass(frozen=True)
class GPUCheck:
    environments: int
    right_hand_sides: int
    maximum_full_relative_residual: float
    maximum_peak_relative_error: float
    maximum_absolute_peak_error_pa: float
    zero_peak_pa: float
    zero_residual_n: float
    freefall_peak_pa: float
    centrifugal_peak_pa: float


def permutation_check() -> None:
    rng = np.random.default_rng(9173)
    matrix = sparse.random(41, 41, density=0.25, random_state=rng, format="csc")
    matrix += sparse.diags(rng.uniform(0.01, 0.02, 41))
    factor = splu(matrix)
    assert np.any(factor.perm_r != factor.perm_c)
    gpu_factor = SharedSparseFactor(factor)
    for columns in (1, 3, 8, 17):
        rhs = rng.normal(size=(41, columns))
        expected = factor.solve(rhs)
        for order in ("C", "F"):
            observed = cp.asnumpy(gpu_factor.solve(cp.array(rhs, order=order)))
            np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
        strided = cp.zeros((41, 2 * columns))
        strided[:, ::2] = cp.asarray(rhs)
        observed = cp.asnumpy(gpu_factor.solve(strided[:, ::2]))
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
    assert gpu_factor.solve(cp.zeros((41, 0))).shape == (41, 0)


def mechanics_check(model: EggRecoveryCPU, environments: int) -> GPUCheck:
    fem = model.fem
    gpu = EggRecoveryGPU(fem, environments)
    rng = np.random.default_rng(3913 + environments)
    raw = rng.normal(size=(fem.ndof, environments))
    omega = rng.uniform(-5, 5, (environments, 3))
    relative = fem.xyz - fem.com
    centrifugal = np.cross(omega[:, None], np.cross(omega[:, None], relative))
    rhs = fem.balance(raw - fem.m @ centrifugal.reshape(environments, -1).T)
    device_rhs = gpu.compatible_rhs(cp.asarray(raw), cp.asarray(omega))
    np.testing.assert_allclose(cp.asnumpy(device_rhs), rhs, rtol=1e-11, atol=1e-12)
    expected = np.zeros_like(rhs)
    expected[fem.free] = fem.factor.solve(rhs[fem.free])
    expected_peak = np.array([model.peak(expected[:, i_b])[0] for i_b in range(environments)])
    result = gpu.recover(cp.asarray(rhs))
    np.testing.assert_array_equal(cp.asnumpy(result.is_accepted), True)
    peak = cp.asnumpy(result.peak_pa)
    errors = abs(peak - expected_peak) / np.maximum(expected_peak, 1)
    assert np.max(errors) <= 1e-4
    np.testing.assert_allclose(cp.asnumpy(result.displacement_m), expected, rtol=1e-6, atol=1e-11)
    full_residual = np.linalg.norm(fem.k @ cp.asnumpy(result.displacement_m) - rhs, axis=0)
    relative_residual = full_residual / np.linalg.norm(rhs, axis=0)
    assert np.max(relative_residual) <= 1e-6
    np.testing.assert_allclose(cp.asnumpy(result.absolute_residual_n), full_residual, rtol=0.03, atol=1e-9)
    zero = gpu.recover(cp.zeros_like(device_rhs))
    np.testing.assert_array_equal(cp.asnumpy(zero.peak_pa), 0)
    np.testing.assert_array_equal(cp.asnumpy(zero.displacement_m), 0)
    invalid_rhs = cp.zeros_like(device_rhs)
    invalid_rhs[0] = cp.nan
    rejected = gpu.recover(invalid_rhs)
    np.testing.assert_array_equal(cp.asnumpy(rejected.is_accepted), False)
    gravity = (fem.m @ np.tile([0, 0, -9.81], len(fem.xyz)))[:, None]
    freefall = gpu.recover(gpu.compatible_rhs(cp.asarray(np.repeat(gravity, environments, axis=1))))
    assert cp.asnumpy(freefall.peak_pa).max() < 1e-3
    spin = gpu.recover(gpu.compatible_rhs(cp.zeros_like(device_rhs), cp.asarray(omega)))
    assert cp.asnumpy(spin.peak_pa).min() > 1
    # A strain patch and arbitrary fields exercise the peak independently of factorization
    displacement = np.einsum("ij,nj->ni", rng.normal(size=(3, 3)) * 1e-6, fem.xyz).reshape(fem.ndof, 1)
    fields = np.column_stack((displacement, rng.normal(scale=1e-8, size=(fem.ndof, environments))))
    field_peak = cp.asnumpy(gpu.peak(cp.asarray(fields)))
    expected_field_peak = np.array([reference.peak(fem, fields[:, i_b]) for i_b in range(fields.shape[1])])
    np.testing.assert_allclose(field_peak, expected_field_peak, rtol=1e-10, atol=1e-6)
    return GPUCheck(
        environments,
        environments,
        float(relative_residual.max()),
        float(errors.max()),
        float(abs(peak - expected_peak).max()),
        float(cp.asnumpy(zero.peak_pa).max()),
        float(cp.asnumpy(zero.absolute_residual_n).max()),
        float(cp.asnumpy(freefall.peak_pa).max()),
        float(cp.asnumpy(spin.peak_pa).min()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    with threadpool_limits(limits=1):
        permutation_check()
        model = EggRecoveryCPU(config=EggConfig(level=args.level), direct=True, history=0)
        checks = [asdict(mechanics_check(model, environments)) for environments in (1, 3, 8, 17)]
    cp.cuda.get_current_stream().synchronize()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    result = {
        "passed": True,
        "scope": "Synthetic full-shell mechanics and shared sparse CUDA direct baseline; no live scene or throughput",
        "gpu_name": properties["name"].decode(),
        "compute_capability": [properties["major"], properties["minor"]],
        "vram_bytes": properties["totalGlobalMem"],
        "driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cupy": cp.__version__,
        "platform": platform.platform(),
        "mesh": asdict(model.config),
        "dofs": model.fem.ndof,
        "tetrahedra": model.fem.ne,
        "factor_nonzeros": model.fem.factor.L.nnz + model.fem.factor.U.nnz,
        "physical_mesh_converged": False,
        "checks": checks,
        "validation_seconds_including_build": perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
