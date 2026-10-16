"""Measure exact device-side boundary load packing on an actual native Panda snapshot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual, kernel_peak
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_pack(
    columns: qd.Tensor, counts: qd.Tensor, stress_state: StressState, surface_inverse_info: StressSurfaceInverseInfo
):
    for i_b in range(stress_state.active.shape[0]):
        count = 0
        for j in range(surface_inverse_info.nodes.shape[0]):
            node = surface_inverse_info.nodes[j]
            force = stress_state.force[node, i_b]
            if force[0] != 0.0 or force[1] != 0.0 or force[2] != 0.0:
                columns[count, i_b] = j
                count += 1
        counts[i_b] = count


@qd.kernel(graph=True)
def kernel_sparse(
    young: float,
    omega: qd.Tensor,
    columns: qd.Tensor,
    counts: qd.Tensor,
    stress_state: StressState,
    surface_inverse_info: StressSurfaceInverseInfo,
):
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] ** 2, w[1] ** 2, w[2] ** 2, w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        value = surface_inverse_info.centrifugal[i_n] @ terms
        for slot in range(counts[i_b]):
            j = columns[slot, i_b]
            node = surface_inverse_info.nodes[j]
            value += surface_inverse_info.force_blocks[i_n, j] @ stress_state.force[node, i_b]
        stress_state.displacement[i_n, i_b] = value / young
    for i_b in range(stress_state.active.shape[0]):
        stress_state.active[i_b] = 1
        stress_state.valid[i_b] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(args.envs, varied=True)
    for _ in range(900):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    model, state = entry.model, entry.state
    columns = V(dtype=gs.qd_int, shape=(model.surface_inverse.info.nodes.shape[0], args.envs))
    counts = V(dtype=gs.qd_int, shape=(args.envs,))

    def dense():
        model.surface_inverse.apply(model.options.young, entry.omega, state, packed=False)

    def sparse():
        kernel_pack(columns, counts, state, model.surface_inverse.info)
        kernel_sparse(model.options.young, entry.omega, columns, counts, state, model.surface_inverse.info)

    dense()
    expected = qd_to_numpy(state.displacement, copy=True)
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    expected_peak = qd_to_numpy(state.peak, copy=True)
    sparse()
    actual = qd_to_numpy(state.displacement, copy=True)
    kernel_full_residual(
        model.options.young,
        model.options.tolerance,
        model.options.absolute_tolerance,
        state,
        model.info,
        only_active=False,
    )
    assert qd_to_numpy(state.valid).all()
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    peak = qd_to_numpy(state.peak, copy=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(peak, expected_peak)
    times = {}
    for name, operation in (("dense", dense), ("sparse_including_pack", sparse)):
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        times[name] = 1e3 * (time.perf_counter() - started) / args.repetitions
    result = {
        "envs": args.envs,
        "application_ms": times,
        "nonzero_boundary_nodes": qd_to_numpy(counts).tolist(),
        "displacement_max_difference_m": float(abs(actual - expected).max()),
        "global_peak_max_difference_Pa": float(abs(peak - expected_peak).max()),
        "full_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "nonzero_boundary_nodes"}), flush=True)


if __name__ == "__main__":
    main()
