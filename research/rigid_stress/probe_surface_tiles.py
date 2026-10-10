"""Compare FP64 shared-memory operator tiles against native complete-boundary application."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_balance, kernel_full_residual, kernel_peak
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy


@qd.kernel(graph=True)
def kernel_surface_tiled(
    young: float,
    omega: qd.Tensor,
    stress_state: StressState,
    surface_inverse_info: StressSurfaceInverseInfo,
    source_tile: qd.template(),
):
    qd.loop_config(block_dim=128)
    for i_thread in range(((stress_state.rhs.shape[0] + 3) // 4) * ((stress_state.active.shape[0] + 31) // 32) * 128):
        thread = qd.simt.block.thread_idx()
        block = i_thread // 128
        n_env_tiles = (stress_state.active.shape[0] + 31) // 32
        row = thread // 32
        i_n = (block // n_env_tiles) * 4 + row
        i_b = (block % n_env_tiles) * 32 + thread % 32
        sh_operator = qd.simt.block.SharedArray((4, source_tile, 3, 3), gs.qd_float)
        value = qd.Vector.zero(gs.qd_float, 3)
        if i_n < stress_state.rhs.shape[0] and i_b < stress_state.active.shape[0]:
            w = omega[i_b]
            terms = qd.Vector([w[0] ** 2, w[1] ** 2, w[2] ** 2, w[0] * w[1], w[0] * w[2], w[1] * w[2]])
            value = surface_inverse_info.centrifugal[i_n] @ terms
        for start in range((surface_inverse_info.nodes.shape[0] + source_tile - 1) // source_tile):
            for slot in range((4 * source_tile + 127) // 128):
                item = thread + slot * 128
                j_row = item // source_tile
                j_col = item % source_tile
                j_n = (block // n_env_tiles) * 4 + j_row
                j = start * source_tile + j_col
                if item < 4 * source_tile:
                    for a, b in qd.static(qd.ndrange(3, 3)):
                        sh_operator[j_row, j_col, a, b] = 0.0
                        if j_n < stress_state.rhs.shape[0] and j < surface_inverse_info.nodes.shape[0]:
                            sh_operator[j_row, j_col, a, b] = surface_inverse_info.force_blocks[j_n, j][a, b]
            qd.simt.block.sync()
            if i_n < stress_state.rhs.shape[0] and i_b < stress_state.active.shape[0]:
                for col in range(source_tile):
                    j = start * source_tile + col
                    if j < surface_inverse_info.nodes.shape[0]:
                        force = stress_state.force[surface_inverse_info.nodes[j], i_b]
                        for a, b in qd.static(qd.ndrange(3, 3)):
                            value[a] += sh_operator[row, col, a, b] * force[b]
            qd.simt.block.sync()
        if i_n < stress_state.rhs.shape[0] and i_b < stress_state.active.shape[0]:
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
    model = StressModel(RigidStressOptions(mesh=Path("examples/rigid/assets/hollow_egg/level1/elastic.npz")))
    state = model.create_state(args.envs)
    random = np.random.default_rng(10917)
    force = np.zeros((model.info.vertices.shape[0], args.envs, 3))
    nodes = qd_to_numpy(model.surface_inverse.info.nodes)
    force[nodes] = random.normal(size=(len(nodes), args.envs, 3)) * 0.01
    state.force.from_numpy(force)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    omega.from_numpy(random.normal(size=(args.envs, 3)) * 2)
    kernel_balance(omega, state, model.info)
    model.surface_inverse.apply(model.options.young, omega, state)
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    expected = qd_to_numpy(state.displacement, copy=True)
    expected_peak = qd_to_numpy(state.peak, copy=True)
    reports = []
    operations = [("native", lambda: model.surface_inverse.apply(model.options.young, omega, state))]
    for tile in (8, 16, 32, 64):
        operations.append(
            (
                f"shared-{tile}",
                lambda t=tile: kernel_surface_tiled(model.options.young, omega, state, model.surface_inverse.info, t),
            )
        )
    for name, operation in operations:
        operation()
        kernel_full_residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            only_active=False,
        )
        kernel_peak(model.options.young, model.options.poisson, state, model.info)
        actual = qd_to_numpy(state.displacement, copy=True)
        peak = qd_to_numpy(state.peak, copy=True)
        valid = qd_to_numpy(state.valid)
        assert valid.all(), name
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-15)
        np.testing.assert_allclose(peak, expected_peak, rtol=1e-4, atol=1e-3)
        operation()
        qd.sync()
        started = time.perf_counter()
        for _ in range(args.repetitions):
            operation()
        qd.sync()
        reports.append(
            {
                "name": name,
                "milliseconds": 1e3 * (time.perf_counter() - started) / args.repetitions,
                "peak_error_Pa": float(abs(peak - expected_peak).max()),
                "complete_residual_max_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "reports": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
