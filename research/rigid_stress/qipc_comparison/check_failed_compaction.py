"""Compare device compaction with masked exact recovery at controlled failure fractions."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual
from genesis.engine.solvers.rigid.stress.surface_inverse import (
    StressSurfaceInverseInfo,
    func_surface_apply_packed,
    func_surface_pack,
)
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .check_response_candidates import relative, timings
from .check_two_frame_prediction import kernel_append


@qd.kernel
def kernel_product(young: float, product: qd.Tensor, state: StressState, info: StressInfo):
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        value = qd.Vector.zero(gs.qd_float, 3)
        for entry in range(info.row_start[i_n], info.row_start[i_n + 1]):
            j_n = info.columns[entry]
            value += young * info.stiffness[entry] @ state.displacement[j_n, i_b]
        product[i_n, i_b] = value


@qd.kernel(graph=True)
def kernel_cached_residual(product: qd.Tensor, state: StressState):
    for i_b in range(state.active.shape[0]):
        state.residual_norm_squared[i_b] = 0.0
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        residual = state.rhs[i_n, i_b] - product[i_n, i_b]
        state.residual[i_n, i_b] = residual
        qd.atomic_add(state.residual_norm_squared[i_b], residual.norm_sqr())


def refreshed_residual(entry, product):
    kernel_product(entry.options.young, product, entry.state, entry.model.info)
    kernel_cached_residual(product, entry.state)


def maintenance(entry, result):
    state, options = entry.state, entry.options
    product = V_VEC(3, dtype=gs.qd_float, shape=state.displacement.shape)
    fresh = partial(
        kernel_full_residual,
        options.young,
        options.tolerance,
        options.absolute_tolerance,
        state,
        entry.model.info,
        False,
        state.active.shape[0] >= 8192,
    )
    result["fresh_residual_seconds"] = timings(fresh, 5, 50)
    before = qd_to_numpy(state.residual, transpose=True)
    kernel_product(options.young, product, state, entry.model.info)
    result["cached_residual_seconds"] = timings(partial(kernel_cached_residual, product, state), 5, 50)
    difference = qd_to_numpy(state.residual, transpose=True) - before
    result["cached_vs_fresh_absolute_residual_N"] = float(np.abs(difference).max())
    assert float(np.abs(difference).max()) < options.absolute_tolerance
    result["cache_refresh_seconds"] = timings(
        partial(kernel_product, options.young, product, state, entry.model.info), 5, 50
    )
    result["refresh_and_gate_seconds"] = timings(partial(refreshed_residual, entry, product), 5, 50)
    result["cached_kq_bytes"] = int(np.prod(state.displacement.shape)) * 3 * 8
    result["cached_kq_scope"] = (
        "Actual CSR operator product, frozen displacement. A changed accepted displacement requires refresh. No stale cache is admitted or used in production."
    )
    previous = V_VEC(3, dtype=gs.qd_float, shape=(*state.force.shape, 2))
    counts = V(dtype=gs.qd_int, shape=(state.active.shape[0],))
    heads = V(dtype=gs.qd_int, shape=(state.active.shape[0],))
    counts.fill(0)
    heads.fill(0)
    result["rotation_seconds"] = timings(partial(kernel_append, previous, counts, heads, state, True), 5, 100)
    result["copy_seconds"] = timings(partial(kernel_append, previous, counts, heads, state, False), 5, 100)
    tokens = ("tf32", "wmma", "mma_sync", "tensorcore")
    api_hits = []
    for path in Path(qd.__file__).parent.rglob("*.py"):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if any(token in line.lower() for token in tokens):
                api_hits.append(dict(path=str(path), line=index + 1, text=line.strip()[:240]))
    result["precision_api_probe"] = dict(
        quadrants_file=qd.__file__,
        python_source_hits=api_hits,
        tested_accumulations=["FP64", "FP32"],
        tf32_executed=False,
    )


@qd.func
def func_mask(n_failed: int, state: StressState):
    for i_b in range(state.valid.shape[0]):
        state.valid[i_b] = (i_b * 4051 + 17) % state.valid.shape[0] >= n_failed


@qd.kernel
def kernel_prepare_native(n_failed: int, state: StressState):
    func_mask(n_failed, state)
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.valid.shape[0]):
        if not state.valid[i_b]:
            state.displacement[i_n, i_b] = qd.Vector([1.0, -2.0, 3.0])


def native_case(n_failed, entry):
    kernel_prepare_native(n_failed, entry.state)
    entry.model.surface_inverse.apply(entry.options.young, entry.omega, entry.state, only_failed=True)


@qd.kernel(graph=True)
def kernel_masked(
    n_failed: int,
    young: float,
    omega: qd.Tensor,
    state: StressState,
    response: StressSurfaceInverseInfo,
    block_size: qd.template(),
):
    func_mask(n_failed, state)
    func_surface_pack(state, response, True)
    func_surface_apply_packed(young, omega, state, response, block_size, True)


@qd.kernel(graph=True)
def kernel_compacted(
    n_failed: int,
    young: float,
    omega: qd.Tensor,
    indices: qd.Tensor,
    count: qd.Tensor,
    state: StressState,
    response: StressSurfaceInverseInfo,
    block_size: qd.template(),
):
    func_mask(n_failed, state)
    func_surface_pack(state, response, True)
    for _ in range(1):
        count[None] = 0
    for i_b in range(state.valid.shape[0]):
        state.active[i_b] = gs.qd_int(not state.valid[i_b])
        if not state.valid[i_b]:
            slot = qd.atomic_add(count[None], 1)
            indices[slot] = i_b
        state.valid[i_b] = True
    qd.loop_config(block_dim=block_size)
    for i_task in range(state.displacement.shape[0] * count[None]):
        i_n, slot = i_task // count[None], i_task % count[None]
        i_b = indices[slot]
        w = omega[i_b]
        terms = qd.Vector([w[0] ** 2, w[1] ** 2, w[2] ** 2, w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        value = response.centrifugal[i_n] @ terms
        for i_column in range(state.boundary_count[i_b]):
            j = state.boundary_columns[i_column, i_b]
            node = response.nodes[j]
            value += response.force_blocks[i_n, j] @ state.force[node, i_b]
        state.displacement[i_n, i_b] = value / young


def run_case(operation, entry):
    operation()
    options = entry.options
    kernel_full_residual(
        options.young,
        options.tolerance,
        options.absolute_tolerance,
        entry.state,
        entry.model.info,
        True,
        entry.state.valid.shape[0] >= 8192,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), default="shell")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maintenance-only", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(identity=runtime_identity(), complete=False, rows=[])
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        conditions=args.conditions,
        output_mode="full",
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=16,
        box_edge_policy="planar",
        stress_backend=args.stress_backend,
    )
    for _ in range(130):
        workload.step()
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    entry.model.recover(entry.omega, entry.state, surface_load=True)
    expected = qd_to_numpy(entry.state.displacement, transpose=True)
    expected_tensor = qd_to_numpy(entry.state.stress_tensor, transpose=True)
    if args.maintenance_only:
        maintenance(entry, result)
        result["complete"] = True
        write_json(args.output / "result.json", result)
        return
    indices = V(dtype=gs.qd_int, shape=(args.envs,))
    count = V(dtype=gs.qd_int, shape=())
    response = entry.model.surface_inverse.info
    result["workspace_bytes"] = (args.envs + 1) * 4
    for fraction in [0.0, 0.01, 0.1, 0.5, 1.0]:
        n_failed = int(args.envs * fraction)
        if args.native:
            operation = partial(native_case, n_failed, entry)
            samples = timings(partial(run_case, operation, entry), 5, 30)
            actual = qd_to_numpy(entry.state.displacement, transpose=True)
            failed = (np.arange(args.envs) * 4051 + 17) % args.envs < n_failed
            np.testing.assert_array_equal(actual[~failed], expected[~failed])
            error = relative(actual, expected)
            assert error < 1e-10
            assert qd_to_numpy(entry.state.valid).all()
            if args.envs >= 1024:
                assert int(qd_to_numpy(entry.state.failed_count)) == n_failed
                ids = qd_to_numpy(entry.state.failed_environments)[:n_failed]
                np.testing.assert_array_equal(np.sort(ids), np.flatnonzero(failed))
            entry.model.evaluate_stress(entry.options, entry.state)
            tensor = qd_to_numpy(entry.state.stress_tensor, transpose=True)
            field_error = relative(tensor, expected_tensor)
            assert field_error < 1e-8
            result["rows"].append(
                dict(
                    native=True,
                    failure_count=n_failed,
                    fraction=n_failed / args.envs,
                    seconds=samples,
                    displacement_relative=error,
                    field_relative=field_error,
                    untouched_environments_equal=True,
                    failed_displacements_poisoned=True,
                )
            )
            write_json(args.output / "result.json", result)
            continue
        for compact in [False, True]:
            for block_size in [128, 256, 512]:
                operation = (
                    partial(
                        kernel_compacted,
                        n_failed,
                        entry.options.young,
                        entry.omega,
                        indices,
                        count,
                        entry.state,
                        response,
                        block_size,
                    )
                    if compact
                    else partial(
                        kernel_masked, n_failed, entry.options.young, entry.omega, entry.state, response, block_size
                    )
                )
                samples = timings(partial(run_case, operation, entry), 5, 30)
                actual = qd_to_numpy(entry.state.displacement, transpose=True)
                error = relative(actual, expected)
                failed = (np.arange(args.envs) * 4051 + 17) % args.envs < n_failed
                np.testing.assert_array_equal(actual[~failed], expected[~failed])
                assert error < 1e-10
                assert qd_to_numpy(entry.state.valid).all()
                residual = np.sqrt(qd_to_numpy(entry.state.residual_norm_squared))
                rhs = np.sqrt(qd_to_numpy(entry.state.rhs_norm_squared))
                assert (residual <= np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * rhs)).all()
                result["rows"].append(
                    dict(
                        failure_count=n_failed,
                        fraction=n_failed / args.envs,
                        compact=compact,
                        block_size=block_size,
                        seconds=samples,
                        displacement_relative=error,
                        untouched_environments_equal=True,
                    )
                )
                write_json(args.output / "result.json", result)
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
