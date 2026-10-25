"""Measure native complete stress responses and isolated storage/accumulation precision.

This bounded prototype uses actual native DPR loads. The complete response alone
is a lower cost bound for a production replacement: original-operator acceptance
still needs the displacement and its residual. All response preparation uses Quadrants.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.utils.array_class import V_MAT, V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json


@qd.kernel
def kernel_copy_boundary(source: StressSurfaceInverseInfo, target: StressSurfaceInverseInfo):
    for i, j in qd.ndrange(source.force_blocks.shape[0], source.force_blocks.shape[1]):
        target.force_blocks[i, j] = source.force_blocks[i, j]
    for i in range(source.centrifugal.shape[0]):
        target.centrifugal[i] = source.centrifugal[i]


@qd.kernel
def kernel_unit_displacements(young: float, state: StressState, response: StressSurfaceInverseInfo):
    for i_n, i_b in qd.ndrange(state.displacement.shape[0], state.active.shape[0]):
        value = qd.Vector.zero(gs.qd_float, 3)
        if i_b < 3 * response.nodes.shape[0]:
            for a in qd.static(range(3)):
                value[a] = response.force_blocks[i_n, i_b // 3][a, i_b % 3]
        else:
            for a in qd.static(range(3)):
                value[a] = response.centrifugal[i_n][a, i_b - 3 * response.nodes.shape[0]]
        state.displacement[i_n, i_b] = value / young


@qd.kernel
def kernel_store_response(response: qd.Tensor, state: StressState):
    for i_p, i_j in qd.ndrange(response.shape[0], response.shape[1]):
        n_stress = state.stress_tensor.shape[0] * 4
        value = qd.Vector.zero(gs.qd_float, 6)
        if i_p < n_stress:
            value = state.stress_tensor[i_p // 4, i_p % 4, i_j]
        else:
            n_resultant = state.membrane_resultant.shape[0] * 4
            i_field, i_q = (i_p - n_stress) // n_resultant, (i_p - n_stress) % n_resultant
            if i_field == 0:
                value = state.membrane_resultant[i_q // 4, i_q % 4, i_j]
            elif i_field == 1:
                value = state.bending_resultant[i_q // 4, i_q % 4, i_j]
            else:
                for a in qd.static(range(3)):
                    value[a] = state.transverse_resultant[i_q // 4, i_q % 4, i_j][a]
        response[i_p, i_j] = value


@qd.kernel
def kernel_inputs(
    omega: qd.Tensor,
    loads: qd.Tensor,
    columns: qd.Tensor,
    counts: qd.Tensor,
    state: StressState,
    response: StressSurfaceInverseInfo,
):
    for i_b in range(loads.shape[1]):
        count = 0
        w = omega[i_b]
        terms = qd.Vector([w[0] ** 2, w[1] ** 2, w[2] ** 2, w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        for j in range(loads.shape[0]):
            value = gs.qd_float(0.0)
            if j < 3 * response.nodes.shape[0]:
                node = response.nodes[j // 3]
                value = state.force[node, i_b][j % 3]
            else:
                value = terms[j - 3 * response.nodes.shape[0]]
            loads[j, i_b] = value
            if qd.abs(value) > 0.0:
                columns[count, i_b] = j
                count += 1
        counts[i_b] = count


@qd.func
def func_vm(value: qd.types.vector(6)):
    squared = 0.5 * ((value[0] - value[1]) ** 2 + (value[1] - value[2]) ** 2 + (value[2] - value[0]) ** 2)
    squared += 3.0 * (value[3] ** 2 + value[4] ** 2 + value[5] ** 2)
    return qd.sqrt(qd.max(squared, 0.0))


@qd.kernel(graph=True)
def kernel_apply(
    response: qd.Tensor,
    loads: qd.Tensor,
    columns: qd.Tensor,
    counts: qd.Tensor,
    output: qd.Tensor,
    peak: qd.Tensor,
    n_stress: int,
    n_output: int,
    sparse: qd.template(),
    accumulation: qd.template(),
    block_size: qd.template(),
):
    for i_b in range(peak.shape[0]):
        peak[i_b] = 0.0
    qd.loop_config(block_dim=block_size)
    for i_p, i_b in qd.ndrange(n_output, loads.shape[1]):
        value = qd.Vector.zero(accumulation, 6)
        n_columns = loads.shape[0]
        if qd.static(sparse):
            n_columns = counts[i_b]
        for slot in range(n_columns):
            j = slot
            if qd.static(sparse):
                j = columns[slot, i_b]
            value += qd.cast(response[i_p, j], accumulation) * qd.cast(loads[j, i_b], accumulation)
        if output.shape[0] > 0:
            output[i_p, i_b] = value
        if i_p < n_stress:
            qd.atomic_max(peak[i_b], func_vm(value))


@qd.kernel
def kernel_bounds(response: qd.Tensor, bounds: qd.Tensor, n_stress: int):
    for i_block, j in qd.ndrange(bounds.shape[0], bounds.shape[1]):
        value = gs.qd_float(0.0)
        for lane in range(32):
            i_p = 32 * i_block + lane
            if i_p < n_stress:
                value = qd.max(value, func_vm(response[i_p, j]))
        bounds[i_block, j] = value * (1.0 + 1e-10)


@qd.kernel
def kernel_screen(
    response: qd.Tensor,
    bounds: qd.Tensor,
    loads: qd.Tensor,
    columns: qd.Tensor,
    counts: qd.Tensor,
    peak: qd.Tensor,
    evaluated: qd.Tensor,
    n_stress: int,
    capacity: qd.template(),
):
    qd.loop_config(block_dim=32)
    for i_thread in range(32 * peak.shape[0]):
        lane, i_b = i_thread % 32, i_thread // 32
        sh_upper = qd.simt.block.SharedArray((capacity,), gs.qd_float)
        for chunk in range((bounds.shape[0] + 31) // 32):
            i_block = 32 * chunk + lane
            if i_block < bounds.shape[0]:
                upper = gs.qd_float(0.0)
                for slot in range(counts[i_b]):
                    j = columns[slot, i_b]
                    upper += qd.abs(loads[j, i_b]) * bounds[i_block, j]
                sh_upper[i_block] = upper * (1.0 + 1e-10)
        qd.simt.block.sync()
        maximum, n_evaluated = gs.qd_float(0.0), 0
        for _ in range(bounds.shape[0]):
            best, chosen = gs.qd_float(-1.0), bounds.shape[0]
            for chunk in range((bounds.shape[0] + 31) // 32):
                i_block = 32 * chunk + lane
                if i_block < bounds.shape[0]:
                    if sh_upper[i_block] > best:
                        best, chosen = sh_upper[i_block], i_block
            upper = qd.simt.subgroup.reduce_all_max(best)
            chosen = qd.simt.subgroup.reduce_all_min(qd.select(best >= upper, chosen, bounds.shape[0]))
            if upper <= maximum:
                break
            i_p = chosen * 32 + lane
            value = qd.Vector.zero(gs.qd_float, 6)
            if i_p < n_stress:
                for slot in range(counts[i_b]):
                    j = columns[slot, i_b]
                    value += response[i_p, j] * loads[j, i_b]
            maximum = qd.max(maximum, qd.simt.subgroup.reduce_all_max(func_vm(value)))
            n_evaluated += 1
            if lane == 0:
                sh_upper[chosen] = -1.0
            qd.simt.block.sync()
        if lane == 0:
            peak[i_b], evaluated[i_b] = maximum, n_evaluated


def timings(operation, repeats, steps):
    operation()
    qd.sync()
    rows = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(steps):
            operation()
        qd.sync()
        rows.append((time.perf_counter() - started) / steps)
    return rows


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), default="shell")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=130)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--packed-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        identity=runtime_identity(),
        settings=vars(args) | {"conditions": str(args.conditions), "output": str(args.output)},
        rows=[],
        complete=False,
    )
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    started = time.perf_counter()
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
    for _ in range(args.warmup):
        workload.step()
    qd.sync()
    result["scene_preparation_seconds"] = time.perf_counter() - started
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    model, state = entry.model, entry.state
    assert model.surface_inverse is not None
    boundary = model.surface_inverse.info
    model.recover(entry.omega, state, surface_load=True)
    assert qd_to_numpy(state.valid).all()
    # Match the native field order (element, quadrature, environment).
    expected = qd_to_numpy(state.stress_tensor).reshape((-1, args.envs, 6))
    expected_peak = qd_to_numpy(state.peak)
    expected_stress = qd_to_numpy(state.stress_tensor)
    boundary_single = StressSurfaceInverseInfo(
        nodes=boundary.nodes,
        force_blocks=V_MAT(3, 3, dtype=qd.f32, shape=boundary.force_blocks.shape),
        centrifugal=V_MAT(3, 6, dtype=qd.f32, shape=boundary.centrifugal.shape),
    )
    kernel_copy_boundary(boundary, boundary_single)
    result["packed_response"] = []
    for storage, current in ((64, boundary), (32, boundary_single)):
        model.surface_inverse.info = current
        samples = timings(lambda: model.recover(entry.omega, state, surface_load=True), args.repeats, args.steps)
        residual = np.sqrt(qd_to_numpy(state.residual_norm_squared))
        rhs = np.sqrt(qd_to_numpy(state.rhs_norm_squared))
        field_error = relative(qd_to_numpy(state.stress_tensor), expected_stress)
        peak_error = relative(qd_to_numpy(state.peak), expected_peak)
        row = dict(
            storage=storage,
            accumulation=64,
            factor_and_correction_inverse_storage=64,
            seconds=samples,
            complete_residual_N=residual.tolist(),
            rhs_norm_N=rhs.tolist(),
            field_relative=field_error,
            peak_relative=peak_error,
            corrections=qd_to_numpy(state.corrections).tolist(),
            fallbacks=qd_to_numpy(state.fallbacks).tolist(),
        )
        result["packed_response"].append(row)
        write_json(args.output / "result.json", result)
        assert qd_to_numpy(state.valid).all()
        assert (residual <= np.maximum(entry.options.absolute_tolerance, entry.options.tolerance * rhs)).all()
        assert max(field_error, peak_error) < 1e-4
    model.surface_inverse.info = boundary
    model.recover(entry.omega, state, surface_load=True)
    if args.packed_only:
        result["complete"] = True
        write_json(args.output / "result.json", result)
        return
    n_stress = expected.shape[0]
    for field in (state.membrane_resultant, state.bending_resultant, state.transverse_resultant):
        if field.shape[0]:
            values = qd_to_numpy(field)
            values = values.reshape((-1, args.envs, values.shape[-1]))
            if values.shape[-1] == 3:
                values = np.pad(values, ((0, 0), (0, 0), (0, 3)))
            expected = np.concatenate((expected, values))
    n_inputs = 3 * boundary.nodes.shape[0] + 6
    loads = V(dtype=gs.qd_float, shape=(n_inputs, args.envs))
    columns = V(dtype=gs.qd_int, shape=(n_inputs, args.envs))
    counts = V(dtype=gs.qd_int, shape=(args.envs,))
    peak = V(dtype=gs.qd_float, shape=(args.envs,))
    evaluated = V(dtype=gs.qd_int, shape=(args.envs,))
    output = V_VEC(6, dtype=gs.qd_float, shape=(expected.shape[0], args.envs))
    empty = V_VEC(6, dtype=gs.qd_float, shape=(0, args.envs))
    kernel_inputs(entry.omega, loads, columns, counts, state, boundary)
    result["nonzero_input_columns"] = qd_to_numpy(counts).tolist()
    result["native_recovery_seconds"] = timings(
        lambda: model.recover(entry.omega, state, surface_load=True), args.repeats, args.steps
    )
    started = time.perf_counter()
    units = model.create_state(n_inputs, "full")
    kernel_unit_displacements(entry.options.young, units, boundary)
    model.evaluate_stress(entry.options, units)
    qd.sync()
    result["unit_output_preparation_seconds"] = time.perf_counter() - started
    result["response_shape"] = [expected.shape[0], n_inputs, 6]
    for storage in (64, 32):
        started = time.perf_counter()
        response = V_VEC(6, dtype=gs.qd_float if storage == 64 else qd.f32, shape=(expected.shape[0], n_inputs))
        kernel_store_response(response, units)
        qd.sync()
        result[f"response_store_{storage}_seconds"] = time.perf_counter() - started
        result[f"response_{storage}_bytes"] = expected.shape[0] * n_inputs * 6 * (storage // 8)
        for single in (False, True):
            for sparse in (True, False) if args.dense else (True,):
                for mode in ("full", "max"):
                    for block in (128, 256):
                        operation = partial(
                            kernel_apply,
                            response,
                            loads,
                            columns,
                            counts,
                            output if mode == "full" else empty,
                            peak,
                            n_stress,
                            expected.shape[0] if mode == "full" else n_stress,
                            sparse,
                            qd.f32 if single else gs.qd_float,
                            block,
                        )
                        samples = timings(operation, args.repeats, args.steps)
                        peak_error = relative(qd_to_numpy(peak), expected_peak)
                        field_error = relative(qd_to_numpy(output), expected) if mode == "full" else None
                        row = dict(
                            storage=storage,
                            accumulation=32 if single else 64,
                            sparse=sparse,
                            mode=mode,
                            block=block,
                            seconds=samples,
                            peak_relative=peak_error,
                            field_relative=field_error,
                            budget_passed=peak_error < 1e-4 and (field_error is None or field_error < 1e-4),
                        )
                        result["rows"].append(row)
                        write_json(args.output / "result.json", result)
        if storage == 64:
            n_blocks = (n_stress + 31) // 32
            capacity = next(size for size in (64, 128, 256, 512) if n_blocks <= size)
            bounds = V(dtype=gs.qd_float, shape=(n_blocks, n_inputs))
            kernel_bounds(response, bounds, n_stress)
            samples = timings(
                lambda: kernel_screen(response, bounds, loads, columns, counts, peak, evaluated, n_stress, capacity),
                args.repeats,
                args.steps,
            )
            error = relative(qd_to_numpy(peak), expected_peak)
            result["screen"] = dict(
                seconds=samples,
                peak_relative=error,
                evaluated_blocks=qd_to_numpy(evaluated).tolist(),
                total_blocks=n_blocks,
                bounds_bytes=n_blocks * n_inputs * 8,
                relative_bound_inflation=1e-10,
            )
            assert error < 1e-8
            write_json(args.output / "result.json", result)
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
