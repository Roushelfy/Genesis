"""Test two-frame prediction with accepted results feeding subsequent native history."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.solve import kernel_full_residual
from genesis.utils.array_class import V_VEC, V
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .check_response_candidates import timings


@qd.kernel
def kernel_predict(
    previous: qd.Tensor,
    counts: qd.Tensor,
    heads: qd.Tensor,
    state: StressState,
    reference: StressState,
):
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.active.shape[0]):
        state.force[i_n, i_b] = reference.force[i_n, i_b]
        state.rhs[i_n, i_b] = reference.rhs[i_n, i_b]
        value = qd.Vector.zero(gs.qd_float, 3)
        if counts[i_b] == 2:
            i_head = heads[i_b]
            value = 2.0 * previous[i_n, i_b, i_head] - previous[i_n, i_b, 1 - i_head]
        state.displacement[i_n, i_b] = value
    for i_b in range(state.active.shape[0]):
        state.rhs_norm_squared[i_b] = reference.rhs_norm_squared[i_b]
        state.valid[i_b] = True
        state.step_peak[i_b] = 0.0


@qd.kernel
def kernel_gate(counts: qd.Tensor, hits: qd.Tensor, state: StressState):
    for i_b in range(state.active.shape[0]):
        state.valid[i_b] = state.valid[i_b] and counts[i_b] == 2
        hits[i_b] = state.valid[i_b]


@qd.kernel(graph=True)
def kernel_append(
    previous: qd.Tensor,
    counts: qd.Tensor,
    heads: qd.Tensor,
    state: StressState,
    rotate: qd.template(),
):
    for i_n, i_b in qd.ndrange(state.force.shape[0], state.active.shape[0]):
        slot = 0
        if qd.static(rotate):
            slot = 1 - heads[i_b]
        else:
            previous[i_n, i_b, 1] = previous[i_n, i_b, 0]
        previous[i_n, i_b, slot] = state.displacement[i_n, i_b]
    for i_b in range(state.active.shape[0]):
        if qd.static(rotate):
            heads[i_b] = 1 - heads[i_b]
        counts[i_b] = qd.min(2, counts[i_b] + 1)


@qd.kernel
def kernel_reset(i_b: int, counts: qd.Tensor):
    for _ in range(1):
        counts[i_b] = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), default="shell")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(identity=runtime_identity(), complete=False, budget=1e-5, rows=[])
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
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    state = entry.model.create_state(args.envs, "full")
    previous = V_VEC(3, dtype=gs.qd_float, shape=(*state.force.shape, 2))
    counts = V(dtype=gs.qd_int, shape=(args.envs,))
    heads = V(dtype=gs.qd_int, shape=(args.envs,))
    hits = V(dtype=gs.qd_bool, shape=(args.envs,))
    counts.fill(0)
    heads.fill(0)
    options = entry.options
    snapshots = 0
    for step in range(args.steps):
        if step == 250 and args.envs > 1:
            workload.reset(np.array([1]))
            kernel_reset(1, counts)
        workload.step()
        workload.scene.rigid_solver.check_errno()
        kernel_predict(previous, counts, heads, state, entry.state)
        kernel_full_residual(
            options.young,
            options.tolerance,
            options.absolute_tolerance,
            state,
            entry.model.info,
            False,
            args.envs >= 8192,
        )
        kernel_gate(counts, hits, state)
        entry.model.solve(options, state, entry.omega, surface_load=True, only_failed=True)
        kernel_full_residual(
            options.young,
            options.tolerance,
            options.absolute_tolerance,
            state,
            entry.model.info,
            True,
            args.envs >= 8192,
        )
        entry.model.evaluate_stress(options, state)
        assert qd_to_numpy(state.valid).all()
        actual = qd_to_numpy(state.stress_tensor, transpose=True)
        expected = qd_to_numpy(entry.state.stress_tensor, transpose=True)
        error = np.linalg.norm((actual - expected).reshape((args.envs, -1)), axis=1) / np.maximum(
            np.linalg.norm(expected.reshape((args.envs, -1)), axis=1), 1e-30
        )
        peak, reference_peak = qd_to_numpy(state.peak), qd_to_numpy(entry.state.peak)
        peak_error = np.abs(peak - reference_peak) / np.maximum(np.abs(reference_peak), 1e-30)
        admitted = qd_to_numpy(hits)
        loaded = np.linalg.norm(qd_to_numpy(entry.state.force, transpose=True).reshape((args.envs, -1)), axis=1) > 1e-12
        row = dict(
            step=step,
            predicted=admitted.tolist(),
            loaded=loaded.tolist(),
            field_relative=error.tolist(),
            peak_relative=peak_error.tolist(),
            peak_absolute_Pa=np.abs(peak - reference_peak).tolist(),
        )
        result["rows"].append(row)
        if max(error.max(), peak_error.max()) > 1e-5 and snapshots < 3:
            np.savez_compressed(
                args.output / f"field-failure-{step:04d}.npz", actual=actual, expected=expected, prediction=admitted
            )
            snapshots += 1
        # Every accepted prediction is stored, so later extrapolation includes its numerical error.
        kernel_append(previous, counts, heads, state, True)
        if step % 50 == 0:
            write_json(args.output / "result.json", result)
    result["rotation_seconds"] = timings(partial(kernel_append, previous, counts, heads, state, True), 5, 100)
    result["copy_seconds"] = timings(partial(kernel_append, previous, counts, heads, state, False), 5, 100)
    result["complete"] = True
    result["budget_passed"] = all(max(row["field_relative"] + row["peak_relative"]) <= 1e-5 for row in result["rows"])
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
