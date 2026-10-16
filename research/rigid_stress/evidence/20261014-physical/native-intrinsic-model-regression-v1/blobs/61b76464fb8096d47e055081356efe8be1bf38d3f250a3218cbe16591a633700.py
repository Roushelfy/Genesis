"""Measure boundary operator precision including required residual corrections and fallback."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressState
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import (
    kernel_balance,
    kernel_direct_init,
    kernel_full_residual,
    kernel_peak,
)
from genesis.engine.solvers.rigid.stress.surface_inverse import StressSurfaceInverseInfo
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_MAT, V_VEC
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_quantize(high: qd.Tensor, low: qd.Tensor, surface_inverse_info: StressSurfaceInverseInfo):
    for i, j in qd.ndrange(high.shape[0], high.shape[1]):
        value = surface_inverse_info.force_blocks[i, j]
        high[i, j] = value
        low[i, j] = value - high[i, j].cast(gs.qd_float)


@qd.kernel(graph=True)
def kernel_candidate(
    young: float,
    omega: qd.Tensor,
    stress_state: StressState,
    surface_inverse_info: StressSurfaceInverseInfo,
    high: qd.Tensor,
    low: qd.Tensor,
    arithmetic32: qd.template(),
    paired: qd.template(),
):
    for i_n, i_b in qd.ndrange(stress_state.rhs.shape[0], stress_state.active.shape[0]):
        w = omega[i_b]
        terms = qd.Vector([w[0] ** 2, w[1] ** 2, w[2] ** 2, w[0] * w[1], w[0] * w[2], w[1] * w[2]])
        value = surface_inverse_info.centrifugal[i_n] @ terms
        if qd.static(arithmetic32):
            total = qd.Vector.zero(qd.f32, 3)
            compensation = qd.Vector.zero(qd.f32, 3)
            for j in range(surface_inverse_info.nodes.shape[0]):
                node = surface_inverse_info.nodes[j]
                term = high[i_n, j] @ stress_state.force[node, i_b].cast(qd.f32)
                corrected = term - compensation
                updated = total + corrected
                compensation = (updated - total) - corrected
                total = updated
            value += total.cast(gs.qd_float)
        else:
            for j in range(surface_inverse_info.nodes.shape[0]):
                node = surface_inverse_info.nodes[j]
                block = high[i_n, j].cast(gs.qd_float)
                if qd.static(paired):
                    block += low[i_n, j].cast(gs.qd_float)
                value += block @ stress_state.force[node, i_b]
        stress_state.displacement[i_n, i_b] = value / young
    for i_b in range(stress_state.active.shape[0]):
        stress_state.active[i_b] = 1
        stress_state.valid[i_b] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=50)
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
    kernel_direct_init(state)
    high = V_MAT(3, 3, dtype=qd.f32, shape=model.surface_inverse.info.force_blocks.shape)
    low = V_MAT(3, 3, dtype=qd.f32, shape=high.shape)
    kernel_quantize(high, low, model.surface_inverse.info)
    model.surface_inverse.apply(model.options.young, omega, state)
    kernel_peak(model.options.young, model.options.poisson, state, model.info)
    expected_peak = qd_to_numpy(state.peak, copy=True)

    def residual(only_active=False):
        kernel_full_residual(
            model.options.young,
            model.options.tolerance,
            model.options.absolute_tolerance,
            state,
            model.info,
            only_active,
        )

    def checked(operation):
        operation()
        residual()
        for _ in range(model.options.inverse_corrections):
            model.inverse.apply(model.options.young, state, correction=True)
            residual(True)
        model.factor.solve(model.options.young, state, model.info, only_failed=True)
        residual(True)

    reports = []
    operations = [("native64", lambda: model.surface_inverse.apply(model.options.young, omega, state))]
    for arithmetic, paired in ((False, False), (True, False), (False, True)):
        operations.append(
            (
                f"arithmetic32-{arithmetic}-paired-{paired}",
                lambda a=arithmetic, p=paired: kernel_candidate(
                    model.options.young, omega, state, model.surface_inverse.info, high, low, a, p
                ),
            )
        )
    for name, operation in operations:
        kernel_direct_init(state)
        operation()
        residual()
        initial_failures = int((~qd_to_numpy(state.valid)).sum())
        initial_residual = float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max())
        checked(operation)
        assert qd_to_numpy(state.valid).all()
        kernel_peak(model.options.young, model.options.poisson, state, model.info)
        peak = qd_to_numpy(state.peak, copy=True)
        np.testing.assert_allclose(peak, expected_peak, rtol=1e-4, atol=1e-3)
        timings = {}
        for scope, run in (("application", operation), ("checked", lambda op=operation: checked(op))):
            run()
            qd.sync()
            started = time.perf_counter()
            for _ in range(args.repetitions):
                run()
            qd.sync()
            timings[scope] = 1e3 * (time.perf_counter() - started) / args.repetitions
        report = {
            "name": name,
            "milliseconds": timings,
            "initial_failed_envs": initial_failures,
            "initial_max_residual_N": initial_residual,
            "peak_error_Pa": float(abs(peak - expected_peak).max()),
            "corrected_max_residual_N": float(np.sqrt(qd_to_numpy(state.residual_norm_squared)).max()),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "reports": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
