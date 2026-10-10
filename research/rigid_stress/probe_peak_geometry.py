"""Evaluate native geometry reuse and corner scheduling for the complete global peak."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo, StressState
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.operators import func_shape_gradient
from genesis.engine.solvers.rigid.stress.solve import kernel_balance, kernel_peak
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_prepare(gradients: qd.Tensor, stress_info: StressInfo):
    for i_e, corner, local in qd.ndrange(stress_info.elements.shape[0], 4, 10):
        bary = qd.Vector.zero(gs.qd_float, 4)
        bary[corner] = 1.0
        gradients[i_e, corner, local] = func_shape_gradient(local, bary, stress_info.gradients[i_e], stress_info.edges)


@qd.func
def func_corner(
    young: float,
    poisson: float,
    i_e: int,
    corner: int,
    i_b: int,
    gradients: qd.Tensor,
    stress_state: StressState,
    stress_info: StressInfo,
    cached: qd.template(),
):
    derivative = qd.Matrix.zero(gs.qd_float, 3, 3)
    for local in range(10):
        gradient = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(cached):
            gradient = gradients[i_e, corner, local]
        else:
            bary = qd.Vector.zero(gs.qd_float, 4)
            bary[corner] = 1.0
            gradient = func_shape_gradient(local, bary, stress_info.gradients[i_e], stress_info.edges)
        derivative += stress_state.displacement[stress_info.elements[i_e, local], i_b].outer_product(gradient)
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    sigma = mu * (derivative + derivative.transpose())
    for a in qd.static(range(3)):
        sigma[a, a] += lam * derivative.trace()
    value = 0.5 * (
        (sigma[0, 0] - sigma[1, 1]) ** 2 + (sigma[1, 1] - sigma[2, 2]) ** 2 + (sigma[2, 2] - sigma[0, 0]) ** 2
    )
    value += 3.0 * (sigma[0, 1] ** 2 + sigma[0, 2] ** 2 + sigma[1, 2] ** 2)
    return value


@qd.kernel(graph=True)
def kernel_candidate(
    young: float,
    poisson: float,
    gradients: qd.Tensor,
    stress_state: StressState,
    stress_info: StressInfo,
    cached: qd.template(),
    parallel_corners: qd.template(),
):
    for i_b in range(stress_state.active.shape[0]):
        stress_state.peak[i_b] = 0.0
    for item, i_b in qd.ndrange(
        stress_info.elements.shape[0] * (4 if parallel_corners else 1), stress_state.active.shape[0]
    ):
        value = gs.qd_float(0.0)
        if qd.static(parallel_corners):
            value = func_corner(young, poisson, item // 4, item % 4, i_b, gradients, stress_state, stress_info, cached)
        else:
            for corner in range(4):
                value = qd.max(
                    value, func_corner(young, poisson, item, corner, i_b, gradients, stress_state, stress_info, cached)
                )
        qd.atomic_max(stress_state.peak[i_b], qd.sqrt(value))
    for i_b in range(stress_state.active.shape[0]):
        stress_state.step_peak[i_b] = qd.max(stress_state.step_peak[i_b], stress_state.peak[i_b])


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
    state.force.from_numpy(random.normal(size=(model.info.vertices.shape[0], args.envs, 3)) * 0.01)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(args.envs,))
    omega.from_numpy(random.normal(size=(args.envs, 3)))
    kernel_balance(omega, state, model.info)
    model.solve(model.options, state)
    gradients = V_VEC(3, dtype=gs.qd_float, shape=(model.info.elements.shape[0], 4, 10))
    kernel_prepare(gradients, model.info)
    kernel_peak(model.options.young, model.options.poisson, state, model.info, False)
    expected = qd_to_numpy(state.peak, copy=True)
    operations = [("native", lambda: kernel_peak(model.options.young, model.options.poisson, state, model.info, False))]
    for cached, corners in ((True, False), (False, True), (True, True)):
        operations.append(
            (
                f"cached-{cached}-corners-{corners}",
                lambda c=cached, p=corners: kernel_candidate(
                    model.options.young, model.options.poisson, gradients, state, model.info, c, p
                ),
            )
        )
    reports = []
    for name, operation in operations:
        operation()
        actual = qd_to_numpy(state.peak, copy=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)
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
                "peak_error_Pa": float(abs(actual - expected).max()),
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    args.output.write_text(json.dumps({"envs": args.envs, "reports": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
