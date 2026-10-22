"""Validate supplied-region native edge scales, pressure limits, isolation and resident cost."""

import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.edge import StressEdge, StressEdgeState, func_edge_pressure
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import edge_scales


def fill_inputs(state, values):
    state.force.from_numpy(values[:, 0].copy())
    state.modulus.from_numpy(values[:, 1].copy())
    state.slope.from_numpy(values[:, 2].copy())
    state.curvature.from_numpy(values[:, 3].copy())
    state.shear_ratio.from_numpy(values[:, 4].copy())
    state.poisson.from_numpy(values[:, 5].copy())


def outputs(state):
    return np.column_stack(
        [
            qd_to_numpy(state.width, transpose=True),
            qd_to_numpy(state.half_length, transpose=True),
            qd_to_numpy(state.exponent, transpose=True),
            qd_to_numpy(state.amplitude, transpose=True),
        ]
    )


@qd.kernel
def kernel_pressure_samples(samples: qd.types.ndarray(), pressure: qd.types.ndarray(), edge_state: StressEdgeState):
    for i_p, i_q in qd.ndrange(edge_state.force.shape[0], samples.shape[0]):
        pressure[i_p, i_q] = 0
        if edge_state.status[i_p] == 0:
            position = qd.Vector(
                [samples[i_q, 0] * edge_state.width[i_p], samples[i_q, 1] * edge_state.half_length[i_p]]
            )
            pressure[i_p, i_q] = func_edge_pressure(
                position,
                edge_state.width[i_p],
                edge_state.half_length[i_p],
                edge_state.exponent[i_p],
                edge_state.amplitude[i_p],
            )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--precision", choices=["32", "64"], default="64")
    cli.add_argument("--physical-inputs", type=Path, required=True)
    cli.add_argument("--timing", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tolerance = 1e-9 if args.precision == "64" else 8e-6
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(args.physical_inputs)],
            "backend": args.backend,
            "precision": args.precision,
            "scale_relative_target": tolerance,
            "scope": "Supplied physical-region normal load, reduced modulus, edge slope, along-edge curvature and transverse shear ratio. Independent SciPy digamma/Brent scales, finite/unbounded pressure limits, invalid/unloaded/slender-regime isolation and reset, plus optional warmed resident dispatch timings. Physical pressure budgets are supplied by separate coupled BEM comparisons. Collider binding, friction history, complete fields and accepted max/full scene throughput are outside this component test.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision=args.precision, logging_level="warning")
    candidates = [
        [f, 1e10 / (1 - nu**2), s, k, t, nu]
        for f, s, k, t, nu in product(
            [0.01, 0.05, 0.1], [0.04, 0.08, 0.12], [30.0, 80.0], [-1.2, -0.8, 0.0, 0.7, 1.3], [0.15, 0.3, 0.45]
        )
    ]
    candidates.extend(row["inputs"] for row in json.loads(args.physical_inputs.read_text())["cases"])
    valid, unsupported = [], []
    for values in candidates:
        try:
            expected = edge_scales(*values)
        except AssertionError:
            unsupported.append((values, [0.0, 0.0, 0.0, 0.0], 3))
        else:
            valid.append((values, [expected.width, expected.along, expected.exponent, expected.amplitude], 0))
    assert len(valid) >= 200
    records = valid + unsupported
    values = np.array([row[0] for row in records], dtype=gs.np_float)
    expected = np.array([row[1] for row in records])
    expected_status = np.array([row[2] for row in records])
    native = StressEdge(len(records))
    fill_inputs(native.state, values)
    begun = perf_counter()
    native.update()
    qd.sync()
    compilation = perf_counter() - begun
    actual = outputs(native.state)
    status = qd_to_numpy(native.state.status, transpose=True, copy=True)
    np.testing.assert_array_equal(status, expected_status)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)
    maximum = np.max(np.abs(actual[: len(valid)] / expected[: len(valid)] - 1), axis=0)
    samples = np.array(
        [
            [0.001, 0],
            [0.1, 0.2],
            [0.7, 0.2],
            [0.1, 0.85],
            [0, 0],
            [-0.01, 0],
            [1.1, 0],
            [0, 1.1],
            [0.01, -0.9],
            [0.2, -0.5],
        ],
        dtype=gs.np_float,
    )
    pressure = np.empty((len(records), len(samples)), dtype=gs.np_float)
    kernel_pressure_samples(samples, pressure, native.state)
    for i in range(len(valid)):
        w, b, exponent, amplitude = actual[i]
        extent = w * (1 - samples[:, 1] ** 2)
        x = samples[:, 0] * w
        reference = np.zeros(len(samples))
        interior = (x > 0) & (x < extent)
        reference[interior] = amplitude * ((extent[interior] - x[interior]) / x[interior]) ** exponent
        reference[(x == 0) & (extent > 0)] = np.inf
        np.testing.assert_array_equal(np.isinf(pressure[i]), np.isinf(reference))
        finite = np.isfinite(reference)
        np.testing.assert_allclose(pressure[i, finite], reference[finite], rtol=tolerance * 4, atol=0)
    assert not pressure[len(valid) :].any()
    invalid = values.copy()
    invalid[0, 0] = 0
    invalid[1, 0] = -1
    invalid[2, 1] = np.nan
    invalid[3, 2] = 0
    invalid[4, 3] = -1
    invalid[5, 5] = 0.5
    invalid[6, 4] = np.inf
    invalid[7, 2] = 1e-8
    invalid[8, 3] = 1e9
    fill_inputs(native.state, invalid)
    native.update()
    refused = qd_to_numpy(native.state.status, transpose=True, copy=True)
    np.testing.assert_array_equal(refused[:9], [1, 2, 2, 2, 2, 2, 2, 3, 3])
    np.testing.assert_array_equal(refused[9:], status[9:])
    changed = outputs(native.state)
    np.testing.assert_array_equal(changed[:9], 0)
    np.testing.assert_array_equal(changed[9:], actual[9:])
    fill_inputs(native.state, values)
    native.update()
    np.testing.assert_array_equal(outputs(native.state), actual)
    benchmarks = []
    if args.timing:
        for count in (64, 2048, 65536):
            begun = perf_counter()
            estimator = StressEdge(count)
            fill_inputs(estimator.state, np.resize(values[: len(valid)], (count, 6)))
            for _ in range(3):
                estimator.update()
            qd.sync()
            preparation = perf_counter() - begun
            timings = []
            for _ in range(7):
                begun = perf_counter()
                for _ in range(20):
                    estimator.update()
                qd.sync()
                timings.append((perf_counter() - begun) / 20)
            assert not qd_to_numpy(estimator.state.status, transpose=True).any()
            benchmarks.append(
                {
                    "regions": count,
                    "preparation_seconds": preparation,
                    "inclusive_seconds_per_dispatch": timings,
                    "median_seconds_per_dispatch": float(np.median(timings)),
                    "logical_tensor_bytes": count
                    * (10 * np.dtype(gs.np_float).itemsize + np.dtype(gs.np_int).itemsize),
                }
            )
    np.savez_compressed(
        args.output / "fields.npz", inputs=values, expected=expected, actual=actual, status=status, pressure=pressure
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "precision": args.precision,
            "valid_cases": len(valid),
            "unsupported_cases": len(unsupported),
            "maximum_relative_errors": maximum.tolist(),
            "cold_update_seconds": compilation,
            "sharp_edge_unbounded_pressure_checked": True,
            "invalid_unloaded_isolation_and_reset_checked": True,
            "benchmarks": benchmarks,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
