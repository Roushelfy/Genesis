"""Validate native Hertz lookup and measure resident estimation cost, excluding imports/JIT."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, ellipse_pressure, hertz_integrals
from .native_hertz import HertzState, NativeHertz, func_hertz_pressure


@qd.kernel
def kernel_pressure_samples(points: qd.types.ndarray(), pressure: qd.types.ndarray(), hertz_state: HertzState):
    for i, j in qd.ndrange(hertz_state.force.shape[0], points.shape[0]):
        offset = qd.Vector([points[j, 0], points[j, 1]])
        pressure[i, j] = func_hertz_pressure(
            offset, hertz_state.axes[i], hertz_state.direction[i], hertz_state.peak_pressure[i]
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    force, modulus, curvature, expected = [], [], [], []
    for index, q in enumerate(np.geomspace(0.051, 1.0, 101)):
        ix, iy, _ = hertz_integrals(q)
        angle = index * 0.31
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        values = np.array([50 * ix / iy, 50.0])
        c = rotation @ np.diag(values) @ rotation.T
        f, e = 0.01 * 10 ** ((index % 5) / 2), 1e9 * (1 + index % 3)
        force.append(f)
        modulus.append(e)
        curvature.append(c)
        expected.append(ellipse(f, e, c))
    force, modulus, curvature = np.array(force), np.array(modulus), np.array(curvature)
    native = NativeHertz(len(force))
    native.state.force.from_numpy(force)
    native.state.modulus.from_numpy(modulus)
    native.state.curvature.from_numpy(curvature)
    native.update()
    axes = qd_to_numpy(native.state.axes, transpose=True, copy=True)
    direction = qd_to_numpy(native.state.direction, transpose=True, copy=True)
    depth = qd_to_numpy(native.state.indentation, transpose=True, copy=True)
    peak = qd_to_numpy(native.state.peak_pressure, transpose=True, copy=True)
    assert np.all(qd_to_numpy(native.state.status, transpose=True) == 0)
    exact_axes, exact_direction, exact_depth, exact_peak = map(np.array, zip(*expected))
    errors = {
        "axes_max_relative": float(np.max(np.abs(axes / exact_axes - 1))),
        "indentation_max_relative": float(np.max(np.abs(depth / exact_depth - 1))),
        "peak_pressure_max_relative": float(np.max(np.abs(peak / exact_peak - 1))),
        "direction_outer_product_max_absolute": float(
            np.max(
                np.abs(
                    direction[:-1, :, None] * direction[:-1, None, :]
                    - exact_direction[:-1, :, None] * exact_direction[:-1, None, :]
                )
            )
        ),
    }
    assert max(errors.values()) < 5e-5, errors
    circle = (3 * force[-1] / (4 * modulus[-1] * 50)) ** (1 / 3)
    np.testing.assert_allclose(axes[-1], circle, rtol=1e-12, atol=0)
    np.testing.assert_allclose(depth[-1], circle * circle * 50, rtol=1e-12, atol=0)
    points = np.random.default_rng(23).uniform(-0.001, 0.001, (4096, 2))
    pressure = np.empty((len(force), len(points)))
    kernel_pressure_samples(points, pressure, native.state)
    expected_pressure = np.array([ellipse_pressure(points, axes[i], direction[i], peak[i]) for i in range(len(force))])
    np.testing.assert_allclose(pressure, expected_pressure, rtol=1e-11, atol=1e-8)
    # Changing one region cannot leave stale pressure in an unloaded/invalid region
    # or change other independently supplied regions.
    force[0] = 0
    force[1] = -1
    curvature[2] = np.diag([-1.0, 50.0])
    curvature[3] = np.diag([0.0001, 50.0])
    modulus[4] = np.nan
    native.state.force.from_numpy(force)
    native.state.modulus.from_numpy(modulus)
    native.state.curvature.from_numpy(curvature)
    native.update()
    np.testing.assert_array_equal(qd_to_numpy(native.state.status, transpose=True)[:5], [1, 2, 3, 3, 2])
    updated = qd_to_numpy(native.state.axes, transpose=True, copy=True)
    np.testing.assert_array_equal(updated[:5], 0)
    np.testing.assert_array_equal(updated[5:], axes[5:])
    benchmark = []
    for count in (1, 128, 512, 8192):
        begin = perf_counter()
        estimator = NativeHertz(count)
        qd.sync()
        preparation = perf_counter() - begin
        estimator.state.force.fill(0.1)
        estimator.state.modulus.fill(1e10 / (1 - 0.3**2))
        estimator.state.curvature.from_numpy(np.broadcast_to(np.diag([30.0, 70.0]), (count, 2, 2)).copy())
        estimator.update()
        qd.sync()
        times = []
        for _ in range(21):
            begin = perf_counter()
            for _ in range(100):
                estimator.update()
            qd.sync()
            times.append((perf_counter() - begin) / 100)
        benchmark.append(
            {
                "regions": count,
                "median_seconds_per_update": float(np.median(times)),
                "p90_seconds_per_update": float(np.quantile(times, 0.9)),
                "preparation_seconds_warm_kernel": preparation,
                "logical_tensor_bytes": count * (12 * 8 + 4) + 257 * 3 * 8,
            }
        )
    path = args.output / "hertz.npz"
    np.savez_compressed(
        path,
        axes=axes,
        exact_axes=exact_axes,
        direction=direction,
        exact_direction=exact_direction,
        indentation=depth,
        exact_indentation=exact_depth,
        peak_pressure=peak,
        exact_peak_pressure=exact_peak,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "cases": len(force),
            "errors": errors,
            "zero_invalid_unsupported_isolation_controls": True,
            "benchmark": benchmark,
            "arrays": file_id(path),
            "scope": "Supplied total physical-region force, tangent curvature and reduced modulus. Analytical ellipse/integral controls, pressure evaluation and resident estimation timing only; no collider grouping, shell or friction accuracy, public integration or scene throughput is claimed.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
