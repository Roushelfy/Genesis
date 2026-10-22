"""Check and time complete native normal or tangential Hertz traction tensors."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .hertz_stress import elliptic_stress
from .hertz_tangential import elliptic_tangential_stress
from .native_hertz import NativeHertz
from .native_hertz_stress import NativeHertzStress
from .native_hertz_tangential import NativeHertzTangentialStress


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--order", type=int, default=16)
    cli.add_argument("--tangential", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    count = 37
    ratio = np.geomspace(0.05, 1, count)
    major = np.geomspace(1e-5, 1e-3, count)
    axes = major[:, None] * np.column_stack((np.ones(count), ratio))
    peak = np.geomspace(1e3, 1e8, count)
    poisson = np.linspace(0, 0.49, count)
    angles = np.linspace(0, 2 * np.pi, count)
    tangents = np.column_stack((np.cos(angles), np.sin(angles)))
    angle = np.linspace(0, 2 * np.pi, 17)
    edge = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(angle))))
    surface = np.array([[0, 0, 0], [0.3, 0.4, 0], [1.2, 0.2, 0], [2, 1, 0]])
    depth = np.r_[1e-5, 1e-3, np.geomspace(0.01, 20, 17)]
    axial = np.column_stack((np.zeros((len(depth), 2)), depth))
    random = np.random.default_rng(101).uniform([0, 0, 0], [2, 2, 3], (33, 3))
    far = np.array([[20, 0, 0], [200, 0, 0], [20, 5, 20], [200, 50, 200]])
    unit = np.concatenate((edge, surface, axial, random, far))
    if args.tangential:
        unit = np.concatenate((unit[: -len(far)], [[1.00001, 0, 0], [1.00001, 0, 1e-8]], far))
    positions = np.broadcast_to(unit, (count, len(unit), 3)).copy()
    positions[:, :, 1] *= ratio[:, None]
    positions[:, :, 2] *= np.sqrt(ratio[:, None])
    positions *= major[:, None, None]
    evaluator = NativeHertzTangentialStress if args.tangential else NativeHertzStress
    hertz, native = NativeHertz(count), evaluator(count, len(unit), order=args.order)
    hertz.state.axes.from_numpy(axes)
    hertz.state.peak_pressure.from_numpy(peak)
    hertz.state.status.fill(0)
    native.state.position.from_numpy(positions.swapaxes(0, 1).copy())
    native.state.poisson.from_numpy(poisson)
    if args.tangential:
        native.state.tangent.from_numpy(tangents)
    native.update(hertz.state)
    qd.sync()
    status = qd_to_numpy(native.state.status, transpose=True, copy=True)
    assert np.all(status == 0), np.argwhere(status != 0)
    actual = qd_to_numpy(native.state.tensor, transpose=True, copy=True)
    expected = np.stack(
        [
            elliptic_tangential_stress(p, a, tangents[i], s, n, order=args.order)
            if args.tangential
            else elliptic_stress(p, a, s, n, order=args.order)
            for i, (p, a, s, n) in enumerate(zip(positions, axes, peak, poisson))
        ]
    )
    relative = np.linalg.norm((actual - expected) / peak[:, None, None, None], axis=(2, 3))
    assert relative.max() < 1e-7, (relative.max(), np.unravel_index(relative.argmax(), relative.shape))
    reference = np.stack(
        [
            elliptic_tangential_stress(p, a, tangents[i], s, n, order=192)
            if args.tangential
            else elliptic_stress(p, a, s, n, order=192)
            for i, (p, a, s, n) in enumerate(zip(positions, axes, peak, poisson))
        ]
    )
    quadrature = np.linalg.norm((actual - reference) / peak[:, None, None, None], axis=(2, 3))
    assert quadrature.max() < (5e-3 if args.tangential else 1e-3), quadrature.max()
    far_relative = np.linalg.norm(actual[:, -len(far) :] - reference[:, -len(far) :], axis=(2, 3)) / np.linalg.norm(
        reference[:, -len(far) :], axis=(2, 3)
    )
    assert far_relative.max() < 1e-5, far_relative.max()
    # Verify exact surface traction, finite edge values, scale invariance and
    # the symmetry-axis vertical stress independently of the potential integrals.
    surface_points = np.r_[np.arange(len(edge) + len(surface))]
    pressure = peak[:, None] * np.sqrt(
        np.maximum(1 - (positions[:, surface_points, :2] / axes[:, None]) ** 2 @ np.ones(2), 0)
    )
    if args.tangential:
        np.testing.assert_allclose(
            actual[:, surface_points][:, :, :2, 2] / peak[:, None, None],
            -pressure[:, :, None] / peak[:, None, None] * tangents[:, None],
            rtol=1e-7,
            atol=1e-7,
        )
        np.testing.assert_array_equal(actual[:, surface_points, 2, 2], 0)
    else:
        np.testing.assert_allclose(actual[:, surface_points, 2, 2], -pressure, rtol=1e-7, atol=1e-7 * peak.max())
        np.testing.assert_array_equal(actual[:, surface_points, :2, 2], 0)
    axial_indices = np.arange(len(edge) + len(surface), len(edge) + len(surface) + len(axial))
    z = positions[:, axial_indices, 2]
    vertical = (
        -peak[:, None]
        * axes.prod(axis=1)[:, None]
        / np.sqrt((axes[:, 0, None] ** 2 + z * z) * (axes[:, 1, None] ** 2 + z * z))
    )
    if args.tangential:
        np.testing.assert_array_equal(actual[:, axial_indices, 2, 2], 0)
        depth_circle = z[-1] / major[-1]
        shear_circle = (
            -1 + 1.5 * depth_circle * np.arctan2(1, depth_circle) - 0.5 * depth_circle**2 / (1 + depth_circle**2)
        )
        np.testing.assert_allclose(
            actual[-1, axial_indices, :2, 2] / peak[-1], shear_circle[:, None] * tangents[-1], rtol=1e-7, atol=1e-12
        )
    else:
        np.testing.assert_allclose(actual[:, axial_indices, 2, 2], vertical, rtol=1e-9, atol=1e-8)
    changed = np.zeros(count, dtype=gs.np_int)
    changed[1], changed[2] = 1, 2
    hertz.state.status.from_numpy(changed)
    bad_positions = positions.copy()
    bad_positions[3, 0, 2] = -major[3]
    bad_positions[4, 0, 0] = np.nan
    native.state.position.from_numpy(bad_positions.swapaxes(0, 1).copy())
    bad_poisson = poisson.copy()
    bad_poisson[5] = 0.5
    native.state.poisson.from_numpy(bad_poisson)
    if args.tangential:
        changed_tangents = tangents.copy()
        changed_tangents[6], changed_tangents[7] = np.nan, 0
        native.state.tangent.from_numpy(changed_tangents)
    native.update(hertz.state)
    altered = qd_to_numpy(native.state.tensor, transpose=True, copy=True)
    states = qd_to_numpy(native.state.status, transpose=True, copy=True)
    assert np.all(states[1] == 1) and np.all(states[2] == 2) and np.all(states[5] == 2)
    assert states[3, 0] == states[4, 0] == 2
    np.testing.assert_array_equal(altered[1:3], 0)
    np.testing.assert_array_equal(altered[5], 0)
    np.testing.assert_array_equal(altered[3:5, 0], 0)
    if args.tangential:
        assert np.all(states[6] == 2) and np.all(states[7] == 0)
        np.testing.assert_array_equal(altered[6:8], 0)
    np.testing.assert_array_equal(altered[8 if args.tangential else 6 :], actual[8 if args.tangential else 6 :])
    hertz.state.status.fill(0)
    native.state.position.from_numpy(positions.swapaxes(0, 1).copy())
    native.state.poisson.from_numpy(poisson)
    if args.tangential:
        native.state.tangent.from_numpy(tangents)
    native.update(hertz.state)
    np.testing.assert_array_equal(qd_to_numpy(native.state.tensor, transpose=True), actual)
    benchmark = []
    # Every region receives the same set of probes but distinct axes, peak and
    # Poisson values; all tensors remain resident during these repeated updates.
    for batch in (1, 128, 512):
        hertz, native = NativeHertz(batch), evaluator(batch, len(unit), order=args.order)
        selection = np.arange(batch) % count
        hertz.state.axes.from_numpy(axes[selection])
        hertz.state.peak_pressure.from_numpy(peak[selection])
        hertz.state.status.fill(0)
        native.state.position.from_numpy(positions[selection].swapaxes(0, 1).copy())
        native.state.poisson.from_numpy(poisson[selection])
        if args.tangential:
            native.state.tangent.from_numpy(tangents[selection])
        native.update(hertz.state)
        qd.sync()
        assert np.all(qd_to_numpy(native.state.status, transpose=True) == 0)
        elapsed = []
        for _ in range(7):
            begin = perf_counter()
            for _ in range(10):
                native.update(hertz.state)
            qd.sync()
            elapsed.append((perf_counter() - begin) / 10)
        benchmark.append(
            {
                "regions": batch,
                "probes_per_region": len(unit),
                "median_seconds": float(np.median(elapsed)),
                "p90_seconds": float(np.quantile(elapsed, 0.9)),
            }
        )
    path = args.output / "tensors.npz"
    np.savez_compressed(
        path,
        positions_m=positions,
        axes_m=axes,
        peak_pressure_pa=peak,
        poisson=poisson,
        native_stress_pa=actual,
        independent_stress_pa=reference,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "tangential": args.tangential,
            "regions": count,
            "probes_per_region": len(unit),
            "quadrature_order": args.order,
            "maximum_implementation_tensor_difference_over_peak_pressure": float(relative.max()),
            "maximum_high_order_tensor_difference_over_peak_pressure": float(quadrature.max()),
            "maximum_far_field_tensor_relative_difference": float(far_relative.max()),
            "release_invalid_input_isolation_and_recovery": True,
            "benchmark": benchmark,
            "arrays": file_id(path),
            "scope": "Native elliptic half-space tensors for the selected normal or proportional tangential traction at supplied principal-frame probes, including surface/edge/axis and near/far points. Timing excludes ellipse estimation, friction law/history, shell coupling, max reduction and moving-scene input preparation. No physical contact-law or global shell-peak admission.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
