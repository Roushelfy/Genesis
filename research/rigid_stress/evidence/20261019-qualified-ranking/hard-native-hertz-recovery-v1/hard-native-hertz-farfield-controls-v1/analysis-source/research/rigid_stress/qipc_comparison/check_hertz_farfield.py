"""Compare the native four-moment far field with independent pressure integrals."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import (
    StressHertz,
    StressHertzState,
    func_hertz_particular,
    kernel_hertz_stress_quadrature,
)
from genesis.utils.array_class import V, V_VEC, V_MAT

from .artifacts import runtime_identity, write_json
from .check_native_hertz_recovery import array
from .hertz_reference import halfspace_stress


@qd.kernel
def evaluate(
    position: qd.Tensor,
    poisson: qd.Tensor,
    quadrature: qd.Tensor,
    tensor: qd.Tensor,
    status: qd.Tensor,
    state: StressHertzState,
):
    for i_q, i_p in qd.ndrange(status.shape[0], status.shape[1]):
        value, resolved = func_hertz_particular(
            position[i_q, i_p],
            state.axes[i_p],
            state.peak_pressure[i_p],
            state.force[i_p],
            poisson[i_p],
            quadrature,
            far_field=True,
        )
        tensor[i_q, i_p] = value
        status[i_q, i_p] = 0 if resolved and value.norm() < float("inf") else 3


def pressure_integral(points, axes, load, poisson, radial, angular):
    t, w = np.polynomial.legendre.leggauss(radial)
    t, w = (t + 1) / 2, w / 2
    theta = 2 * np.pi * (np.arange(angular) + 0.5) / angular
    source = (
        np.sqrt(1 - t[:, None] ** 2)[..., None]
        * np.column_stack([np.cos(theta) * axes[0], np.sin(theta) * axes[1]])[None]
    ).reshape(-1, 2)
    force = np.repeat(3 * t * t * w * load / angular, angular)
    return halfspace_stress(points, source, force, poisson)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"tensor_absolute_over_p0": 1e-5, "independent_quadrature_change_over_p0": 1e-8}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "targets": targets,
            "scope": "Four equal source forces at (+/-a/sqrt5,+/-b/sqrt5) match Hertz total and second pressure moments. Compare r/a>=8 to independently integrated Boussinesq displacement derivatives, with12x64/20x96 pressure rules. No shell or scene qualification.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    parameters = np.array([(ratio, nu) for ratio in [1, 0.73, 0.1, 0.05] for nu in [0, 0.3, 0.49]])
    axes = np.column_stack([np.ones(len(parameters)), parameters[:, 0]])
    load = 2 * np.pi * axes.prod(axis=1) / 3
    points = np.array(
        [
            radius * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            for radius in [8, 12, 32, 100]
            for theta in [0, np.pi / 6, np.pi / 3, np.pi / 2 - 1e-8]
            for phi in np.arange(12) * 2 * np.pi / 12
        ]
    )
    positions = np.broadcast_to(points, (len(parameters), *points.shape)).copy()
    state = StressHertz(len(parameters))
    state.state.axes.from_numpy(axes)
    state.state.force.from_numpy(load)
    state.state.peak_pressure.fill(1)
    state.state.status.fill(0)
    coordinate = V_VEC(3, dtype=gs.qd_float, shape=(len(points), len(parameters)))
    poisson = V(dtype=gs.qd_float, shape=(len(parameters),))
    quadrature = V_VEC(2, dtype=gs.qd_float, shape=(16,))
    tensor = V_MAT(3, 3, dtype=gs.qd_float, shape=(len(points), len(parameters)))
    status = V(dtype=gs.qd_int, shape=(len(points), len(parameters)))
    coordinate.from_numpy(positions.swapaxes(0, 1).copy())
    poisson.from_numpy(parameters[:, 1].copy())
    kernel_hertz_stress_quadrature(quadrature)
    evaluate(coordinate, poisson, quadrature, tensor, status, state.state)
    qd.sync()
    actual, accepted = array(tensor), array(status)
    assert np.all(accepted == 0)
    low, high, rows = [], [], []
    for i, (ratio, nu) in enumerate(parameters):
        first = pressure_integral(points, axes[i], load[i], nu, 12, 64)
        second = pressure_integral(points, axes[i], load[i], nu, 20, 96)
        error = float(np.linalg.norm(actual[i] - second, axis=(-1, -2)).max())
        change = float(np.linalg.norm(first - second, axis=(-1, -2)).max())
        row = {
            "axis_ratio": ratio,
            "poisson": nu,
            "maximum_tensor_error_over_p0": error,
            "reference_change_over_p0": change,
        }
        rows.append(row)
        write_json(args.output / "progress.json", {"rows": rows})
        assert (
            error < targets["tensor_absolute_over_p0"] and change < targets["independent_quadrature_change_over_p0"]
        ), row
        low.append(first)
        high.append(second)
    np.savez_compressed(
        args.output / "fields.npz",
        points=points,
        parameters=parameters,
        native_tensor=actual,
        low_reference=np.stack(low),
        high_reference=np.stack(high),
    )
    write_json(
        args.output / "result.json", {"passed": True, "rows": rows, "targets": targets, "whole_goal_complete": False}
    )


if __name__ == "__main__":
    main()
