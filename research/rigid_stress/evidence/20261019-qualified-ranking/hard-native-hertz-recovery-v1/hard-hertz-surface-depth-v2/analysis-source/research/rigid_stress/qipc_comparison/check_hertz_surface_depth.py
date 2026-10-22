"""Check a confocal root bracket at small positive material depths."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz, StressHertzState, func_hertz_tensor
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .native_hertz import NativeHertz
from .native_hertz_stress import NativeHertzStress
from .study_hertz_residual import elliptic_stress


@qd.kernel
def evaluate(
    position: qd.Tensor,
    poisson: qd.Tensor,
    quadrature: qd.Tensor,
    tensor: qd.Tensor,
    status: qd.Tensor,
    hertz: StressHertzState,
):
    for i_q, i_p in qd.ndrange(status.shape[0], status.shape[1]):
        axes = hertz.axes[i_p]
        value, resolved = func_hertz_tensor(position[i_q, i_p] / axes[0], axes[1] / axes[0], poisson[i_p], quadrature)
        tensor[i_q, i_p] = value * hertz.peak_pressure[i_p]
        status[i_q, i_p] = 0 if resolved and value.norm() < float("inf") else 3


def array(value):
    return qd_to_numpy(value, transpose=True, copy=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "axis_absolute_over_p0": 1e-7,
        "interior_surface_limit": 2e-5,
        "edge_surface_limit": 2e-4,
        "unchanged_resolved_depths": 1e-7,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "targets": targets,
            "scope": "Confocal numerical control down to z/a=1e-16; analytic axis stress and positive-depth surface limits. No finite-body, friction or scene claim.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    ratios = np.array([1.0, 0.73, 0.1, 0.05])
    xy = np.array([[0, 0], [0.3, 0.4], [0.5, 0], [0, 0.5], [0.999999, 0], [1, 0], [1.000001, 0]])
    depths = np.array([0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-5, 0.01, 1, 20])
    positions = np.concatenate([np.column_stack([np.tile(point, (len(depths), 1)), depths]) for point in xy])
    positions = np.broadcast_to(positions, (len(ratios), *positions.shape)).copy()
    positions[:, :, 1] *= ratios[:, None]
    axes = np.column_stack([np.ones(len(ratios)), ratios])
    native = NativeHertzStress(len(ratios), positions.shape[1])
    baseline = NativeHertzStress(len(ratios), positions.shape[1])
    hertz, old = StressHertz(len(ratios)), NativeHertz(len(ratios))
    for source in [hertz, old]:
        source.state.axes.from_numpy(axes)
        source.state.peak_pressure.fill(1)
        source.state.status.fill(0)
    for probe in [native, baseline]:
        probe.state.position.from_numpy(positions.swapaxes(0, 1).copy())
        probe.state.poisson.fill(0.3)
    evaluate(
        native.state.position,
        native.state.poisson,
        native.quadrature,
        native.state.tensor,
        native.state.status,
        hertz.state,
    )
    baseline.update(old.state)
    qd.sync()
    result, status, prior, prior_status = (
        array(native.state.tensor),
        array(native.state.status),
        array(baseline.state.tensor),
        array(baseline.state.status),
    )
    np.savez_compressed(
        args.output / "fields.npz",
        positions=positions,
        axes=axes,
        new_tensor=result,
        new_status=status,
        old_tensor=prior,
        old_status=prior_status,
    )
    assert np.all(status == 0), np.argwhere(status != 0)
    vertical = -ratios[:, None] / np.sqrt((1 + depths**2)[None] * (ratios[:, None] ** 2 + depths[None] ** 2))
    axis_error = float(np.max(np.abs(result[:, : len(depths), 2, 2] - vertical)))
    assert axis_error < targets["axis_absolute_over_p0"], axis_error
    field = result.reshape(len(ratios), len(xy), len(depths), 3, 3)
    difference = np.linalg.norm(field[:, :, 1:4] - field[:, :, :1], axis=(-1, -2))
    interior, edge = float(difference[:, :4].max()), float(difference[:, 4:].max())
    assert interior < targets["interior_surface_limit"] and edge < targets["edge_surface_limit"], (interior, edge)
    mask = (positions[:, :, 2] >= 1e-5) & (prior_status == 0)
    unchanged = float(np.linalg.norm(result - prior, axis=(-1, -2))[mask].max())
    assert unchanged < targets["unchanged_resolved_depths"], unchanged
    reference = np.stack([elliptic_stress(points, a, order=16) for points, a in zip(positions, axes)])
    reference_field = reference.reshape(len(ratios), len(xy), len(depths), 3, 3)
    reference_limit = np.linalg.norm(reference_field[:, :, 1:4] - reference_field[:, :, :1], axis=(-1, -2))
    assert reference_limit[:, :4].max() < targets["interior_surface_limit"]
    assert reference_limit[:, 4:].max() < targets["edge_surface_limit"]
    np.savez_compressed(args.output / "cpu-reference.npz", tensor=reference, near_limit_change=reference_limit)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "targets": targets,
            "axis_error": axis_error,
            "interior_limit_change": interior,
            "edge_limit_change": edge,
            "unchanged_resolved_depths_max_difference": unchanged,
            "original_unresolved_count": int(np.count_nonzero(prior_status)),
            "new_unresolved_count": int(np.count_nonzero(status)),
            "cpu_reference_interior_limit_change": float(reference_limit[:, :4].max()),
            "cpu_reference_edge_limit_change": float(reference_limit[:, 4:].max()),
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
