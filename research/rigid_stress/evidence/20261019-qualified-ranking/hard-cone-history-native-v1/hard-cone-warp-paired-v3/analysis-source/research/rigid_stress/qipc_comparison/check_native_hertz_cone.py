"""Compare native cone memory with exact scalar projections and frame controls."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.optimize import brentq

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .hertz_history_cone import line_weights
from .native_hertz_cone_history import NativeConeHistory


def array(field):
    return qd_to_numpy(field, transpose=True, copy=True)


def scalar_projection(prior, weights, cap, target):
    shift = brentq(lambda value: weights @ np.clip(prior + value, -cap, cap) - target, -3, 3, xtol=1e-14)
    return np.clip(prior + shift, -cap, cap)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "targets": {"line_absolute": 1e-6, "force_residual": 1e-9, "continuous_slope_bound": 1 + 1e-10},
            "scope": "Exact weighted scalar closest-line projections, coordinate-frame covariance, independent shared tridiagonal response, release/reset/failure isolation. No general-vector friction or physical stress/scene-throughput qualification.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    rows = []
    for count in [16, 32]:
        native = NativeConeHistory(3, count)
        nodes = np.linspace(0, 1, count + 1)
        weight = line_weights(nodes)
        cap = 1 - nodes**2
        difference = np.eye(count) - np.eye(count, k=1)
        matrix = np.diag(weight[:-1]) + native.penalty * (difference.T @ difference)
        expected_response = np.linalg.solve(matrix, weight[:-1])
        np.testing.assert_allclose(array(native.info.weight), weight, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(array(native.info.response), expected_response, rtol=1e-12, atol=1e-15)
        native.input.radius.fill(7e-5)
        native.input.normal.fill(0.2)
        native.input.friction.fill(0.4)
        prior = np.zeros(count + 1)
        previous_angle = 0.0
        for step, target in enumerate([0.15, 0.6, 0.3, 0.0, -0.3, -0.6, 0.0, 0.6]):
            prior = scalar_projection(prior, weight, cap, target)
            angle = step * 0.13
            direction = np.array([np.cos(angle), np.sin(angle)])
            force = np.array([[target, 0], direction * target, [0.2, 0]]) * 0.08
            native.input.force.from_numpy(force)
            native.input.angle.from_numpy(np.array([0, angle - previous_angle, 0]))
            native.update()
            qd.sync()
            status, line = array(native.state.status), array(native.state.line)
            expected = np.stack(
                [
                    np.column_stack((prior, np.zeros(count + 1))),
                    prior[:, None] * direction,
                    np.column_stack((scalar_projection(np.zeros(count + 1), weight, cap, 0.2), np.zeros(count + 1))),
                ]
            )
            path = args.output / f"n{count}-step{step}.npz"
            np.savez_compressed(
                path,
                status=status,
                line=line,
                expected_line=expected,
                force_n=force,
                residual=array(native.state.residual),
                dual_residual=array(native.state.dual_residual),
                line_residual=array(native.state.line_residual),
                iterations=array(native.state.iterations),
            )
            error = float(np.max(np.abs(line - expected)))
            row = {
                "intervals": count,
                "step": step,
                "status": status.tolist(),
                "line_maximum_error": error,
                "iterations": array(native.state.iterations).tolist(),
                "force_residuals": array(native.state.residual).tolist(),
                "dual_residuals": array(native.state.dual_residual).tolist(),
                "line_residuals": array(native.state.line_residual).tolist(),
                "slope_bounds": array(native.state.traction_bound).tolist(),
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
            assert np.array_equal(status, np.zeros(3)), row
            assert error < 1e-6, row
            assert np.max(array(native.state.residual)) < 1e-9
            assert np.max(array(native.state.traction_bound)) <= 1 + 1e-10
            previous_angle = angle
        before = array(native.state.line).copy()
        force = np.array([[0.6, 0], [0.1, 0.2], [0, 0]]) * 0.08
        native.input.force.from_numpy(force)
        native.input.angle.fill(0)
        native.input.normal.from_numpy(np.array([0.2, 0.2, 0]))
        native.input.radius.from_numpy(np.array([7e-5, 7e-5, 0]))
        native.input.reset.from_numpy(np.array([False, True, False]))
        native.update()
        qd.sync()
        np.testing.assert_array_equal(array(native.state.status), [0, 0, 1])
        direction = np.array([np.cos(0.3), np.sin(0.3)])
        native.input.reset.fill(False)
        native.input.force.from_numpy(np.array([0.08 * direction, -0.08 * direction, [0.0, 0.0]]))
        native.update()
        qd.sync()
        np.testing.assert_array_equal(array(native.state.status), [0, 0, 1])
        np.testing.assert_allclose(array(native.state.line)[0], cap[:, None] * direction, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(array(native.state.line)[1], -cap[:, None] * direction, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(array(native.state.line)[0], before[0], rtol=0, atol=1e-6)
        direction = np.array([0.1, 0.2])
        magnitude = np.linalg.norm(direction)
        expected = scalar_projection(np.zeros(count + 1), weight, cap, magnitude)[:, None] * direction / magnitude
        np.testing.assert_allclose(array(native.state.line)[1], expected, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(array(native.state.line)[2], 0)
        force[1] = [0.1, 0]
        native.input.force.from_numpy(force)
        native.input.reset.fill(False)
        native.update()
        qd.sync()
        np.testing.assert_array_equal(array(native.state.status), [0, 3, 1])
        force[1] = [0.008, 0]
        native.input.force.from_numpy(force)
        native.update()
        qd.sync()
        np.testing.assert_array_equal(array(native.state.status), [0, 5, 1])
        native.input.reset.from_numpy(np.array([False, True, False]))
        native.update()
        qd.sync()
        np.testing.assert_array_equal(array(native.state.status), [0, 0, 1])
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "maximum_line_error": max(row["line_maximum_error"] for row in rows),
            "maximum_iterations": max(max(row["iterations"]) for row in rows),
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
