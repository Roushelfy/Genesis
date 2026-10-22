"""Check complete vector/load histories and measure resident memory-update cost."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .native_hertz_cone_history import ConeHistoryInput, ConeHistoryState, NativeConeHistory


def array(field):
    return qd_to_numpy(field, transpose=True, copy=True)


@qd.kernel
def benchmark_inputs(inputs: ConeHistoryInput, step: int):
    for env in range(inputs.normal.shape[0]):
        phase = 0.09 * step + 0.006 * (env % 97)
        normal = 0.2 * (1 + 0.25 * qd.sin(phase))
        inputs.normal[env], inputs.radius[env], inputs.friction[env] = normal, 7e-5 * (normal / 0.2) ** (1 / 3), 0.4
        inputs.force[env] = 0.4 * normal * qd.Vector([0.55 * qd.sin(1.1 * phase), 0.45 * qd.sin(0.7 * phase)])
        inputs.angle[env], inputs.reset[env] = 0.0, step == 0


@qd.kernel
def capture_history(
    state: ConeHistoryState,
    step: int,
    status: qd.types.ndarray(),
    iterations: qd.types.ndarray(),
    residual: qd.types.ndarray(),
    bound: qd.types.ndarray(),
):
    for env in range(state.status.shape[0]):
        status[step, env] = state.status[env]
        iterations[step, env] = state.iterations[env]
        residual[step, env] = state.residual[env]
        bound[step, env] = state.traction_bound[env]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--benchmark", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert (args.reference.parent / "returncode.txt").read_text().strip() == "0"
    reference = json.loads((args.reference / "result.json").read_text())
    assert reference["force_and_continuous_cone_gates_met"]
    targets = {
        "line_absolute": 1e-5,
        "surface_traction_absolute_over_mu_p0": 1e-4,
        "force_residual": 1e-9,
        "continuous_slope_bound": 1 + 1e-10,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "reference_result": file_id(args.reference / "result.json"),
            "reference_fields": file_id(args.reference / "fields.npz"),
            "targets": targets,
            "scope": "All97 observations of changing normal load and two-direction radial memory at16/32 intervals; an independently rotated frame shares the same physical path. CPU reference uses independent constrained optimization and KKT checks. Resident benchmarks include input law and history update, exclude tensor evaluation/rigid dynamics, and retain full audited admission.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    with np.load(args.reference / "fields.npz") as data:
        fields = {key: data[key] for key in data.files if key.endswith("_line") or key.endswith("_traction")}
        radial = data["radial_queries"]
    rows, arrays = [], {}
    for count in [16, 32]:
        native = NativeConeHistory(3, count)
        native.input.friction.fill(0.4)
        nodes = np.linspace(0, 1, count + 1)
        radial_weight = np.sqrt(np.maximum(nodes[None, 1:] ** 2 - radial[:, None] ** 2, 0)) - np.sqrt(
            np.maximum(nodes[None, :-1] ** 2 - radial[:, None] ** 2, 0)
        )
        lines, coefficients, states, iteration_history = [], [], [], []
        previous_angle = 0.0
        for step, time in enumerate(np.linspace(0, 4 * np.pi, 97)):
            normal = 0.2 * (1 + 0.35 * np.sin(time))
            radius = 7e-5 * (normal / 0.2) ** (1 / 3)
            angle = 0.017 * step
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            first = np.array([0.55 * np.sin(1.1 * time), 0.0])
            second = np.array([first[0], 0.45 * np.sin(0.7 * time)])
            native.input.normal.fill(float(normal))
            native.input.radius.fill(float(radius))
            native.input.force.from_numpy(0.4 * normal * np.array([first, second, rotation @ second]))
            native.input.angle.from_numpy(np.array([0, 0, angle - previous_angle]))
            native.update()
            qd.sync()
            actual, status = array(native.state.line), array(native.state.status)
            expected = np.stack(
                [
                    fields[f"directionFalse_n{count}_line"][step],
                    fields[f"directionTrue_n{count}_line"][step],
                    fields[f"directionTrue_n{count}_line"][step] @ rotation.T,
                ]
            )
            slopes = (actual[:, :-1] - actual[:, 1:]) / np.diff(nodes**2)[None, :, None]
            traction = np.einsum("rk,bki->bri", radial_weight, slopes)
            expected_traction = np.stack(
                [
                    fields[f"directionFalse_n{count}_traction"][step],
                    fields[f"directionTrue_n{count}_traction"][step],
                    fields[f"directionTrue_n{count}_traction"][step] @ rotation.T,
                ]
            )
            line_error = float(np.max(np.abs(actual - expected)))
            traction_error = float(np.max(np.abs(traction - expected_traction)))
            row = {
                "intervals": count,
                "step": step,
                "status": status.tolist(),
                "line_maximum_error": line_error,
                "traction_maximum_error_over_mu_p0": traction_error,
                "force_residuals": array(native.state.residual).tolist(),
                "slope_bounds": array(native.state.traction_bound).tolist(),
                "iterations": array(native.state.iterations).tolist(),
                "dual_residuals": array(native.state.dual_residual).tolist(),
                "line_residuals": array(native.state.line_residual).tolist(),
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
            lines.append(actual)
            coefficients.append(array(native.state.coefficients))
            states.append(status)
            iteration_history.append(array(native.state.iterations))
            if not (
                np.all(status == 0)
                and line_error < targets["line_absolute"]
                and traction_error < targets["surface_traction_absolute_over_mu_p0"]
            ):
                np.savez_compressed(
                    args.output / f"failed-n{count}-step{step}.npz",
                    line=actual,
                    expected_line=expected,
                    status=status,
                    traction=traction,
                    expected_traction=expected_traction,
                )
            assert np.all(status == 0), row
            assert (
                line_error < targets["line_absolute"]
                and traction_error < targets["surface_traction_absolute_over_mu_p0"]
            ), row
            assert np.max(array(native.state.residual)) < targets["force_residual"]
            assert np.max(array(native.state.traction_bound)) <= targets["continuous_slope_bound"]
            previous_angle = angle
        arrays.update(
            {
                f"n{count}_line": np.stack(lines),
                f"n{count}_coefficients": np.stack(coefficients),
                f"n{count}_status": np.stack(states),
                f"n{count}_iterations": np.stack(iteration_history),
            }
        )
    np.savez_compressed(args.output / "conformance.npz", **arrays)
    timings = []
    if args.benchmark:
        for count in [16, 32]:
            for batch in [1, 128, 1024]:
                model = NativeConeHistory(batch, count)

                def block():
                    for step in range(64):
                        benchmark_inputs(model.input, step)
                        model.update()
                    qd.sync()

                block()
                elapsed = []
                for repeat in range(3):
                    started = perf_counter()
                    block()
                    elapsed.append((perf_counter() - started) / 64)
                    assert np.all(array(model.state.status) == 0), (count, batch, repeat, array(model.state.status))
                status = qd.ndarray(dtype=gs.qd_int, shape=(64, batch))
                iterations = qd.ndarray(dtype=gs.qd_int, shape=(64, batch))
                residual = qd.ndarray(dtype=gs.qd_float, shape=(64, batch))
                bound = qd.ndarray(dtype=gs.qd_float, shape=(64, batch))
                for step in range(64):
                    benchmark_inputs(model.input, step)
                    model.update()
                    capture_history(model.state, step, status, iterations, residual, bound)
                qd.sync()
                observed = status.to_numpy()
                np.savez_compressed(
                    args.output / f"benchmark-n{count}-b{batch}.npz",
                    status=observed,
                    iterations=iterations.to_numpy(),
                    residual=residual.to_numpy(),
                    bound=bound.to_numpy(),
                )
                assert np.all(observed == 0), (count, batch, np.unique(observed, return_counts=True))
                assert residual.to_numpy().max() < targets["force_residual"]
                assert bound.to_numpy().max() <= targets["continuous_slope_bound"]
                timings.append(
                    {
                        "intervals": count,
                        "batch": batch,
                        "seconds_per_update": elapsed,
                        "median_seconds_per_update": float(np.median(elapsed)),
                        "environment_updates_per_second": float(batch / np.median(elapsed)),
                        "admitted_updates": int(observed.size),
                        "maximum_audited_iterations": int(iterations.to_numpy().max()),
                    }
                )
                write_json(args.output / "timing-progress.json", {"rows": timings})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "benchmark": timings,
            "targets": targets,
            "maximum_line_error": max(row["line_maximum_error"] for row in rows),
            "maximum_traction_error_over_mu_p0": max(row["traction_maximum_error_over_mu_p0"] for row in rows),
            "independent_general_vector_friction_physics_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
