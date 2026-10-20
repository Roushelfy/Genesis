"""Check batched native pressure, independent pad equations, release and failure isolation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_newton import NativePadNewton


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inputs", type=Path, required=True)
    cli.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((args.inputs / "result.json").read_text())
    assert preparation["passed"]
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
        "preparation": file_id(args.inputs / "result.json"), "backend": args.backend,
        "targets": {"pressure_relative": 1e-6, "pad_law": 1e-7, "wrench": 1e-8, "outer_gradient": 1e-10},
        "scope": "Native numerical pressure stage with independently prepared physical operators and cold rigid-coordinate guesses. No collider geometry producer or production timing is included."})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    rows = []
    for record in preparation["records"]:
        path = Path(record["inputs"]["path"])
        assert file_id(path) == record["inputs"]
        with np.load(path) as saved:
            inputs = {key: saved[key] for key in saved.files}
        batch = inputs["gap"].shape[1]
        native = NativePadNewton(inputs["mapping"], inputs["elastic"], batch)
        state = native.state
        for name in ("gap", "weight", "linear", "elastic_scale"):
            # Explicit field dispatch keeps the numerical stage's owned state typed.
            if name == "gap":
                state.gap.from_numpy(inputs[name])
            elif name == "weight":
                state.weight.from_numpy(inputs[name])
            elif name == "linear":
                state.linear.from_numpy(inputs[name])
            else:
                state.elastic_scale.from_numpy(inputs[name])
        state.coordinate.from_numpy(inputs["initial"])
        requested = np.ones(batch, dtype=gs.np_bool)
        before = perf_counter()
        native.solve(requested)
        qd.sync()
        cold_seconds = perf_counter() - before
        status = qd_to_numpy(state.status, transpose=True, copy=True)
        assert np.all(status == 1), status
        force = qd_to_numpy(state.force, transpose=True, copy=True).T
        coordinate = qd_to_numpy(state.coordinate, transpose=True, copy=True).T
        gradient = qd_to_numpy(state.gradient_squared, transpose=True, copy=True)
        counts = qd_to_numpy(state.iterations, transpose=True, copy=True)
        linear_counts = qd_to_numpy(state.linear_iterations, transpose=True, copy=True)
        pressure_errors, law_errors, wrench_errors = [], [], []
        for env in range(batch):
            pressure_error = np.linalg.norm(force[:, env] - inputs["expected_force"][:, env])
            pressure_error /= np.linalg.norm(inputs["expected_force"][:, env])
            actual_force = force[:, env] * inputs["loads_n"][env]
            alpha = inputs["interpolation"].T @ actual_force
            elastic_gap = inputs["compliance_m_per_n"] @ alpha * (1e10 / inputs["young_pa"][env])
            pose = coordinate[-4:, env] * inputs["length_scales_m"][env]
            gap = inputs["interpolation"] @ (inputs["nodal_gap_m"] + elastic_gap + inputs["rigid"] @ pose)
            law = inputs["spring_n_per_m"][:, env] * np.maximum(-gap, 0.0)
            law_error = np.linalg.norm(actual_force - law) / np.linalg.norm(actual_force)
            target = np.array([0.0, 0.0, 0.0, 1.0])
            wrench_error = np.linalg.norm((inputs["interpolation"] @ inputs["rigid"]).T @ force[:, env] - target)
            assert pressure_error <= 1e-6 and law_error <= 1e-7 and wrench_error <= 1e-8
            pressure_errors.append(pressure_error.item())
            law_errors.append(law_error.item())
            wrench_errors.append(wrench_error.item())
        path = args.output / f"q{record['order']}-pressure.npz"
        np.savez_compressed(path, normalized_force=force, coordinate=coordinate, status=status,
                            gradient_squared=gradient, newton_updates=counts, pcg_iterations=linear_counts)
        native.solve(requested)
        warm_force = qd_to_numpy(state.force, transpose=True, copy=True).T
        np.testing.assert_array_equal(warm_force, force)
        np.testing.assert_array_equal(qd_to_numpy(state.iterations, transpose=True), 0)
        # Release one environment at an exactly separated zero-load state.
        released_gap, released_linear, released_initial = inputs["gap"].copy(), inputs["linear"].copy(), coordinate.copy()
        released_gap[:, 0], released_linear[:, 0], released_initial[:, 0] = 1.0, 0.0, 0.0
        state.gap.from_numpy(released_gap)
        state.linear.from_numpy(released_linear)
        state.coordinate.from_numpy(released_initial)
        requested[:] = False
        requested[0] = True
        native.solve(requested)
        released = qd_to_numpy(state.force, transpose=True, copy=True).T
        np.testing.assert_array_equal(released[:, 0], 0)
        np.testing.assert_array_equal(released[:, 1:], force[:, 1:])
        # An invalid physical spring rejects its own environment.
        bad_weight = inputs["weight"].copy()
        bad_weight[0, 1] = -1
        state.weight.from_numpy(bad_weight)
        requested[:] = True
        native.solve(requested)
        bad_status = qd_to_numpy(state.status, transpose=True, copy=True)
        bad_force = qd_to_numpy(state.force, transpose=True, copy=True).T
        assert bad_status[1] == 6 and np.all(bad_status[np.arange(batch) != 1] == 1)
        assert np.isnan(bad_force[:, 1]).all()
        np.testing.assert_array_equal(bad_force[:, 0], 0)
        np.testing.assert_array_equal(bad_force[:, 2:], force[:, 2:])
        state.weight.from_numpy(inputs["weight"])
        state.gap.from_numpy(inputs["gap"])
        state.linear.from_numpy(inputs["linear"])
        restored = qd_to_numpy(state.coordinate, transpose=True, copy=True).T
        restored[:, :2] = inputs["initial"][:, :2]
        state.coordinate.from_numpy(restored)
        requested[:] = False
        requested[:2] = True
        native.solve(requested)
        recovered = qd_to_numpy(state.force, transpose=True, copy=True).T
        assert np.all(qd_to_numpy(state.status, transpose=True) == 1)
        np.testing.assert_allclose(recovered[:, :2], force[:, :2], rtol=1e-8, atol=1e-12)
        np.testing.assert_array_equal(recovered[:, 2:], force[:, 2:])
        rows.append({"order": record["order"], "pressure_relative_errors": pressure_errors,
            "independent_pad_law_errors": law_errors, "wrench_errors": wrench_errors,
            "maximum_gradient": np.sqrt(gradient).max().item(), "newton_updates": counts.tolist(),
            "pcg_iterations": linear_counts.tolist(), "cold_compile_and_solve_seconds": cold_seconds,
            "warm_unchanged_exact": True, "release_exact": True, "failure_isolation_exact": True,
            "partial_recovery_preserves_others_exact": True, "pressure": file_id(path), "inputs": record["inputs"]})
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(args.output / "result.json", {"passed": True, "backend": args.backend, "rows": rows,
        "production_automatic_model_admitted": False, "whole_goal_complete": False,
        "scope": "Native nonlinear pressure and matrix-free linear solves match every supplied independent normal pad case. Warm repeat, exact separation release, invalid-input isolation and selective recovery pass. Physical operators and initial rigid guesses remain CPU-prepared validation inputs. Dynamic geometry selection, friction integration, production lifecycle wiring and controlled throughput remain required."})


if __name__ == "__main__":
    main()
