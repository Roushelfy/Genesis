"""Check and time native edge fields on captured full-shell coordinates."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.edge_fields import StressEdgeFields
from genesis.utils.array_class import V, V_MAT, V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .check_native_edge_fields import kernel_evaluate
from .study_edge_fields import field_metrics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--geometry", type=Path, required=True)
    cli.add_argument("--fields", type=Path, required=True)
    cli.add_argument("--controls", type=Path)
    cli.add_argument("--precision", choices=["32", "64"], required=True)
    cli.add_argument("--variant", choices=["baseline", "candidate"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tolerance = 1e-8 if args.precision == "64" else 5e-4
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=True),
            "inputs": [file_id(p) for p in [args.geometry, args.fields, args.controls] if p is not None],
            "precision": args.precision,
            "variant": args.variant,
            "targets": {"numerical_maximum_point_relative": tolerance, "whole_sample_tensor_change": 0.005},
            "scope": "Resident edge particular field tensors on the actual captured complete-shell volume/probe coordinates. Includes inclusive warmed dispatch and synchronization. Excludes geometry/pressure prediction, boundary correction, elastic recovery, max reduction, transfers and complete scenes. Baseline-before/candidate/baseline-after run sequentially in one GPU allocation for both precisions.",
        },
    )
    gs.init(backend=gs.cuda, precision=args.precision, logging_level="warning")
    field = StressEdgeFields()
    cases = json.loads(args.geometry.read_text())["cases"]
    with np.load(args.fields) as f:
        common, saved = f["points_m"], f["particular_pa"]
    positions, inputs, expected = [], [], []
    for i, case in enumerate(cases):
        if case["kind"] != "pressure":
            continue
        basis, center, force = np.array(case["basis"]), np.array(case["center"]), np.array(case["force"])
        local = (common - center) @ basis
        assert local[:, 2].min() >= -1e-12
        local[:, 2] = np.maximum(local[:, 2], 0)
        local[:, 0] += 2 * (1 - case["exponent"]) * case["width"] / 5
        positions.extend(local)
        inputs.extend(
            [
                [
                    case["width"],
                    case["along"],
                    case["exponent"],
                    case["amplitude"],
                    force[2],
                    *(force[:2] / force[2]),
                    0.3,
                ]
            ]
            * len(local)
        )
        expected.extend(basis.T @ saved[i] @ basis)
    positions, inputs, expected = np.array(positions), np.array(inputs), np.array(expected)
    original_count = len(positions)
    if args.controls is not None:
        with np.load(args.controls) as f:
            positions = np.concatenate([positions, f["points_m"]])
            inputs = np.concatenate([inputs, f["inputs"]])
            expected = np.concatenate([expected, f["expected_pa"]])
    count = len(positions)
    points, parameters = V_VEC(3, dtype=gs.qd_float, shape=(count,)), V_VEC(8, dtype=gs.qd_float, shape=(count,))
    tensors, status = V_MAT(3, 3, dtype=gs.qd_float, shape=(count,)), V(dtype=gs.qd_int, shape=(count,))
    points.from_numpy(positions.astype(gs.np_float))
    parameters.from_numpy(inputs.astype(gs.np_float))
    begun = perf_counter()
    kernel_evaluate(points, parameters, field.quadrature, tensors, status)
    qd.sync()
    cold = perf_counter() - begun
    actual = qd_to_numpy(tensors, transpose=True, copy=True)
    codes = qd_to_numpy(status, transpose=True)
    np.savez_compressed(
        args.output / "native.npz",
        points_m=positions,
        inputs=inputs,
        expected_pa=expected,
        native_pa=actual,
        status=codes,
    )
    np.testing.assert_array_equal(codes, 0)
    relative = np.linalg.norm(actual - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))
    far = np.linalg.norm(positions, axis=1) >= 8 * np.maximum(inputs[:, 0], inputs[:, 1])
    compare = np.ones(count, dtype=bool)
    if args.variant == "candidate":
        compare[:original_count] &= ~far[:original_count]
    metrics = field_metrics(actual[:original_count], expected[:original_count])
    numerical = {
        "maximum_numerical_point_error": float(relative[compare].max()),
        "maximum_captured_point_change": float(relative[:original_count].max()),
        "captured_far_fraction": float(far[:original_count].mean()),
        "captured_field_metrics": metrics,
    }
    write_json(args.output / "numerical.json", numerical)
    assert numerical["maximum_numerical_point_error"] < tolerance
    assert metrics["tensor_relative"] < 0.005
    positions, inputs, actual = positions[:original_count], inputs[:original_count], actual[:original_count]
    benchmarks = []
    for count in [256, 8192, 65536]:
        points, parameters = V_VEC(3, dtype=gs.qd_float, shape=(count,)), V_VEC(8, dtype=gs.qd_float, shape=(count,))
        tensors, status = V_MAT(3, 3, dtype=gs.qd_float, shape=(count,)), V(dtype=gs.qd_int, shape=(count,))
        point_bank, input_bank = np.resize(positions, (count, 3)), np.resize(inputs, (count, 8))
        points.from_numpy(point_bank.astype(gs.np_float))
        parameters.from_numpy(input_bank.astype(gs.np_float))
        for _ in range(3):
            kernel_evaluate(points, parameters, field.quadrature, tensors, status)
        qd.sync()
        times = []
        for _ in range(7):
            begun = perf_counter()
            for _ in range(10):
                kernel_evaluate(points, parameters, field.quadrature, tensors, status)
            qd.sync()
            times.append((perf_counter() - begun) / 10)
        np.testing.assert_array_equal(qd_to_numpy(status, transpose=True), 0)
        np.testing.assert_allclose(
            qd_to_numpy(tensors, transpose=True), np.resize(actual, (count, 3, 3)), rtol=tolerance, atol=1e-8
        )
        benchmarks.append(
            {
                "queries": count,
                "inclusive_seconds": times,
                "median_seconds": float(np.median(times)),
                "far_fraction": float(
                    np.mean(np.linalg.norm(point_bank, axis=1) >= 8 * np.maximum(input_bank[:, 0], input_bank[:, 1]))
                ),
                "logical_tensor_bytes": count * (20 * np.dtype(gs.np_float).itemsize + np.dtype(gs.np_int).itemsize),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "precision": args.precision,
            "variant": args.variant,
            **numerical,
            "cold_seconds": cold,
            "benchmarks": benchmarks,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
