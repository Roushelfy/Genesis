"""Check native sharp-edge fields against continuous references and measure resident evaluation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.edge_fields import StressEdgeFields, func_edge_particular
from genesis.engine.solvers.rigid.stress.hertz import kernel_hertz_stress_quadrature
from genesis.utils.array_class import V, V_MAT, V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import edge_scales
from .edge_fields import composite_tensor
from .study_edge_fields import field_metrics


@qd.kernel
def kernel_evaluate(points: qd.Tensor, inputs: qd.Tensor, quadrature: qd.Tensor, tensors: qd.Tensor, status: qd.Tensor):
    for i_q in range(points.shape[0]):
        values = inputs[i_q]
        tensors[i_q], status[i_q] = func_edge_particular(
            points[i_q],
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            qd.Vector([values[5], values[6]]),
            values[7],
            quadrature,
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--precision", choices=["32", "64"], default="64")
    cli.add_argument("--physical-inputs", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--timing", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tolerance = 1e-8 if args.precision == "64" else 5e-4
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(args.physical_inputs)],
            "backend": args.backend,
            "precision": args.precision,
            "maximum_point_tensor_relative_target": tolerance,
            "scope": "Native prescribed sharp-edge particular tensors at near/far positive-depth and surface locations, independent continuous pressure references, original Hertz quadrature preservation, singular/invalid isolation and restoration. Resident tensor evaluation timing excludes scale estimation, geometry, finite-shell correction, max reduction, transfers and scenes.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision=args.precision, logging_level="warning")
    field = StressEdgeFields()
    original_rule = V_VEC(2, dtype=gs.qd_float, shape=(16,))
    kernel_hertz_stress_quadrature(original_rule)
    x, weight = np.polynomial.legendre.leggauss(16)
    theta = (x[::-1] + 1) * np.pi / 4
    tangent = np.tan(theta)
    expected_rule = np.column_stack([tangent * tangent, tangent * (1 + tangent * tangent) * weight[::-1] * np.pi / 2])
    rule_tolerance = 1e-12 if args.precision == "64" else 3e-5
    np.testing.assert_allclose(qd_to_numpy(original_rule, transpose=True), expected_rule, rtol=rule_tolerance, atol=0)
    positions, parameters, expected, reference, slices, receipts = [], [], [], [], [], []
    for case in json.loads(args.physical_inputs.read_text())["cases"]:
        model = edge_scales(*case["inputs"])
        load, _, _, _, _, poisson = case["inputs"]
        coupling_path = Path(case["sources"][1]["path"])
        ratio = case["direction"] * np.array(json.loads(coupling_path.read_text())["tangent_over_normal_components"])
        path = args.reference / f"group{case['contact']}/fields/direction{case['direction']}.npz"
        receipts += [file_id(coupling_path), file_id(path)]
        with np.load(path) as data:
            points, cpu, independent = data["points_m"], data["composite32_pa"], data["reference_high_pa"]
        surface = np.array(
            [
                [0.2 * model.width, 0, 0],
                [0.7 * model.width, 0.5 * model.along, 0],
                [model.width, 0, 0],
                [-model.width, 0, 0],
                [0, model.along * 1.2, 0],
            ]
        )
        surface_tensor = composite_tensor(surface, model, load, ratio)
        begin = len(positions)
        positions.extend(np.concatenate([points, surface]))
        parameters.extend(
            [[model.width, model.along, model.exponent, model.amplitude, load, *ratio, poisson]]
            * (len(points) + len(surface))
        )
        expected.extend(np.concatenate([cpu, surface_tensor]))
        reference.extend(independent)
        slices.append((begin, begin + len(points), begin + len(points) + len(surface)))
    write_json(args.output / "references.json", {"files": receipts})
    positions, parameters, expected = (
        np.array(positions, dtype=gs.np_float),
        np.array(parameters, dtype=gs.np_float),
        np.array(expected),
    )
    count = len(positions)
    points, inputs = V_VEC(3, dtype=gs.qd_float, shape=(count,)), V_VEC(8, dtype=gs.qd_float, shape=(count,))
    tensors, status = V_MAT(3, 3, dtype=gs.qd_float, shape=(count,)), V(dtype=gs.qd_int, shape=(count,))
    points.from_numpy(positions)
    inputs.from_numpy(parameters)
    started = perf_counter()
    kernel_evaluate(points, inputs, field.quadrature, tensors, status)
    qd.sync()
    compilation = perf_counter() - started
    actual = qd_to_numpy(tensors, transpose=True, copy=True)
    codes = qd_to_numpy(status, transpose=True, copy=True)
    np.savez_compressed(
        args.output / "fields.npz",
        positions_m=positions,
        parameters=parameters,
        expected_pa=expected,
        native_pa=actual,
        status=codes,
    )
    np.testing.assert_array_equal(codes, 0)
    relative = np.linalg.norm(actual - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))
    write_json(
        args.output / "numerical.json",
        {
            "maximum_point_tensor_relative": float(relative.max()),
            "maximum_index": int(relative.argmax()),
            "worst_position_m": positions[relative.argmax()].tolist(),
        },
    )
    assert relative.max() < tolerance
    rows, reference_start = [], 0
    reference = np.array(reference)
    for begin, end, surface_end in slices:
        independent = reference[reference_start : reference_start + end - begin]
        reference_start += end - begin
        rows.append(field_metrics(actual[begin:end], independent))
        surface_error = np.linalg.norm(
            actual[end:surface_end, :, 2] - expected[end:surface_end, :, 2]
        ) / np.linalg.norm(expected[end:surface_end, :, 2])
        assert surface_error < tolerance
    changed = positions.copy()
    changed[1, 2] = -1
    changed[2, 0] = np.nan
    changed[3] = 0
    changed[4, 1] = np.inf
    points.from_numpy(changed)
    kernel_evaluate(points, inputs, field.quadrature, tensors, status)
    np.testing.assert_array_equal(qd_to_numpy(status, transpose=True)[1:5], [2, 2, 3, 2])
    modified = qd_to_numpy(tensors, transpose=True, copy=True)
    np.testing.assert_array_equal(modified[1:5], 0)
    np.testing.assert_array_equal(modified[5:], actual[5:])
    np.testing.assert_array_equal(modified[0], actual[0])
    points.from_numpy(positions)
    kernel_evaluate(points, inputs, field.quadrature, tensors, status)
    np.testing.assert_array_equal(qd_to_numpy(tensors, transpose=True), actual)
    benchmarks = []
    if args.timing:
        for count in [256, 8192, 65536]:
            point_bank, input_bank = (
                V_VEC(3, dtype=gs.qd_float, shape=(count,)),
                V_VEC(8, dtype=gs.qd_float, shape=(count,)),
            )
            tensor_bank, status_bank = (
                V_MAT(3, 3, dtype=gs.qd_float, shape=(count,)),
                V(dtype=gs.qd_int, shape=(count,)),
            )
            point_bank.from_numpy(np.resize(positions, (count, 3)))
            input_bank.from_numpy(np.resize(parameters, (count, 8)))
            for _ in range(3):
                kernel_evaluate(point_bank, input_bank, field.quadrature, tensor_bank, status_bank)
            qd.sync()
            times = []
            for _ in range(7):
                begun = perf_counter()
                for _ in range(10):
                    kernel_evaluate(point_bank, input_bank, field.quadrature, tensor_bank, status_bank)
                qd.sync()
                times.append((perf_counter() - begun) / 10)
            assert not qd_to_numpy(status_bank, transpose=True).any()
            output = qd_to_numpy(tensor_bank, transpose=True)
            np.testing.assert_allclose(output, np.resize(actual, (count, 3, 3)), rtol=tolerance, atol=1e-8)
            benchmarks.append(
                {
                    "queries": count,
                    "inclusive_seconds": times,
                    "median_seconds": float(np.median(times)),
                    "logical_tensor_bytes": count
                    * (20 * np.dtype(gs.np_float).itemsize + np.dtype(gs.np_int).itemsize),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "backend": args.backend,
            "precision": args.precision,
            "queries": len(positions),
            "maximum_point_tensor_relative": float(relative.max()),
            "reference_errors": rows,
            "original_hertz_quadrature_checked": True,
            "invalid_singular_isolation_restore_checked": True,
            "cold_field_seconds": compilation,
            "benchmarks": benchmarks,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
