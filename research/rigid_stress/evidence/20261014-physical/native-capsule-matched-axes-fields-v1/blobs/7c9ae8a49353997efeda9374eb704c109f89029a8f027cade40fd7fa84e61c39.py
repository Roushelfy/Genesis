"""Verify complete frozen corner stresses using exact polynomial interpolation and decimal geometry inversion."""

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def invert(matrix):
    """Invert a small matrix using pivoted Gauss-Jordan elimination."""
    count = len(matrix)
    zero = matrix[0][0] * 0
    one = zero + 1
    augmented = [
        row.copy() + [one if i_row == i_column else zero for i_column in range(count)]
        for i_row, row in enumerate(matrix)
    ]
    for i_column in range(count):
        i_pivot = max(range(i_column, count), key=lambda i_row: abs(augmented[i_row][i_column]))
        assert augmented[i_pivot][i_column] != 0
        augmented[i_pivot], augmented[i_column] = augmented[i_column], augmented[i_pivot]
        scale = augmented[i_column][i_column]
        augmented[i_column] = [value / scale for value in augmented[i_column]]
        for i_row in range(count):
            if i_row != i_column:
                scale = augmented[i_row][i_column]
                augmented[i_row] = [
                    first - scale * second for first, second in zip(augmented[i_row], augmented[i_column], strict=True)
                ]
    return [row[count:] for row in augmented]


def interpolation_derivatives():
    """Derive exact corner differentiation weights from the ten quadratic monomials."""
    vertices = [[Fraction(value) for value in row] for row in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    nodes = vertices + [
        [(first + second) / 2 for first, second in zip(vertices[i_first], vertices[i_second], strict=True)]
        for i_first, i_second in edges
    ]
    powers = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ]

    def evaluate(position, power):
        value = Fraction(1)
        for coordinate, exponent in zip(position, power, strict=True):
            value *= coordinate**exponent
        return value

    vandermonde = [[evaluate(position, power) for power in powers] for position in nodes]
    inverse = invert(vandermonde)
    weights = []
    for corner in vertices:
        corner_weights = []
        for i_axis in range(3):
            derivatives = []
            for power in powers:
                exponent = list(power)
                coefficient = exponent[i_axis]
                exponent[i_axis] = max(0, exponent[i_axis] - 1)
                derivatives.append(coefficient * evaluate(corner, exponent))
            values = [
                sum(derivatives[i_power] * inverse[i_power][i_node] for i_power in range(10)) for i_node in range(10)
            ]
            assert all(value.denominator == 1 for value in values)
            for i_power in range(10):
                assert (
                    sum(values[i_node] * vandermonde[i_node][i_power] for i_node in range(10)) == derivatives[i_power]
                )
            corner_weights.append([int(value) for value in values])
        weights.append(corner_weights)
    return weights


def compare(first, reference):
    error = abs(first - reference)
    fraction = error / (1e-5 + 1e-9 * abs(reference))
    position = np.unravel_index(np.argmax(fraction), fraction.shape)
    return {
        "maximum_absolute_difference_Pa": float(error.max()),
        "maximum_existing_component_budget_fraction": float(fraction.max()),
        "worst_element_corner_component": [int(value) for value in position],
        "first_at_worst_Pa": float(first[position]),
        "reference_at_worst_Pa": float(reference[position]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--precision", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.precision >= 64
    closed_source(args.probe_run)
    assert int((args.probe_run / "returncode.txt").read_text()) == 0
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.probe_run / "results/summary.json"
    report = json.loads(summary_path.read_text())
    inputs = {str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    diagnostics = list(dict.fromkeys(row["diagnostic_run"] for row in report["rows"]))
    weights = interpolation_derivatives()
    rows = []
    with localcontext() as context:
        context.prec = args.precision
        young, poisson = Decimal.from_float(1e10), Decimal.from_float(0.3)
        mu = young / (2 * (1 + poisson))
        lame_lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        for i_case, diagnostic in enumerate(diagnostics):
            run = Path(diagnostic)
            closed_source(run)
            assert int((run / "returncode.txt").read_text()) == 0
            geometry = run / "results/geometry.npz"
            inputs[str(geometry)] = hashlib.sha256(geometry.read_bytes()).hexdigest()
            with np.load(geometry) as saved:
                vertices, elements = saved["native_vertices_m"].copy(), saved["native_elements"].copy()
            fields = {}
            for row in report["rows"]:
                if row["diagnostic_run"] == diagnostic:
                    mode = row["mode"]
                    field_path = run / f"results/{mode}.npz"
                    candidate_path = args.probe_run / f"results/case{i_case:02d}-{mode}.npz"
                    for path in [field_path, candidate_path]:
                        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    with np.load(field_path) as saved:
                        fields[mode] = {
                            "displacement": saved["frozen_native_displacement_m"].copy(),
                            "cofactor_extended": saved["independent_extended_geometry_components_Pa"].copy(),
                        }
                    with np.load(candidate_path) as saved:
                        fields[mode].update(
                            candidate=saved["difference_form_components_Pa"].copy(),
                            original_reference=saved["original_reference_components_Pa"].copy(),
                        )
                    fields[mode]["decimal"] = np.empty_like(fields[mode]["cofactor_extended"])
            maximum_inverse_residual = Decimal(0)
            for i_element, nodes in enumerate(elements):
                points = [[Decimal.from_float(float(value)) for value in vertices[i_node]] for i_node in nodes[:4]]
                matrix = [
                    [points[i_corner + 1][i_axis] - points[0][i_axis] for i_corner in range(3)] for i_axis in range(3)
                ]
                inverse = invert(matrix)
                maximum_inverse_residual = max(
                    maximum_inverse_residual,
                    max(
                        abs(
                            sum(matrix[i_row][i_inner] * inverse[i_inner][i_column] for i_inner in range(3))
                            - (i_row == i_column)
                        )
                        for i_row in range(3)
                        for i_column in range(3)
                    ),
                )
                for data in fields.values():
                    nodal = [
                        [Decimal.from_float(float(value)) for value in data["displacement"][i_node]] for i_node in nodes
                    ]
                    for i_corner in range(4):
                        reference_derivative = [
                            [
                                sum(
                                    nodal[i_node][i_row] * coefficient
                                    for i_node, coefficient in enumerate(weights[i_corner][i_axis])
                                    if coefficient
                                )
                                for i_axis in range(3)
                            ]
                            for i_row in range(3)
                        ]
                        derivative = [
                            [
                                sum(
                                    reference_derivative[i_row][i_inner] * inverse[i_inner][i_column]
                                    for i_inner in range(3)
                                )
                                for i_column in range(3)
                            ]
                            for i_row in range(3)
                        ]
                        trace = sum(derivative[i_axis][i_axis] for i_axis in range(3))
                        components = [2 * mu * derivative[i_axis][i_axis] + lame_lambda * trace for i_axis in range(3)]
                        components += [
                            mu * (derivative[i_first][i_second] + derivative[i_second][i_first])
                            for i_first, i_second in [(0, 1), (0, 2), (1, 2)]
                        ]
                        data["decimal"][i_element, i_corner] = [np.longdouble(str(value)) for value in components]
                if (i_element + 1) % 2000 == 0:
                    print(json.dumps({"diagnostic": diagnostic, "completed_elements": i_element + 1}), flush=True)
            assert maximum_inverse_residual <= Decimal("1e-60")
            for mode, data in fields.items():
                row = {
                    "diagnostic_run": diagnostic,
                    "mode": mode,
                    "tetrahedra": len(elements),
                    "maximum_decimal_inverse_identity_residual": str(maximum_inverse_residual),
                    "extended_cofactor_vs_decimal_polynomial": compare(data["cofactor_extended"], data["decimal"]),
                    "candidate_vs_decimal_polynomial": compare(data["candidate"], data["decimal"]),
                    "original_reference_vs_decimal_polynomial": compare(data["original_reference"], data["decimal"]),
                }
                rows.append(row)
                np.savez_compressed(
                    args.output / f"case{i_case:02d}-{mode}.npz",
                    decimal_polynomial_components_Pa=data["decimal"],
                    candidate_components_Pa=data["candidate"],
                    original_reference_components_Pa=data["original_reference"],
                )
                print(json.dumps(row), flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    result = {
        "scope": __doc__,
        "probe_run": str(args.probe_run),
        "decimal_precision": args.precision,
        "exact_rational_ten_monomial_corner_qualification_passed": True,
        "exact_polynomial_reference_differentiation_weights": weights,
        "rows": rows,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Every frozen element/corner/component is checked using exact rational Vandermonde differentiation and pivoted decimal geometry inversion. Float64 inputs are converted exactly before96-digit arithmetic; outputs retain extended precision. Original failed replay codes and original reference discrepancies remain. Candidate CPU arithmetic and this independent oracle audit alone do not accept production CUDA, the old failed component comparisons, new physical states, convergence or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
