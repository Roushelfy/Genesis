"""Check final error readers against published complete-array results and exact norm inequalities."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .stress_error_measures import (
    check_raw_receipt,
    conditional_interval,
    measure_case,
    pair_relation,
    raw_stress_peaks,
    uncertainty_product,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, details, raw_cache = {}, {}, [], {}

    def record(path, expected=None):
        path = Path(path)
        if str(path) not in receipts:
            receipts[str(path)] = file_id(path)
        if expected is not None:
            assert receipts[str(path)] == expected

    def read(relative):
        path = args.data_root / relative
        record(path)
        return json.loads(path.read_text())

    published = read("pad-absolute-errors-cpu-v2/group0/audit/result.json")
    assert published["passed"] and len(published["rows"]) == 81
    expected = {(row["load_n"], row["method"], row["metric"]): row for row in published["rows"]}
    for level, load in enumerate((0.1, 0.3, 1.0)):
        for group, method in enumerate(("dpr", "scr", "shell")):
            directory = args.data_root / f"full-source-joint-normal-fields-v1/group{group}/{method}-{level}"
            case = read(directory / "result.json")["cases"][0]
            peaks = read(
                f"common-pad-peaks-v1/group1/{method}-0/result.json"
                if level == 0
                else f"common-pad-peak-loads-v1/group{group + 3}/{method}-{level}/result.json"
            )["cases"][0]
            column = directory / f"columns-{load:g}-0.npz"
            record(column)
            result = measure_case(case, column, peak_case=peaks, require_surfaces=False)
            assert len(result["rows"]) == 9 and all(result["checks"].values())
            for row in result["rows"]:
                original = expected[load, method, row["metric"]]
                assert original["unit"] == row["unit"]
                np.testing.assert_allclose(
                    [row[key] for key in ("absolute_error", "reference_rms_or_peak", "relative_error")],
                    [original[key] for key in ("absolute_error", "reference_rms_or_peak", "relative_error")],
                    rtol=1e-10,
                    atol=1e-14,
                )
                checks[f"published81/{load}/{method}/{row['metric']}"] = True
            for raw in case["raw"]:
                for key in ("mesh", "field"):
                    record(raw[key]["path"], raw[key])
    for group, method in enumerate(("dpr", "scr", "shell")):
        directory = args.data_root / f"pad-normal-surface-fields-v1/group{group}/{method}-0"
        result = read(directory / "result.json")
        assert result["surface_quadrature_points_per_face"] == 1
        case = result["cases"][0]
        column, surface = directory / "columns-0.1-0.npz", directory / "surface-fields-0.1-0.npz"
        record(column)
        record(surface)
        measured = measure_case(case, column, surface)
        assert (
            measured["surface_arrays_reintegrated"] and len(measured["rows"]) == 7 and all(measured["checks"].values())
        )
        checks[method + "/complete_surface_and_resultant_reintegration"] = True
        for index, raw in enumerate(case["raw"]):
            for key in ("mesh", "field"):
                record(raw[key]["path"], raw[key])
            key = raw["mesh"]["sha256"], raw["field"]["sha256"], method == "shell" and index == 0
            if key not in raw_cache:
                raw_cache[key] = raw_stress_peaks(raw["mesh"]["path"], raw["field"]["path"], shell=key[-1])
            check_raw_receipt(raw, raw_cache[key], require_positions=False)
            checks[f"{method}/raw_{index}/independent_invariants_and_retained_positions"] = True
        details.append({"method": method, "q1_control": measured})
    generator = np.random.default_rng(623016)
    bound_cases = 0
    for error in (0.0, 0.08, 0.25, 1.0):
        for uncertainty in (0.0, 0.02, 0.09):
            for measurement in (0.0, 0.003):
                lower, upper = conditional_interval(error, uncertainty, measurement)
                for _ in range(100):
                    reference = generator.normal(size=8)
                    norm = np.linalg.norm(reference)
                    directions = generator.normal(size=(3, 8))
                    directions /= np.linalg.norm(directions, axis=1)[:, None]
                    reported_method = reference + error * norm * directions[0]
                    true_reference = reference + uncertainty * norm * generator.uniform() * directions[1]
                    true_method = reported_method + measurement * norm * generator.uniform() * directions[2]
                    actual = np.linalg.norm(true_method - true_reference) / np.linalg.norm(true_reference)
                    assert lower - 1e-12 <= actual <= upper + 1e-12
                    bound_cases += 1
    checks["2400_independent_vector_triangle_inequalities"] = bound_cases == 2400
    checks["touching_intervals_remain_unresolved"] = pair_relation((0.1, 0.2), (0.2, 0.3)) == "unresolved"
    checks["separated_intervals_order_correctly"] = (
        pair_relation((0.1, 0.2), (0.3, 0.4)) == "first_lower"
        and pair_relation((0.3, 0.4), (0.1, 0.2)) == "second_lower"
    )
    components = (0.02, 0.005, 0.001, 0.0001, 0.0001)
    checks["multiplicative_envelope_at_least_observed_sum"] = uncertainty_product(components) >= sum(components)
    for label, values in (
        ("negative_error", (-1, 0.01, 0)),
        ("uncertainty_one", (0.1, 1, 0)),
        ("negative_uncertainty", (0.1, -0.01, 0)),
        ("nonfinite", (float("nan"), 0.01, 0)),
        ("negative_measurement", (0.1, 0.01, -0.001)),
    ):
        rejected = False
        try:
            conditional_interval(*values)
        except ValueError:
            rejected = True
        checks[label + "/invalid_bound_rejected"] = rejected
    passed = all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "details": details,
            "receipts": list(receipts.values()),
            "raw_peak_controls": list(raw_cache.values()),
            "independent_vector_bound_cases": bound_cases,
            "fine_reference_qualified": False,
            "scope": "Reproduces all81 published dimensional normal errors without changing their q9/q25 definitions; independently reintegrates complete q1 outer/inner/resultant arrays and raw invariant/location samples. Legacy missing q9 surfaces are explicit in the regression only; final readers require full surface exports. Random vector inequalities test conditional interval mathematics, not physical uncertainty qualification or statistical confidence. Pending fine friction errors/ranks remain uncomputed.",
        },
    )
    if not passed:
        raise SystemExit("Complete-array stress error or conditional-bound controls failed")


if __name__ == "__main__":
    main()
