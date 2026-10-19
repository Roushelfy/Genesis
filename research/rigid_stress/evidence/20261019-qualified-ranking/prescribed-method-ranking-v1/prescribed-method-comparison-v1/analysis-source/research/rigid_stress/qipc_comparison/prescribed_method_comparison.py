"""Audit qualified C2 method errors and signed tensor contributions at common sampling."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = {}

    def record(path, expected=None):
        path = Path(path)
        if str(path) not in receipts:
            receipts[str(path)] = file_id(path)
        if expected is not None:
            assert receipts[str(path)] == expected, f"Changed input: {path}"

    def read(relative):
        path = args.data_root / relative
        record(path)
        return json.loads(path.read_text())

    qualification = read("finite-area-prescribed-reference-qualification-v1/group0/qualification/result.json")
    assert qualification["prescribed_reference_change_targets_passed"] and all(qualification["checks"].values())
    for receipt in qualification["receipts"]:
        record(receipt["path"], receipt)
    envelopes = {row["metric"]: row["observed_reference_change_envelope"] for row in qualification["envelopes"]}
    targets = read("meta/finite-area-prescribed-decomposition-targets-v1.json")
    analytic = read("column-decomposition-controls-v1/group0/analytic/result.json")
    assert analytic["passed"] and len(analytic["cases"]) == 7
    reference = read("finite-area-prescribed-final125-v1/group0/reference/result.json")
    final_field = next(c["field"] for c in reference["cases"] if c["quadrature_order"] == 10)
    native = read("finite-area-native-pcg-v1/group0/native/result.json")
    native_start = read("finite-area-native-pcg-v1/group0/native/start.json")
    assert native_start["radius_m"] == 0.002
    native_cases = {case["route"]: case for case in native["cases"]}
    assert set(native_cases) == {"core", "dpr", "scr"}
    assert all(c["native_valid"] and c["errno"] == 0 and c["load_n"] == 0.1 for c in native_cases.values())
    assert native_cases["core"]["core_maximum_stress_relative_difference"] <= 1e-8
    assert all(
        len(native_cases[m]["contact_status"]) == 2 and not any(native_cases[m]["contact_status"])
        for m in ("dpr", "scr")
    )
    shell = read("finite-area-prescribed-shell250-v2/group0/shell/result.json")
    shell_start = read("finite-area-prescribed-shell250-v2/group0/shell/start.json")
    assert shell["static_equations_admitted"] and shell["load_n"] == 0.1
    assert shell_start["args"]["coarse_patches"] == 12 and not shell_start["args"]["solve_pad_contact"]
    journal = args.data_root / "finite-area-prescribed-shell250-v2/group0/shell/steps.jsonl"
    record(journal)
    shell_steps = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(shell_steps) == shell["iterations"] and all(not any(row["solver_failure_flags"]) for row in shell_steps)
    for parent in (
        "finite-area-native-inputs-v1/group0/reference",
        "finite-area-cholesky-refinement-v1/group0/surface250",
    ):
        saved = read(parent + "/result.json")
        assert saved["radius_m"] == 0.002
        assert all(case["elastic_equations_admitted"] for case in saved["cases"])
    rows, splits, details, array_checks, checks = [], [], [], [], {}
    reference_values = {}

    def close(first, second):
        np.testing.assert_allclose(first, second, rtol=1e-9, atol=0)

    def add(method, metric, absolute, denominator, sampling, delta, unit):
        assert np.isfinite([absolute, denominator, sampling, delta]).all()
        assert absolute >= 0 and denominator > 0 and sampling >= 0 and 0 <= delta < 1
        reference_values.setdefault(metric, denominator)
        close(denominator, reference_values[metric])
        relative = absolute / denominator
        rows.append(
            {
                "load_n": 0.1,
                "method": method,
                "metric": metric,
                "absolute_error": float(absolute),
                "unit": unit,
                "reference_rms_or_peak": float(denominator),
                "relative_error": float(relative),
                "observed_reference_envelope": delta,
                "observed_method_sampling_change_over_reference": float(sampling),
                "conditional_lower": max(relative - delta - sampling, 0) / (1 + delta),
                "conditional_upper": (relative + delta + sampling) / (1 - delta),
            }
        )

    for group, method in enumerate(("dpr", "scr", "shell")):
        samples = []
        for q in (9, 25):
            folder = (
                f"finite-area-prescribed-native-decomposition-v1/group{group}/{method}-q{q}"
                if method != "shell"
                else f"finite-area-prescribed-shell-decomposition-v1/group0/shell-q{q}"
            )
            result = read(folder + "/result.json")
            assert result["physical_volume_weighting"] and result["common_peaks_evaluated"]
            assert result["surface_quadrature_points_per_face"] == q and result["depth_quadrature_points"] == 2
            assert len(result["cases"]) == 1
            case = result["cases"][0]
            assert case["load_n"] == 0.1 and len(case["raw"]) == 3 and len(case["adjacent_comparisons"]) == 2
            assert case["raw"][2]["field"] == final_field
            assert case["raw"][0]["mesh"]["sha256"] == case["raw"][1]["mesh"]["sha256"]
            actual = native_cases[method]["field"] if method != "shell" else shell["field"]
            control = native_cases[method]["reference_field"] if method != "shell" else shell["source_field"]
            assert case["raw"][0]["field"] == actual and case["raw"][1]["field"] == control
            for raw in case["raw"]:
                for key in ("mesh", "field"):
                    record(raw[key]["path"], raw[key])
            decomposition = case["three_field_decomposition"]
            assert decomposition["passed_tensor_identity"]
            metrics = decomposition["metrics"]
            for side, adjacent in zip(("first", "second"), case["adjacent_comparisons"]):
                for metric, key in (
                    ("tangent", "tangent_rms_error_pa"),
                    ("membrane", "membrane_rms_error_n_per_m"),
                    ("bending", "bending_rms_error_n"),
                    ("outer_tangent", "outer_tangent_rms_error_pa"),
                    ("inner_tangent", "inner_tangent_rms_error_pa"),
                ):
                    close(metrics[metric][side + "_difference_rms"], adjacent[key])
            # Independently reconstruct signed resultant differences from the
            # two saved complete-column arrays, not from reported scalar norms.
            paths = [args.data_root / folder / f"columns-0.1-{i}.npz" for i in (0, 1)]
            for path in paths:
                record(path)
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                area = a["area_m2"]
                np.testing.assert_array_equal(area, b["area_m2"])
                for metric, key in (("membrane", "membrane_n_per_m"), ("bending", "bending_n")):
                    left, right = a[key], b[key]

                    def inner(x, y):
                        return np.einsum("qij,qij,q->", x, y, area) / area.sum()

                    control_difference = left[1] - right[0]
                    control_relative = np.sqrt(
                        inner(control_difference, control_difference) / inner(right[0], right[0])
                    )
                    assert control_relative < 1e-10
                    first, second, total = left[0] - left[1], right[0] - right[1], left[0] - right[1]
                    row = metrics[metric]
                    for name, value in (("first", first), ("second", second), ("total", total)):
                        close(np.sqrt(inner(value, value)), row[name + "_difference_rms"])
                    cross = 2 * inner(first, second)
                    cross_scale = 2 * np.sqrt(inner(first, first) * inner(second, second))
                    assert abs(cross - row["twice_cross_mean_square"]) <= 1e-9 * cross_scale
                    array_checks.append(
                        {
                            "method": method,
                            "quadrature": q,
                            "metric": metric,
                            "control_relative_integration_difference": float(control_relative),
                            "independent_twice_cross_mean_square": float(cross),
                            "passed": True,
                        }
                    )
            samples.append(case)
        low, high = samples
        assert low["raw"] == high["raw"]
        first_peak, fine_peak = [c["common_spatial_measures"]["tangent_peaks"] for c in high["adjacent_comparisons"]]
        low_first = low["adjacent_comparisons"][0]["common_spatial_measures"]["tangent_peaks"]
        for metric, row in high["three_field_decomposition"]["metrics"].items():
            previous = low["three_field_decomposition"]["metrics"][metric]
            assert not row["zero_reference_norm"] and row["identity_relative_defect"] <= 1e-12
            sampling = abs(row["total_relative_l2"] - previous["total_relative_l2"])
            checks[f"{method}_{metric}_sampling"] = sampling <= targets["sampling_field_target"]
            add(
                method,
                metric,
                row["total_difference_rms"],
                row["reference_rms"],
                sampling,
                envelopes[metric + "_relative_l2"],
                row["unit"],
            )
            splits.append({"method": method, "metric": metric, **row})
        for metric in ("mises", "tension"):
            actual, fine = [high["raw"][i][f"raw_maximum_{metric}_pa"] for i in (0, 2)]
            add(method, "raw_" + metric, abs(actual - fine), fine, 0, envelopes[f"raw_{metric}_relative_change"], "Pa")
            a = next(p for p in first_peak if p["metric"] == metric)
            b = next(p for p in fine_peak if p["metric"] == metric)
            before = next(p for p in low_first if p["metric"] == metric)
            close(a["reference"]["stress_pa"], b["actual"]["stress_pa"])
            actual, fine = a["actual"]["stress_pa"], b["reference"]["stress_pa"]
            sampling = abs(actual - before["actual"]["stress_pa"])
            checks[f"{method}_common_{metric}_sampling"] = sampling / actual <= targets["sampling_peak_target"]
            add(
                method,
                "common_" + metric,
                abs(actual - fine),
                fine,
                sampling / fine,
                envelopes["common_" + metric],
                "Pa",
            )
        details.append(
            {
                "method": method,
                "q25_raw": high["raw"],
                "q25_pairwise_common_measures": [c["common_spatial_measures"] for c in high["adjacent_comparisons"]],
                "scope": "First-pair3D-only components compare candidate/control, second pair control/fine; no total3D decomposition or Shell transverse rank is inferred.",
            }
        )
    assert len(rows) == 27 and len(splits) == 15 and len(array_checks) == 12
    rankings = []
    for metric in reference_values:
        selected = {r["method"]: r for r in rows if r["metric"] == metric}
        pairs = []
        for first, second in (("dpr", "scr"), ("dpr", "shell"), ("scr", "shell")):
            relation = "unresolved"
            if selected[first]["conditional_upper"] < selected[second]["conditional_lower"]:
                relation = first + " lower error"
            elif selected[second]["conditional_upper"] < selected[first]["conditional_lower"]:
                relation = second + " lower error"
            pairs.append({"first": first, "second": second, "relation": relation})
        rankings.append({"metric": metric, "pairs": pairs})
    for name, data in (("errors", rows), ("decomposition", splits)):
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    passed = all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "errors": rows,
            "decomposition": splits,
            "independent_resultant_array_checks": array_checks,
            "conditional_rankings": rankings,
            "details": details,
            "receipts": list(receipts.values()),
            "certified_continuum_bound": False,
            "whole_goal_complete": False,
            "scope": "Prescribed C2 load only,0.1N per pole and fixed2mm radius. Native two-source reconstruction "
            "uses the original coarse solid mesh; Shell is the admitted250um velocity-cleared static replay. "
            "Native first differences measure source-profile reconstruction relative to the independent "
            "same-mesh load; Shell first differences combine thin-shell formulation/kinematics and load transfer. "
            "Second differences measure their separate solid controls' discretization/geometry differences. "
            "Signed tensor cross terms are measured, not inferred by scalar-error subtraction. Ranks are "
            "conditional on observed refinement/sampling envelopes bounding errors. This neither attributes "
            "physical-pad contact error by subtracting these norms nor qualifies sustained-friction dynamics.",
        },
    )
    assert passed, "Prescribed method sampling gates failed; measured evidence retained"


if __name__ == "__main__":
    main()
