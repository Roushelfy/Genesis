"""Audit dimensional normal/friction errors and conditional ranks on the qualified fine reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .pad_candidate_admission import admit_candidates
from .stress_error_measures import (
    check_raw_receipt,
    conditional_interval,
    measure_case,
    pair_relation,
    raw_stress_peaks,
    uncertainty_product,
)

METRICS = (
    "tangent",
    "membrane",
    "bending",
    "outer_tangent",
    "inner_tangent",
    "raw_mises",
    "raw_tension",
    "common_mises",
    "common_tension",
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--reference-qualification", type=Path, required=True)
    cli.add_argument("--fields-campaign", default="canonical-stvk-method-fields-v2")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, receipts, cases, error_rows, details, rankings, raw_cache = {}, {}, {}, [], [], [], {}
    candidates, failures, error, reference_accepted = {}, [], None, False
    reference_mass = None

    def verify(expected=None, *, path=None):
        path = Path(path if expected is None else expected["path"])
        if str(path) not in receipts:
            receipts[str(path)] = file_id(path)
        if expected is not None:
            assert receipts[str(path)] == expected, f"Changed input: {path}"
        return receipts[str(path)]

    def read(path, *, journal=False):
        verify(path=path)
        data = Path(path).read_text()
        return [json.loads(line) for line in data.splitlines()] if journal else json.loads(data)

    def check(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    try:
        qualified = read(args.reference_qualification)
        check(
            "fine_reference_physical_and_resolution_admission",
            qualified["passed"] and qualified["fine_reference_qualified"] and all(qualified["checks"].values()),
        )
        reference_accepted = True
        for receipt in (*qualified["receipts"], *qualified["verified_inputs"]):
            verify(receipt)
        accepted = qualified["accepted_reference_frames"]
        check("two_distinct_admitted_reference_endpoints", len(accepted) == 2 and accepted[0] != accepted[1])
        bound = {receipt["path"]: receipt for receipt in qualified["verified_inputs"]}
        reference_frames = [bound[path] for path in accepted]
        reference_starts = [read(Path(path).parent.parent / "start.json") for path in accepted]
        reference_mesh = reference_starts[0]["mesh"]
        reference_mass = reference_starts[0]["mass_kg"]
        check(
            "both_reference_endpoints_share_admitted_geometry",
            all(start["mesh"] == reference_mesh for start in reference_starts),
        )
        verify(reference_mesh)
        envelope = {(row["level"], row["metric"]): row for row in qualified["reference_envelopes"]}
        check("eighteen_metric_endpoint_reference_envelopes", len(envelope) == 18)
        candidates, failures = admit_candidates(args.data_root, read, verify, check)
        physical_keys = (
            "loads",
            "dt",
            "ramp",
            "hold",
            "damping",
            "clearance",
            "plate_offset_x",
            "shear_distance",
            "record_stride",
            "friction",
            "gravity_x",
            "gravity_start",
            "gravity_ramp",
        )
        for method, candidate in candidates.items():
            candidate["mass_relative_difference_from_reference"] = candidate["mass_kg"] / reference_mass - 1
            check(
                method + "/same_declared_physical_load_history",
                all(candidate["configuration"][key] == reference_starts[0]["args"][key] for key in physical_keys),
            )
        reference_norms = {}
        for level in range(2):
            for group, method in enumerate(("dpr", "scr", "shell")):
                measured, source_cases = {}, {}
                for order in (9, 25):
                    directory = (
                        args.data_root
                        / args.fields_campaign
                        / f"group{3 * level + group}"
                        / f"{method}-level{level}-q{order}"
                    )
                    result = read(directory / "result.json")
                    check(
                        f"{method}/{level}/q{order}/declared_complete_sampling",
                        result["surface_quadrature_points_per_face"] == order
                        and result["depth_quadrature_points"] == 4
                        and result["physical_volume_weighting"]
                        and result["common_peaks_evaluated"]
                        and result["shell_indices"] == ([0] if method == "shell" else []),
                    )
                    assert len(result["cases"]) == 1
                    case = result["cases"][0]
                    expected_frames = [candidates[method]["frames"][level], reference_frames[level]]
                    check(
                        f"{method}/{level}/q{order}/correct_ordered_own_frames",
                        case["load_n"] == 0.1
                        and len(case["raw"]) == 2
                        and [raw["field"] for raw in case["raw"]] == expected_frames,
                    )
                    check(
                        f"{method}/{level}/q{order}/correct_candidate_mesh",
                        case["raw"][0]["mesh"] == candidates[method]["mesh"],
                    )
                    check(f"{method}/{level}/q{order}/correct_reference_mesh", case["raw"][1]["mesh"] == reference_mesh)
                    for index, raw in enumerate(case["raw"]):
                        for name in ("mesh", "field"):
                            verify(raw[name])
                        key = raw["mesh"]["sha256"], raw["field"]["sha256"], method == "shell" and index == 0
                        if key not in raw_cache:
                            raw_cache[key] = raw_stress_peaks(raw["mesh"]["path"], raw["field"]["path"], shell=key[-1])
                        check_raw_receipt(raw, raw_cache[key], require_positions=True)
                    column, surface = directory / "columns-0.1-0.npz", directory / "surface-fields-0.1-0.npz"
                    verify(path=column)
                    verify(path=surface)
                    values = measure_case(case, column, surface, require_surfaces=True)
                    check(
                        f"{method}/{level}/q{order}/all_complete_array_identities",
                        values["surface_arrays_reintegrated"]
                        and len(values["rows"]) == 9
                        and all(values["checks"].values()),
                    )
                    measured[order] = {row["metric"]: row for row in values["rows"]}
                    source_cases[order] = case
                    if order == 25:
                        details.append(
                            {
                                "method": method,
                                "level": level,
                                "raw": case["raw"],
                                "common_spatial_measures": case["adjacent_comparisons"][0]["common_spatial_measures"],
                                "column_count": values["column_count"],
                                "column_receipt": verify(path=column),
                                "surface_receipt": verify(path=surface),
                                "array_checks": values["checks"],
                            }
                        )
                cases[level, method] = measured
                for metric in METRICS:
                    low, high = measured[9][metric], measured[25][metric]
                    qualification_key = (
                        metric
                        if metric.startswith("common_")
                        else metric + ("_relative_change" if metric.startswith("raw_") else "_relative_l2")
                    )
                    observed = envelope[level, qualification_key]
                    components = [
                        observed[name]
                        for name in (
                            "spatial_change",
                            "time_change",
                            "contact_change",
                            "angular_change",
                            "depth_change",
                        )
                    ]
                    np.testing.assert_allclose(
                        sum(components), observed["observed_resolution_envelope"], rtol=1e-12, atol=1e-15
                    )
                    reference_change = uncertainty_product(components)
                    rms_sampling = abs(low["reference_rms_or_peak"] / high["reference_rms_or_peak"] - 1)
                    method_sampling = abs(low["relative_error"] - high["relative_error"])
                    if metric.startswith("common_"):
                        peak_name = metric.removeprefix("common_")
                        low_peak = next(
                            p
                            for p in source_cases[9]["adjacent_comparisons"][0]["common_spatial_measures"][
                                "tangent_peaks"
                            ]
                            if p["metric"] == peak_name
                        )
                        high_peak = next(
                            p
                            for p in source_cases[25]["adjacent_comparisons"][0]["common_spatial_measures"][
                                "tangent_peaks"
                            ]
                            if p["metric"] == peak_name
                        )
                        method_sampling = (
                            abs(low_peak["actual"]["stress_pa"] - high_peak["actual"]["stress_pa"])
                            / high["reference_rms_or_peak"]
                        )
                        check(
                            f"{method}/{level}/{metric}/actual_peak_sampling",
                            abs(low_peak["actual"]["stress_pa"] / high_peak["actual"]["stress_pa"] - 1) <= 0.01,
                        )
                    target = 0.01 if metric.startswith(("raw_", "common_")) else 0.005
                    check(
                        f"{method}/{level}/{metric}/measurement_sampling_target",
                        method_sampling <= target and rms_sampling <= target,
                    )
                    reference_change = (1 + reference_change) * (1 + rms_sampling) - 1
                    interval = conditional_interval(high["relative_error"], reference_change, method_sampling)
                    reference_key = level, metric
                    reference_norms.setdefault(reference_key, high["reference_rms_or_peak"])
                    np.testing.assert_allclose(
                        high["reference_rms_or_peak"], reference_norms[reference_key], rtol=1e-9, atol=0
                    )
                    check(f"{method}/{level}/{metric}/shared_reference_norm", True)
                    error_rows.append(
                        {
                            "level": level,
                            "case": "normal0.1N" if level == 0 else "sustained_friction0.1N",
                            "load_n": 0.1,
                            "method": method,
                            **high,
                            "reference_observed_change_sum": observed["observed_resolution_envelope"],
                            "conditional_reference_envelope": reference_change,
                            "method_sampling_change": method_sampling,
                            "reference_norm_sampling_change": rms_sampling,
                            "conditional_lower": interval[0],
                            "conditional_upper": interval[1],
                            "surface_quadrature_points_per_face": 25,
                            "depth_quadrature_points": 4,
                        }
                    )
        check("complete54_dimensional_error_rows", len(error_rows) == 54 and len(details) == 6)
        indexed = {(row["level"], row["method"], row["metric"]): row for row in error_rows}
        for level in range(2):
            for metric in METRICS:
                pairs = []
                for first, second in (("dpr", "scr"), ("dpr", "shell"), ("scr", "shell")):
                    a, b = indexed[level, first, metric], indexed[level, second, metric]
                    relation = pair_relation(
                        (a["conditional_lower"], a["conditional_upper"]),
                        (b["conditional_lower"], b["conditional_upper"]),
                    )
                    label = (
                        first + " lower error"
                        if relation == "first_lower"
                        else second + " lower error"
                        if relation == "second_lower"
                        else "unresolved"
                    )
                    pairs.append({"first": first, "second": second, "relation": label})
                rankings.append({"level": level, "metric": metric, "pairs": pairs})
    except (OSError, KeyError, ValueError, AssertionError) as failure:
        error = str(failure)
        checks["complete_qualified_error_contract"] = False
    passed = bool(checks) and all(checks.values())
    if passed:
        with (args.output / "errors.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(error_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(error_rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "errors_admitted": passed,
            "reference_qualification_accepted": reference_accepted,
            "reference_mass_kg": reference_mass,
            "checks": checks,
            "error": error,
            "rows": error_rows,
            "conditional_rankings": rankings if passed else [],
            "details": details,
            "candidates": candidates,
            "retained_every_step_failures": failures,
            "receipts": list(receipts.values()),
            "independent_raw_peaks": list(raw_cache.values()),
            "certified_continuum_error_bound": False,
            "whole_goal_complete": False,
            "scope": "Two matched0.1N normal/friction endpoint cases, three own-admitted candidate fields and nine dimensional/relative metrics each. All volume-column integrals, complete membrane/bending and outer/inner arrays are re-reduced. Full3D raw invariants/locations are independently checked; unsupported Shell transverse components remain separate without a rank. Conditional reference envelopes use the conservative product of the predeclared observed changes plus measured reference-norm quadrature sensitivity; method quadrature error is added separately. Intervals assume these observed envelopes bound unresolved errors; they are not statistical confidence or continuum proofs. Overlap remains unresolved. Native fields cover requested endpoints, never the retained failed every-step mode. Warm cost runs and collider production rates retain their separate configurations/horizons; no speed or universal ranking is inferred here.",
        },
    )
    if not passed:
        raise SystemExit("Qualified physical error/ranking contract failed; no ranks admitted")


if __name__ == "__main__":
    main()
