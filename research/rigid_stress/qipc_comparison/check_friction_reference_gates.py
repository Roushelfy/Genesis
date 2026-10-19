"""Use admitted real controls and a known unresolved mesh to check qualification gates."""

import argparse
import copy
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .friction_reference_gates import assess_pair, ordered_fields_match


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, receipts, pairs = {}, [], []

    def case(relative):
        path = args.data_root / relative
        receipts.append(file_id(path))
        result = json.loads(path.read_text())
        assert result["physical_volume_weighting"] and result["common_peaks_evaluated"]
        assert len(result["cases"]) == 1
        return result["cases"][0]

    for level in range(2):
        for kind, prefix in (
            ("wall", f"stvk-wall250-fields-v1/group{level}/wall250-level{level}"),
            ("surface", f"stvk-spatial250-fields-v1/group{level}/surface500to250-level{level}"),
        ):
            low, high = [case(prefix + f"-q{order}/result.json") for order in (9, 25)]
            pair = assess_pair(low["adjacent_comparisons"][0], high["adjacent_comparisons"][0], spatial=True)
            label = f"{kind}/{level}"
            checks[label + "/retains_known_physical_outcome"] = pair["passed"] if kind == "wall" else not pair["passed"]
            if kind == "surface":
                checks[label + "/rejects_2point749_percent_tangent_change"] = not pair["checks"]["tangent_relative_l2"]
                checks[label + "/passing_peaks_do_not_override_field_failure"] = all(
                    pair["checks"][key]
                    for key in (
                        "raw_mises_relative_change",
                        "raw_tension_relative_change",
                        "common_mises",
                        "common_tension",
                    )
                )
            pairs.append({"label": label, **pair})
    for group, name in enumerate(("half-dt", "contact-orders")):
        for level in range(2):
            low, high = [
                case(f"stvk-time-contact-common-fields-v1/group{group}/{name}-{level}-q{order}/result.json")[
                    "adjacent_comparisons"
                ][0]
                for order in (9, 25)
            ]
            pair = assess_pair(low, high, spatial=False)
            checks[f"{name}/{level}/retains_admitted_control"] = pair["passed"]
            pairs.append({"label": f"{name}/{level}", **pair})
    valid = case("stvk-wall250-fields-v1/group0/wall250-level0-q25/result.json")
    expected = [row["field"] for row in valid["raw"]]
    checks["correct_ordered_frame_receipts"] = ordered_fields_match(valid, expected)
    checks["swapped_denominator_or_endpoint_rejected"] = not ordered_fields_match(valid, list(reversed(expected)))
    altered = copy.deepcopy(expected)
    altered[1]["sha256"] = "0" * 64
    checks["altered_field_receipt_rejected"] = not ordered_fields_match(valid, altered)
    for kind in ("missing_peak", "duplicate_peak", "nonfinite_field", "nonfinite_peak_position"):
        invalid = copy.deepcopy(valid["adjacent_comparisons"][0])
        peaks = invalid["common_spatial_measures"]["tangent_peaks"]
        if kind == "missing_peak":
            peaks.pop()
        elif kind == "duplicate_peak":
            peaks[1]["metric"] = peaks[0]["metric"]
        elif kind == "nonfinite_field":
            invalid["bending_relative_l2"] = float("nan")
        else:
            peaks[0]["reference"]["common_reference_rest_position_m"][0] = float("nan")
        rejected = False
        try:
            assess_pair(invalid, valid["adjacent_comparisons"][0], spatial=True)
        except (ValueError, KeyError):
            rejected = True
        checks[kind + "/fail_closed"] = rejected
    passed = all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "actual_pairs": pairs,
            "receipts": receipts,
            "fine_reference_qualified": False,
            "scope": "Actual250um thickness comparisons and coarse time/contact controls retain their passing targets; actual500-to250um surface changes remain rejected at the original2% field threshold despite passing peaks. Altered receipt, swapped normalization, missing/duplicate peak and nonfinite-data controls are in-memory copies only. No original artifact, acceptance target or physical state is changed, and no fixture qualifies the pending fine reference.",
        },
    )
    if not passed:
        raise SystemExit("Friction reference gate controls failed")


if __name__ == "__main__":
    main()
