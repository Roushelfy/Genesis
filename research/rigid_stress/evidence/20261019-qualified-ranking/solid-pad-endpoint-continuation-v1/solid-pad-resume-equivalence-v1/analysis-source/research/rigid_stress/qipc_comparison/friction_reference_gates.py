"""Predeclared complete-field/peak gates shared by friction qualification and its controls."""

import math

from .pad_normal_qualification import FIELDS, PEAKS

SPATIAL_FIELD_TARGET = 0.02
SPATIAL_PEAK_TARGET = 0.05
CONTROL_FIELD_TARGET = 0.005
CONTROL_PEAK_TARGET = 0.01


def tangent_peaks(comparison):
    for key in FIELDS + PEAKS:
        if not math.isfinite(comparison[key]) or comparison[key] < 0:
            raise ValueError(f"Invalid complete-field/peak change: {key}")
    rows = comparison["common_spatial_measures"]["tangent_peaks"]
    if len(rows) != 2 or {row["metric"] for row in rows} != {"mises", "tension"}:
        raise ValueError("Both complete common tangent peak definitions are required")
    for row in rows:
        if not math.isfinite(row["relative_difference"]) or row["relative_difference"] < 0:
            raise ValueError("Invalid common-peak change")
        for side in ("actual", "reference"):
            if not math.isfinite(row[side]["stress_pa"]) or row[side]["stress_pa"] <= 0:
                raise ValueError("Positive finite peaks are required for these nonzero-load cases")
            point = row[side]["common_reference_rest_position_m"]
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError("Common peak location must be a finite material point")
    return {row["metric"]: row for row in rows}


def ordered_fields_match(case, expected):
    """The actual/reference order fixes the denominator and endpoint identity."""
    return len(case["raw"]) == 2 and [row["field"] for row in case["raw"]] == list(expected)


def integration_changes(low, high):
    low_peaks, high_peaks = tangent_peaks(low), tangent_peaks(high)
    fields = {key: abs(low[key] - high[key]) for key in FIELDS}
    peaks = {
        metric + "/" + side: abs(low_peaks[metric][side]["stress_pa"] / high_peaks[metric][side]["stress_pa"] - 1)
        for metric in high_peaks
        for side in ("actual", "reference")
    }
    return {"fields": fields, "common_peaks": peaks}


def assess_pair(low, high, *, spatial):
    """Keep all fields and both raw/common peaks; no fitted target or omitted component."""
    field_target = SPATIAL_FIELD_TARGET if spatial else CONTROL_FIELD_TARGET
    peak_target = SPATIAL_PEAK_TARGET if spatial else CONTROL_PEAK_TARGET
    peaks = tangent_peaks(high)
    changes = integration_changes(low, high)
    values = [high[key] for key in FIELDS + PEAKS] + [row["relative_difference"] for row in peaks.values()]
    checks = {"finite_nonnegative_metrics": all(math.isfinite(value) and value >= 0 for value in values)}
    checks.update({key: 0 <= high[key] <= field_target for key in FIELDS})
    checks.update({key: 0 <= high[key] <= peak_target for key in PEAKS})
    checks.update({"common_" + key: 0 <= row["relative_difference"] <= peak_target for key, row in peaks.items()})
    checks["angular_field_contrasts"] = all(
        math.isfinite(v) and v <= CONTROL_FIELD_TARGET for v in changes["fields"].values()
    )
    checks["angular_peak_values"] = all(
        math.isfinite(v) and v <= CONTROL_PEAK_TARGET for v in changes["common_peaks"].values()
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "field_target": field_target,
        "peak_target": peak_target,
        "angular_changes": changes,
    }
