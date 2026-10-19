"""Audit completed coarse StVK time/contact controls at common q9/q25 sampling."""

import argparse
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .pad_normal_qualification import FIELDS, PEAKS


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checked_fields, controls = [], {}, []
    checks = {}

    def read(relative):
        path = args.data_root / relative
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    for name, group, steps in (
        ("solid-pad-stvk-v1", 0, 400),
        ("solid-pad-stvk-half-dt-v1", 0, 800),
        ("solid-pad-stvk-contact-quadrature-v1", 0, 400),
        ("solid-pad-stvk-contact-quadrature-v1", 1, 400),
    ):
        history = read(f"{name}/group{group}/reference/result.json")
        moment = read(f"{name}/group{group}/moment/result.json")
        assert moment["run"] == file_id(args.data_root / f"{name}/group{group}/reference/result.json")
        checks[f"history_{name}_{group}"] = bool(
            history["completed"]
            and history["steps"] == steps
            and history["mechanical_history_passed"]
            and all(history["history_checks"].values())
            and len(history["windows"]) == 2
            and all(w["mechanical_window_passed"] for w in history["windows"])
            and moment["passed"]
            and len(moment["frames"]) == 2
        )
        for frame in moment["frames"]:
            expected = frame["field"]
            actual = file_id(Path(expected["path"]))
            assert actual == expected, "Reference field changed after its independent momentum audit"
            checked_fields[expected["path"]] = expected
    for group, name in enumerate(("half-dt", "contact-orders")):
        for level in range(2):
            samples = []
            for order in (9, 25):
                result = read(f"stvk-time-contact-common-fields-v1/group{group}/{name}-{level}-q{order}/result.json")
                assert result["physical_volume_weighting"] and result["common_peaks_evaluated"]
                assert result["surface_quadrature_points_per_face"] == order
                assert result["depth_quadrature_points"] == 4 and len(result["cases"]) == 1
                case = result["cases"][0]
                assert case["load_n"] == 0.1 and len(case["adjacent_comparisons"]) == 1
                for raw in case["raw"]:
                    assert raw["field"] == checked_fields[raw["field"]["path"]]
                samples.append(case["adjacent_comparisons"][0])
            low, high = samples
            peaks = high["common_spatial_measures"]["tangent_peaks"]
            checks[f"{name}_{level}_fields"] = all(0 <= high[k] <= 0.005 for k in FIELDS)
            checks[f"{name}_{level}_raw_peaks"] = all(0 <= high[k] <= 0.01 for k in PEAKS)
            checks[f"{name}_{level}_common_peaks"] = all(0 <= p["relative_difference"] <= 0.01 for p in peaks)
            angular = {}
            for lower, upper in zip(low["common_spatial_measures"]["tangent_peaks"], peaks):
                assert lower["metric"] == upper["metric"]
                for side in ("actual", "reference"):
                    angular[upper["metric"] + "_" + side] = abs(lower[side]["stress_pa"] / upper[side]["stress_pa"] - 1)
            angular_fields = {k: abs(low[k] - high[k]) for k in FIELDS}
            checks[f"{name}_{level}_angular"] = all(v <= 0.01 for v in angular.values()) and all(
                v <= 0.005 for v in angular_fields.values()
            )
            depth = []
            for order in (2, 4):
                result = read(f"stvk-time-quadrature-fields-v1/group{group}/{name}-{level}-depth{order}/result.json")
                assert result["surface_quadrature_points_per_face"] == 3
                assert result["depth_quadrature_points"] == order
                assert result["physical_volume_weighting"] and result["common_peaks_evaluated"]
                for raw in result["cases"][0]["raw"]:
                    assert raw["field"] == checked_fields[raw["field"]["path"]]
                depth.append(result["cases"][0]["adjacent_comparisons"][0])
            depth_change = {k: abs(depth[0][k] - depth[1][k]) for k in FIELDS}
            checks[f"{name}_{level}_depth"] = all(v <= 0.005 for v in depth_change.values())
            controls.append(
                {
                    "control": name,
                    "level": level,
                    "fields": {k: high[k] for k in FIELDS},
                    "raw_peaks": {k: high[k] for k in PEAKS},
                    "common_tangent_peaks": {p["metric"]: p["relative_difference"] for p in peaks},
                    "common_peak_q9_to_q25_relative_changes": angular,
                    "field_contrast_q9_to_q25_absolute_changes": angular_fields,
                    "field_contrast_depth2_to_depth4_at_q3_absolute_changes": depth_change,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "checks": checks,
            "passed": all(checks.values()),
            "controls": controls,
            "receipts": receipts,
            "verified_reference_fields": list(checked_fields.values()),
            "fine_reference_qualified": False,
            "whole_goal_complete": False,
            "scope": "Coarse one-wall StVK reference only: matched1/0.5ms and contact-order6/10 complete histories with independent momentum audits, re-evaluated using q9/q25 common angular sampling and depth4. Unchanged0.5% field/1% peak control targets. Earlier depth2/4 controls use q3 and remain separately identified. This neither measures fine-mesh time/contact sensitivity nor supplies a continuum bound or method error rank.",
        },
    )
    assert all(checks.values()), "Time/contact controls fail; preserve all actual results"


if __name__ == "__main__":
    main()
