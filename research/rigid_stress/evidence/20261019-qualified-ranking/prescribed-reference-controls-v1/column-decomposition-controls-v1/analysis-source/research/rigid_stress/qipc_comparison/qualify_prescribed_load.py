"""Check fixed reference-change targets for the independent C2 prescribed load."""

import argparse
import json
import math
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json
from .pad_normal_qualification import FIELDS, PEAKS


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

    targets = read("meta/finite-area-prescribed-final-targets-v1.json")
    assert targets["load_n"] == 0.1 and targets["radius_m"] == 0.002
    reference = read("finite-area-prescribed-final125-v1/group0/reference/result.json")
    assert reference["radius_m"] == targets["radius_m"]
    assert len(reference["cases"]) == 2
    assert {(c["load_n"], c["quadrature_order"]) for c in reference["cases"]} == {(0.1, 8), (0.1, 10)}
    record(reference["mesh"]["path"], reference["mesh"])
    for case in reference["cases"]:
        record(case["field"]["path"], case["field"])
    checks = {"final_reference_equations": all(c["elastic_equations_admitted"] for c in reference["cases"])}

    def comparison(relative, quadrature):
        result = read(relative)
        assert result["physical_volume_weighting"] and result["common_peaks_evaluated"]
        assert result["surface_quadrature_points_per_face"] == quadrature
        assert result["depth_quadrature_points"] == 2 and len(result["cases"]) == 1
        case = result["cases"][0]
        assert case["load_n"] == 0.1 and len(case["adjacent_comparisons"]) == 1
        for raw in case["raw"]:
            for name in ("mesh", "field"):
                record(raw[name]["path"], raw[name])
        value = case["adjacent_comparisons"][0]
        metrics = {key: value[key] for key in FIELDS + PEAKS}
        peaks = value["common_spatial_measures"]["tangent_peaks"]
        assert {p["metric"] for p in peaks} == {"mises", "tension"}
        metrics.update({"common_" + p["metric"]: p["relative_difference"] for p in peaks})
        assert all(math.isfinite(v) and v >= 0 for v in metrics.values())
        return {"metrics": metrics, "comparison": value, "raw": case["raw"]}

    controls = {}
    old_campaign = "finite-area-prescribed-common-controls-v1"
    final_campaign = "finite-area-prescribed-final-fields-cpu-v2"
    for campaign, group, name in (
        (old_campaign, 0, "surface500to250"),
        (old_campaign, 1, "surface250to125"),
        (old_campaign, 2, "wall250"),
        (final_campaign, 0, "wall125"),
        (final_campaign, 1, "surface3"),
        (final_campaign, 2, "joint250"),
        (final_campaign, 3, "load-quadrature"),
    ):
        for q in (9, 25):
            controls[name, q] = comparison(f"{campaign}/group{group}/{name}-q{q}/result.json", q)
    adapter = comparison("finite-area-prescribed-cvxopt-control-v1/group0/comparison/result.json", 9)
    checks["same_mesh_factor_adapter"] = all(
        value
        <= targets[
            "adapter_field_equivalence_target" if key in FIELDS else "adapter_raw_common_peak_equivalence_target"
        ]
        for key, value in adapter["metrics"].items()
    )
    checks["surface_two_layer_trend"] = all(
        controls["surface250to125", 25]["metrics"][key] < controls["surface500to250", 25]["metrics"][key]
        for key in FIELDS
    )
    checks["wall_refinement_trend"] = all(
        controls["wall125", 25]["metrics"][key] < controls["wall250", 25]["metrics"][key] for key in FIELDS
    )
    spatial_names = ("wall125", "surface3", "joint250")
    for name in spatial_names:
        checks[name + "_field_target"] = all(
            controls[name, 25]["metrics"][key] <= targets["field_target"] for key in FIELDS
        )
        checks[name + "_raw_common_peak_target"] = all(
            value <= targets["raw_common_peak_target"]
            for key, value in controls[name, 25]["metrics"].items()
            if key not in FIELDS
        )
    checks["prescribed_load_quadrature"] = all(
        value
        <= targets["solver_and_integration_field_target" if key in FIELDS else "solver_and_integration_peak_target"]
        for key, value in controls["load-quadrature", 25]["metrics"].items()
    )
    sampling = []
    for name in ("surface500to250", "surface250to125", "wall250", *spatial_names, "load-quadrature"):
        low, high = controls[name, 9], controls[name, 25]
        assert low["raw"] == high["raw"], "Sampling contrast changed physical inputs"
        field_change = max(abs(high["metrics"][key] - low["metrics"][key]) for key in FIELDS)
        peak_changes = []
        for a, b in zip(
            low["comparison"]["common_spatial_measures"]["tangent_peaks"],
            high["comparison"]["common_spatial_measures"]["tangent_peaks"],
        ):
            assert a["metric"] == b["metric"]
            for side in ("actual", "reference"):
                denominator = b[side]["stress_pa"]
                assert denominator > 0
                peak_changes.append(abs(a[side]["stress_pa"] / denominator - 1))
        checks[name + "_field_sampling"] = field_change <= targets["solver_and_integration_field_target"]
        checks[name + "_peak_sampling"] = max(peak_changes) <= targets["solver_and_integration_peak_target"]
        sampling.append(
            {
                "comparison": name,
                "maximum_absolute_relative_field_metric_change": field_change,
                "maximum_actual_or_reference_peak_sampling_change": max(peak_changes),
            }
        )
    envelopes = []
    for key in adapter["metrics"]:
        spatial = max(controls[name, 25]["metrics"][key] for name in spatial_names)
        quadrature = controls["load-quadrature", 25]["metrics"][key]
        if key in FIELDS or key in PEAKS:
            metric_sampling = max(
                abs(controls[name, 25]["metrics"][key] - controls[name, 9]["metrics"][key]) for name in spatial_names
            )
        else:
            metric = key.removeprefix("common_")
            samples = [
                next(
                    p
                    for p in controls["joint250", q]["comparison"]["common_spatial_measures"]["tangent_peaks"]
                    if p["metric"] == metric
                )["reference"]["stress_pa"]
                for q in (9, 25)
            ]
            metric_sampling = abs(samples[0] / samples[1] - 1)
        envelopes.append(
            {
                "metric": key,
                "spatial_change": spatial,
                "load_quadrature_change": quadrature,
                "metric_sampling_change": metric_sampling,
                "observed_reference_change_envelope": (1 + spatial) * (1 + quadrature) * (1 + metric_sampling) - 1,
            }
        )
    passed = all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "checks": checks,
            "prescribed_reference_change_targets_passed": passed,
            "targets": targets,
            "controls": [{"name": name, "quadrature": q, **value} for (name, q), value in controls.items()],
            "adapter": adapter,
            "sampling": sampling,
            "envelopes": envelopes,
            "receipts": list(receipts.values()),
            "certified_continuum_bound": False,
            "whole_goal_complete": False,
            "scope": "Independent C2 prescribed-pressure benchmark only:0.1N per pole, fixed2mm radius and free "
            "linear3D elasticity. The final125um/three-layer reference must pass separate wall/surface/joint "
            "changes, load and common metric sampling, and same-mesh factor equivalence. These measured "
            "envelopes are conditional observed resolution changes, not proved continuum bounds. The earlier "
            "two-layer surface comparison remains above2%; it is retained in the trend and is not rounded "
            "into a pass. This control does not qualify physical-pad contact, friction or method field errors.",
        },
    )
    assert passed, "Prescribed reference gates failed; exact failing checks retained"


if __name__ == "__main__":
    main()
