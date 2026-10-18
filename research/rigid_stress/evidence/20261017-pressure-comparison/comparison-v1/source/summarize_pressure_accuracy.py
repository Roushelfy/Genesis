"""Reduce audited model fields and full-field oracle results to decision metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def bounds(values):
    values = list(values)
    return [float(min(values)), float(max(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    path = args.root / "model-audit-v2/model-comparison.json"
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    audit = json.loads(path.read_text())
    groups = []
    for geometry in ("egg", "ellipsoid", "narrow_gap"):
        for model in ("pressure", "body_source_prior"):
            rows = (
                [r for r in audit["egg"] if r["model"] == model]
                if geometry == "egg"
                else [
                    r
                    for r in audit["controls"]
                    if r["geometry"] == geometry
                    and r["model"] == model
                    and r["classification"] == "numerically_qualified"
                ]
            )
            sources = [s for row in rows for s in row["traction_audit"]["sources"]]
            group = {
                "geometry": geometry,
                "model": model,
                "accepted_fields": len(rows),
                "maximum_local_force_shift_N": max(np.linalg.norm(s["force_shift_N"]) for s in sources),
                "maximum_local_moment_shift_Nm": max(np.linalg.norm(s["moment_shift_Nm"]) for s in sources),
                "maximum_aggregate_force_error_N": max(r["traction_audit"]["total_force_error_N"] for r in rows),
                "maximum_aggregate_moment_error_Nm": max(r["traction_audit"]["total_moment_error_Nm"] for r in rows),
                "loaded_area_range_m2": bounds(s["loaded_area_m2"] for s in sources),
                "maximum_opposing_normal_force_fraction": max(
                    r["traction_audit"]["opposing_material_normal_force_fraction"] for r in rows
                ),
                "cones": {
                    name: {
                        "maximum_raw_violation_Pa": max(
                            r["traction_audit"]["cones"][name]["maximum_raw_violation_Pa"] for r in rows
                        ),
                        "maximum_violating_force_fraction": max(
                            r["traction_audit"]["cones"][name]["violating_force_fraction"] for r in rows
                        ),
                    }
                    for name in ("supplied_pad", "material")
                },
            }
            if geometry == "egg":
                group["by_radius"] = [
                    {
                        "radius_m": radius,
                        "volume_tensor_M_vs_L_range": bounds(
                            r["M_vs_L"]["relative_L2"] for r in rows if r["radius_m"] == radius
                        ),
                        "raw_peak_M_vs_L_fractional_change_range": bounds(
                            r["M_raw_global_peak_Pa"] / r["L_raw_global_peak_Pa"] - 1
                            for r in rows
                            if r["radius_m"] == radius
                        ),
                    }
                    for radius in (0.003, 0.006)
                ]
            else:
                group["weighted_tensor_relative_L2_range"] = bounds(
                    r["weighted_volume_tensor_relative_L2"] for r in rows
                )
                group["raw_peak_fractional_change_range"] = bounds(
                    r["peak_Pa"] / r["reference_peak_Pa"] - 1 for r in rows
                )
                group["hotspot_distance_from_reference_range_m"] = bounds(
                    np.linalg.norm(np.array(r["peak_position_m"]) - r["reference_peak_position_m"]) for r in rows
                )
                if geometry == "narrow_gap":
                    group["cross_gap_force_fraction_range"] = bounds(
                        r["traction_audit"]["cross_gap_force_fraction"] for r in rows
                    )
            groups.append(group)
    hotspots = []
    for pressure in (r for r in audit["egg"] if r["model"] == "pressure"):
        intrinsic = next(
            r
            for r in audit["egg"]
            if r["model"] == "body_source_prior"
            and r["step"] == pressure["step"]
            and r["radius_m"] == pressure["radius_m"]
        )
        hotspots.append(
            {
                "step": pressure["step"],
                "radius_m": pressure["radius_m"],
                "pressure_position_m": pressure["M_raw_global_peak_reference_position_m"],
                "intrinsic_position_m": intrinsic["M_raw_global_peak_reference_position_m"],
                "paired_hotspot_distance_m": float(
                    np.linalg.norm(
                        np.array(pressure["M_raw_global_peak_reference_position_m"])
                        - intrinsic["M_raw_global_peak_reference_position_m"]
                    )
                ),
            }
        )
    oracles = []
    for name in ("tasks32", "tasks128", "tasks256", "fast128"):
        path = args.root / ("pressure-oracle-" + name + "-v1") / "oracle.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        rows = data["oracle_rows"]
        assert data["envs"] == 16 and data["steps"] == 1500 and len(rows) == 120 and data["resets"] == 32
        assert data["output_mode"] == "full" and data["conditions_sha256"] == (
            "1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa"
        )
        oracles.append(
            {
                "case": name,
                "sampled_complete_fields": len(rows),
                "partial_resets": data["resets"],
                "scatter_overflow_calls": data["scatter_overflow_calls"],
                "maximum_complete_gauge_budget_fraction": max(
                    r["full_gauge_residual_N"] / r["residual_budget_N"] for r in rows
                ),
                "maximum_tensor_error_Pa": max(r["tensor_max_absolute_error_Pa"] for r in rows),
                "maximum_global_peak_error_Pa": max(r["global_peak_error_Pa"] for r in rows),
                "maximum_global_peak_relative_error": max(
                    r["global_peak_error_Pa"] / r["global_peak_reference_Pa"]
                    for r in rows
                    if r["global_peak_reference_Pa"] > 0
                ),
                "zero_reference_fields_with_undefined_relative_error": sum(
                    r["global_peak_reference_Pa"] == 0 for r in rows
                ),
                "maximum_absolute_error_on_zero_reference_fields_Pa": max(
                    r["global_peak_error_Pa"] for r in rows if r["global_peak_reference_Pa"] == 0
                ),
                "maximum_individual_force_error_N": max(c["wrench_force_error_N"] for r in rows for c in r["contacts"]),
                "maximum_individual_moment_error_Nm": max(
                    c["wrench_moment_error_Nm"] for r in rows for c in r["contacts"]
                ),
            }
        )
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "groups": groups,
        "paired_egg_hotspots": hotspots,
        "pressure_oracles": oracles,
        "unsupported_controls": [r for r in audit["controls"] if r["classification"] != "numerically_qualified"],
        "input_sha256": inputs,
        "limitations": "Accepted-field ranges use the declared law; pressure excludes four unsupported free-moment cases. Model envelopes with unequal supported cases are not paired accuracy improvements. Hotspot coordinates use the material/reference frame; raw maxima are not smoothed.",
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
