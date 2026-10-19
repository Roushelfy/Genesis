"""Audit matched continuous/static/replay Shell controls with direct signed tensors."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


FIELD_KEYS = ("tangent", "membrane", "bending", "outer_tangent", "inner_tangent")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, rows, splits, checks = {}, [], [], {}

    def record(path, expected=None):
        ident = file_id(path)
        if expected is not None:
            assert ident == expected
        receipts[str(path)] = ident
        return ident

    def read(relative):
        path = args.data_root / relative
        record(path)
        return json.loads(path.read_text())

    admitted = read("pad-absolute-errors-cpu-v2/group0/audit/result.json")
    assert admitted["passed"]
    analytic = read("static-inertia-relief-controls-v1/group0/controls/result.json")
    assert analytic["passed"] and len(analytic["cases"]) == 9
    geometry = read("column-external-volume-controls-v1/group0/controls/result.json")
    assert geometry["passed"] and len(geometry["cases"]) == 8
    for level, load in enumerate((0.1, 0.3, 1.0)):
        for mode in ("replay", "solved-pad"):
            d = read(f"physical-pad-shell250-static-controls-v2/group{level}/{mode}/result.json")
            assert (
                d["load_n"] == load
                and d["static_equations_admitted"]
                and d["inertia_relief_relative_correction"] <= 1e-8
            )
            if mode == "solved-pad":
                assert d["pad_contact"]["static_pad_equations_admitted"]
            for key in ("field", "source_field"):
                record(Path(d[key]["path"]), d[key])
            journal = args.data_root / f"physical-pad-shell250-static-controls-v2/group{level}/{mode}/steps.jsonl"
            record(journal)
            steps = [json.loads(line) for line in journal.read_text().splitlines()]
            assert len(steps) == d["iterations"] and all(
                not any(r["solver_failure_flags"]) and r["inertia_relief_relative_correction"] <= 1e-8 for r in steps
            )
        families = {}
        for family, campaign, group in (
            ("continuous-static-replay", "physical-pad-shell250-control-fields-v2", level),
            ("continuous-replay-solid", "physical-pad-shell250-control-fields-v1", level + 3),
        ):
            values = []
            for q in (9, 25):
                directory = args.data_root / f"{campaign}/group{group}/{family}-{level}-q{q}"
                result = read(str(directory.relative_to(args.data_root) / "result.json"))
                assert result["physical_volume_weighting"] and result["depth_quadrature_points"] == 4
                case = result["cases"][0]
                assert case["three_field_decomposition"]["passed_tensor_identity"]
                for raw in case["raw"]:
                    for key in ("mesh", "field"):
                        record(Path(raw[key]["path"]), raw[key])
                values.append((directory, case))
            families[family] = values
            coarse, primary = (v[1] for v in values)
            for index, (a, b) in enumerate(zip(coarse["adjacent_comparisons"], primary["adjacent_comparisons"])):
                for key in FIELD_KEYS:
                    delta = abs(a[key + "_relative_l2"] - b[key + "_relative_l2"])
                    checks[f"q9-q25/{load}/{family}/{index}/{key}"] = delta <= 0.005
            for index, r in enumerate(primary["adjacent_comparisons"]):
                rows.append(
                    {
                        "load_n": load,
                        "family": family,
                        "adjacent_pair": index,
                        **{
                            key: r[key]
                            for key in r
                            if key.endswith(
                                ("relative_l2", "rms_error_pa", "rms_error_n_per_m", "rms_error_n", "relative_change")
                            )
                        },
                    }
                )
            splits.append({"load_n": load, "family": family, "decomposition": primary["three_field_decomposition"]})
        first_dir, first = families["continuous-static-replay"][1]
        second_dir, second = families["continuous-replay-solid"][1]
        own = first["adjacent_comparisons"][0]
        for key in FIELD_KEYS:
            checks[f"continuous-vs-static/{load}/{key}"] = own[key + "_relative_l2"] <= 0.005
        for key in ("raw_mises_relative_change", "raw_tension_relative_change"):
            checks[f"continuous-vs-static/{load}/{key}"] = own[key] <= 0.01
        for key in FIELD_KEYS:
            a = first["three_field_decomposition"]["metrics"][key]["total_difference_rms"]
            b = second["three_field_decomposition"]["metrics"][key]["first_difference_rms"]
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=0)
            checks[f"shared-continuous-replay/{load}/{key}"] = True
        # Independently reload complete resultant arrays. The first triple is
        # C,S,R; the second is C,R,F. No scalar error subtraction is used.
        arrays = []
        for directory, index in ((first_dir, 0), (first_dir, 1), (second_dir, 0), (second_dir, 1)):
            path = directory / f"columns-{load:g}-{index}.npz"
            record(path)
            with np.load(path) as saved:
                arrays.append({k: saved[k] for k in ("area_m2", "membrane_n_per_m", "bending_n")})
        area = arrays[0]["area_m2"]
        for a in arrays[1:]:
            np.testing.assert_allclose(a["area_m2"], area, rtol=1e-15, atol=0)
        for metric, key in (("membrane", "membrane_n_per_m"), ("bending", "bending_n")):
            C, S = arrays[0][key]
            _, R = arrays[1][key]
            C2, R2 = arrays[2][key]
            _, F = arrays[3][key]
            for left, right in ((C, C2), (R, R2)):
                scale = max(np.linalg.norm(left), np.linalg.norm(right))
                assert np.linalg.norm(left - right) <= 1024 * np.finfo(float).eps * scale
            for label, values, case in (
                ("continuous-static-replay", (C, S, R), first),
                ("continuous-replay-solid", (C2, R2, F), second),
            ):
                a, b, c = values[0] - values[1], values[1] - values[2], values[0] - values[2]
                norms = [np.sqrt(np.einsum("qij,qij,q->", v, v, area) / area.sum()) for v in (a, b, c)]
                row = case["three_field_decomposition"]["metrics"][metric]
                np.testing.assert_allclose(
                    norms,
                    [row[k] for k in ("first_difference_rms", "second_difference_rms", "total_difference_rms")],
                    rtol=1e-9,
                    atol=0,
                )
                cross = 2 * np.einsum("qij,qij,q->", a, b, area) / area.sum()
                assert abs(cross - row["twice_cross_mean_square"]) <= 1e-9 * 2 * norms[0] * norms[1]
                checks[f"full-resultant-array/{load}/{label}/{metric}"] = True
    for group, mode in ((3, "replay"), (4, "solved-pad")):
        for q in (9, 25):
            d = read(f"physical-pad-shell250-control-fields-v2/group{group}/{mode}-old-new-q{q}/result.json")
            r = d["cases"][0]["adjacent_comparisons"][0]
            for key in FIELD_KEYS:
                checks[f"old-new/{mode}/{q}/{key}"] = r[key + "_relative_l2"] <= 1e-8
            for key in ("raw_mises_relative_change", "raw_tension_relative_change"):
                checks[f"old-new/{mode}/{q}/{key}"] = r[key] <= 1e-8
    passed = all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "rows": rows,
            "signed_decompositions": splits,
            "receipts": list(receipts.values()),
            "whole_goal_complete": False,
            "scope": "Matched250um physical-pad Shell controls at0.1/0.3/1N, q9/q25, exact declared P2 physical volume carrier and independent complete resultant-array cross checks. The same-traction solid is the250um two-wall centroid-pressure source, not a certified fine continuum reference. C-S measures static/history/map differences; S-R pressure-feedback differences; R-F combines Shell elasticity/discretization/transfer. Direct tensor identities hold with measured cross terms; scalar RMS values are not additive. Existing qualified125um normal ranks remain separate.",
        },
    )
    if not passed:
        raise SystemExit("Shell pad control gates failed; detailed results retained.")


if __name__ == "__main__":
    main()
