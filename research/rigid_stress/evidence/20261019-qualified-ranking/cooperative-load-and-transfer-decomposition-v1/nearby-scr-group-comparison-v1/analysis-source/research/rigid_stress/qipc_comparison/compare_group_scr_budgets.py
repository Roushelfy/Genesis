"""Audit aggregate-budget fields, actual source changes and separate numerical sensitivities."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_distributed_pad_transfer import field_norm


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.data_root
    receipts, arrays = {}, {}

    def read(path):
        receipts[str(path)] = file_id(path)
        return json.loads(path.read_text())

    def load(receipt):
        path = receipt["path"]
        if path not in arrays:
            assert file_id(path) == receipt
            receipts[path] = receipt
            with np.load(path) as saved:
                arrays[path] = {key: saved[key] for key in saved.files}
        return arrays[path]

    def study(campaign, group):
        directory = root / campaign / f"group{group}/study"
        result, start = read(directory / "result.json"), read(directory / "start.json")
        assert result["passed"] and len(result["rows"]) == 1
        assert start["budget_scope_name"] == "group" and start["budgets"] == [0, 0.001, 0.01, 0.1]
        row = result["rows"][0]
        assert row["budgets"][0]["budget"] == 0 and not row["budgets"][0]["physically_evaluable"]
        assert all(not b["engineering_stress_accepted"] for b in row["budgets"])
        assert all(result["path_controls"].values())
        return row, start

    def difference(first, second):
        a, b = load(first["field"]), load(second["field"])
        np.testing.assert_array_equal(a["vertices"], b["vertices"])
        np.testing.assert_array_equal(a["tetrahedra"], b["tetrahedra"])
        vertices, tets = b["vertices"], b["tetrahedra"]
        volumes = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
        output = {}
        baseline = field_norm(b["baseline_stress_pa"], volumes)
        for key in ("stress_pa", "delta_stress_pa", "baseline_stress_pa"):
            change = field_norm(a[key] - b[key], volumes)
            output[key] = {"rms_change_pa": float(change / np.sqrt(volumes.sum())),
                           "relative_to_finer_baseline": float(change / baseline),
                           "relative_to_finer_own_field": float(change / field_norm(b[key], volumes))}
        return output

    original = [study("nearby-scr-port-budget-v1", i) for i in range(8)]
    scaled = [study("nearby-scr-port-budget-v2", i) for i in range(8)]
    precision = [study("nearby-scr-fit-accuracy-v1", i) for i in range(4)]
    rows, sensitivity, quality = [], [], []
    for environment_index in (0, 1):
        low, high = precision[2 * environment_index], precision[2 * environment_index + 1]
        assert low[1]["state"] == high[1]["state"]
        assert [low[1]["positive_budget_solver_target"], high[1]["positive_budget_solver_target"]] == [1e-9, 1e-10]
        baseline = load(low[0]["baseline"])
        state = load(low[1]["state"])
        source_positions, offsets = baseline["source_positions_m"], baseline["source_offsets"]
        for candidate, tighter in zip(low[0]["budgets"], high[0]["budgets"]):
            assert candidate["budget"] == tighter["budget"]
            if not candidate["physically_evaluable"]:
                assert not tighter["physically_evaluable"]
                continue
            assert tighter["physically_evaluable"], "Numerical refinement remains unresolved"
            assert all(s["status"] in ("Solved", "AnalyticProjection") for s in candidate["sources"] + tighter["sources"])
            comparison = difference(candidate, tighter)
            assert comparison["stress_pa"]["relative_to_finer_own_field"] <= 1e-6
            sensitivity.append({"environment": low[0]["environment"], "variant": candidate["variant"],
                                "kind": "solver_1e-9_to_1e-10", "fields": comparison})
            field = load(candidate["field"])
            point_delta = field["source_point_forces_n"] - baseline["source_point_forces_n"]
            scale = sum(source["force_scale_n"] for source in low[0]["sources"])
            nodal_delta = field["forces_n"] - field["baseline_forces_n"]
            np.testing.assert_allclose(nodal_delta.sum(axis=0), point_delta.sum(axis=0), rtol=0, atol=1e-11 * scale)
            # Force arrays use P2 coordinates, whereas the saved mesh vertices
            # are P1. Global moment is checked directly from source positions
            # against the independently recorded conservative-scatter result.
            np.testing.assert_allclose(np.cross(source_positions, point_delta).sum(axis=0),
                                       candidate["total_origin_moment_change_nm"], rtol=0, atol=1e-11 * scale * 0.03)
            for i, (source, reported) in enumerate(zip(low[0]["sources"], candidate["individual_source_corrections"])):
                selected = slice(offsets[i], offsets[i + 1])
                force_change = point_delta[selected].sum(axis=0)
                anchor = state["contacts.center"][source["source_index"], low[0]["environment"]]
                moment_change = np.cross(source_positions[selected] - anchor, point_delta[selected]).sum(axis=0)
                np.testing.assert_allclose(force_change, reported["force_change_n"], rtol=0, atol=1e-11 * scale)
                np.testing.assert_allclose(moment_change, reported["anchor_moment_change_nm"], rtol=0, atol=1e-11 * scale * 0.03)
                np.testing.assert_allclose(np.linalg.norm(force_change) / source["force_scale_n"],
                                           reported["relative_force_change"], rtol=1e-10, atol=1e-12)
                quality.append({"environment": low[0]["environment"], "variant": candidate["variant"],
                                "source": source["source_index"], **reported})
            rows.append({"environment": low[0]["environment"], "variant": candidate["variant"],
                "budget": candidate["budget"], "load_budget_passed": candidate["accepted"],
                "engineering_stress_accepted": False,
                "full_stress_relative_correction": candidate["complete_stress_relative_change"],
                "full_stress_rms_correction_pa": candidate["complete_stress_rms_change_pa"],
                "max_source_force_relative_correction": max(s["relative_force_change"] for s in candidate["individual_source_corrections"]),
                "max_source_moment_scaled_correction": max(s["scaled_anchor_moment_change"] for s in candidate["individual_source_corrections"]),
                "qp_or_projection_seconds": sum(s["seconds"] for s in candidate["sources"]),
                "two_elastic_solves_seconds": candidate["two_elastic_solves_seconds"]})
    for group in range(8):
        old, new = original[group][0], scaled[group][0]
        assert original[group][1]["state"] == scaled[group][1]["state"]
        for first, second in zip(old["budgets"], new["budgets"]):
            if first["physically_evaluable"] and second["physically_evaluable"]:
                comparison = difference(first, second)
                assert comparison["stress_pa"]["relative_to_finer_own_field"] <= 1e-6
                sensitivity.append({"environment": old["environment"], "variant": first["variant"],
                                    "kind": "equivalent_cone_row_scaling", "group": group, "fields": comparison})
    for environment_index in (0, 1):
        for kind, pairs in (("angular_225_to_625", ((0, 1), (2, 3))),
                            ("ports_8_to_16", ((0, 2), (1, 3)))):
            for a, b in pairs:
                coarse, fine = original[4 * environment_index + a][0], original[4 * environment_index + b][0]
                for first, second in zip(coarse["budgets"], fine["budgets"]):
                    if first["physically_evaluable"] and second["physically_evaluable"]:
                        sensitivity.append({"environment": coarse["environment"], "variant": first["variant"],
                            "kind": kind, "groups": [4 * environment_index + a, 4 * environment_index + b],
                            "fields": difference(first, second)})
    with (args.output / "budgets.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "result.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "passed": True, "rows": rows, "individual_sources": quality, "sensitivities": sensitivity,
        "receipts": list(receipts.values()), "whole_goal_complete": False,
        "scope": "Actual feasible aggregate-budget loads on original supports with retained source priors. Individual changes and signed global point-force/moment corrections are independently remeasured. Equivalent cone conditioning and separate numerical stopping variants preserve all independent checks. Port-path and angular changes are distinct observed sensitivities; missing unsuccessful optimizations never count as accepted. Stress is the explicit linear contact-only correction including equilibrium projection, not a new physical reference or a production acceptance default."})


if __name__ == "__main__":
    main()
