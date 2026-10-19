"""Bind qualified reference-case errors to matched, separately admitted workflow costs.

Reference-case errors remain measurements at their recorded equilibria, not
new error measurements at every timed endpoint. Production rates are separate.
"""

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from .artifacts import file_id, runtime_identity, write_json
from .pad_cost_matrix import gpu_configuration

OBSERVATION_OPTIONS = {
    "output",
    "loads",
    "envs",
    "env_load_scales",
    "primary_cost",
    "intrusive_memory_profile",
    "intrusive_phase_profile",
    "cost_warm_levels",
    "stress_stride",
    "stress_output",
    "paired_output_check",
    "capture_first_linear",
    "capture_linear_step",
}


def configuration_checks(baseline, timed, baseline_source, timed_source):
    """Keep physical/solver inputs strict; report output/cadence/history separately."""
    a, b = baseline["args"], timed["args"]
    mass = timed.get("mass_kg", timed.get("mass_kg_per_env"))
    checks = {
        "same_physical_and_solver_parameters": {k: v for k, v in a.items() if k not in OBSERVATION_OPTIONS}
        == {k: v for k, v in b.items() if k not in OBSERVATION_OPTIONS},
        "same_recovery_mesh": baseline["mesh"] == timed["mesh"],
        "same_actual_mass_per_environment": isinstance(mass, (int, float))
        and math.isfinite(mass)
        and math.isclose(baseline["mass_kg"], mass, rel_tol=1e-12, abs_tol=0),
        "all_genesis_runtime_files_identical": baseline_source == timed_source,
    }
    if "assets" in baseline:
        checks["same_native_material_and_geometry_assets"] = baseline["assets"] == timed.get("assets")
    else:
        geometry = timed.get("physical_pad_geometry", timed.get("pad_geometry"))
        checks["same_shell_physical_pad_geometry"] = baseline["physical_pad_geometry"] == geometry
    return checks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--case", choices=("normal", "friction"), required=True)
    cli.add_argument("--ranking", type=Path, help="Required qualified final ranking for the friction case")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, source_cache = {}, {}, {}
    timings, associated, failures, sources = [], [], [], []
    error = None

    def identify(path):
        key = str(path)
        if key not in receipts:
            receipts[key] = file_id(path)
        return receipts[key]

    def verify(expected):
        assert identify(Path(expected["path"])) == expected, "Changed admitted input: " + expected["path"]

    def read(path):
        identify(path)
        return json.loads(Path(path).read_text())

    def check(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    def source(campaign, method):
        path = args.data_root / campaign / ("extra-source0.json" if method == "shell" else "source.json")
        if str(path) not in source_cache:
            saved = read(path)
            inventory = [row for row in saved["files"] if row["path"].startswith("genesis/")]
            assert len(inventory) > 700
            for row in inventory:
                actual = identify(Path(saved["source"]) / row["path"])
                assert all(actual[key] == row[key] for key in ("bytes", "sha256")), (
                    "Runtime source changed: " + row["path"]
                )
            encoded = json.dumps(
                sorted((row["path"], row["bytes"], row["sha256"]) for row in inventory), separators=(",", ":")
            )
            signature = hashlib.sha256(encoded.encode()).hexdigest()
            source_cache[str(path)] = signature
            sources.append(
                {
                    "manifest": identify(path),
                    "revision": saved["revision"],
                    "genesis_files_rehashed": len(inventory),
                    "runtime_signature": signature,
                }
            )
        return source_cache[str(path)]

    def precision(run):
        campaign, group, name = run.relative_to(args.data_root).parts
        root = args.data_root / campaign
        tasks = read(root / "tasks.json")[int(group.removeprefix("group"))]
        matching = [task for task in tasks if task["name"] == name]
        assert len(matching) == 1
        relative = matching[0]["module"].replace(".", "/") + ".py"
        inventory = {row["path"]: row for row in read(root / "analysis-files.json")}
        path = root / "analysis-source" / relative
        actual = identify(path)
        assert all(actual[key] == inventory[relative][key] for key in ("bytes", "sha256"))
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "gs"
        ]
        assert len(calls) == 1
        values = [
            item.value.value
            for item in calls[0].keywords
            if item.arg == "precision" and isinstance(item.value, ast.Constant)
        ]
        assert values == ["64"], "Captured worker must explicitly select64-bit arithmetic"
        return values[0]

    try:
        memory = {}
        if args.case == "normal":
            errors = read(args.data_root / "pad-absolute-errors-cpu-v2/group0/audit/result.json")
            normal = read(args.data_root / "pad-normal-qualification-v2/group0/qualification/result.json")
            peaks = read(args.data_root / "pad-common-peak-qualification-v2/group0/qualification/result.json")
            check(
                "normal_reference_and_dimensional_errors_admitted",
                errors["passed"]
                and len(errors["rows"]) == 81
                and normal["normal_reference_change_targets_passed"]
                and peaks["common_peak_change_targets_passed"]
                and all(normal["checks"].values())
                and all(peaks["checks"].values()),
            )
            for result in (errors, normal, peaks):
                for receipt in result["receipts"]:
                    verify(receipt)
            error_rows = [row for row in errors["rows"] if row["load_n"] == 0.1]
            field_intervals = {(row["method"], row["metric"]): row for row in normal["errors"] if row["load_n"] == 0.1}
            peak_intervals = {
                (row["method"], "common_" + row["metric"]): row for row in peaks["peaks"] if row["load_n"] == 0.1
            }
            for row in error_rows:
                key = row["metric"] + ("_relative_change" if row["metric"].startswith("raw_") else "_relative_l2")
                interval = (
                    peak_intervals[row["method"], row["metric"]]
                    if row["metric"].startswith("common_")
                    else field_intervals[row["method"], key]
                )
                row.update({key: interval[key] for key in ("conditional_lower", "conditional_upper")})
            details = {row["method"]: row for row in errors["details"] if row["load_n"] == 0.1}
            matrix = read(args.data_root / "pad-cost-matrix-full-v2/group0/matrix/result.json")
            check(
                "complete_normal_primary_matrix_admitted",
                matrix["passed"]
                and matrix["full_B1_B4_matrix"]
                and len(matrix["timings"]) == 24
                and len(matrix["memory"]) == 12,
            )
            for receipt in (*matrix["receipts"], *matrix["verified_audit_inputs"]):
                verify(receipt)
            cost_rows = matrix["timings"]
            memory = {(row["method"], row["envs"], row["output_mode"]): row for row in matrix["memory"]}
            baseline_runs = {
                method: Path(details[method]["raw"][0]["field"]["path"]).parent.parent
                for method in ("dpr", "scr", "shell")
            }
            expected_count, error_time = 24, 0.2
        else:
            check("explicit_final_friction_ranking_provided", args.ranking is not None)
            ranking = read(args.ranking)
            check(
                "qualified_friction_error_ranking_admitted",
                ranking.get("passed", False)
                and ranking.get("errors_admitted", False)
                and ranking.get("reference_qualification_accepted", False)
                and all(ranking["checks"].values())
                and len(ranking["rows"]) == 54,
            )
            for receipt in ranking["receipts"]:
                verify(receipt)
            error_rows = [row for row in ranking["rows"] if row["level"] == 1]
            details = {row["method"]: row for row in ranking["details"] if row["level"] == 1}
            baseline_runs = {method: Path(candidate["run"]) for method, candidate in ranking["candidates"].items()}
            cost_rows = []
            for group, method in enumerate(("dpr", "scr", "shell")):
                run = args.data_root / f"pad-friction-keyframe-cost-v1/group{group}/{method}"
                cost_rows.append(
                    {
                        "method": method,
                        "run": str(run),
                        "admission_audit": str(run.parent / "primary-audit/result.json"),
                    }
                )
            run = args.data_root / "pad-primary-friction-cost-v1/group2/shell"
            cost_rows.append(
                {"method": "shell", "run": str(run), "admission_audit": str(run.parent / "primary-audit/result.json")}
            )
            for group, method in enumerate(("dpr", "scr")):
                run = args.data_root / f"pad-primary-friction-cost-v1/group{group}/{method}"
                failed = read(run / "result.json")
                check(method + "/failed_every_step_mode_preserved", not failed["completed"])
                failures.append(
                    {
                        "method": method,
                        "run": str(run),
                        "completed": False,
                        "written_steps": failed["steps"],
                        "admitted_timing": None,
                        "assigned_error_or_rank": None,
                    }
                )
            expected_count, error_time = 4, 0.4
        check("all27_reference_case_metric_rows", len(error_rows) == 27 and len(details) == 3)
        gpus = set()
        negative_count = 0
        for index, row in enumerate(cost_rows):
            run, method = Path(row["run"]), row["method"]
            audit = read(Path(row["admission_audit"]))
            check(f"{index}/own_primary_admission", audit["passed"])
            for receipt in audit["receipts"]:
                verify(receipt)
            case = next(item for item in audit["cases"] if item["run"] == str(run))
            check(f"{index}/own_case_all_checks", all(case["checks"].values()))
            for name in ("start.json", "result.json", "cost.json", "steps.jsonl"):
                check(f"{index}/audit_binds_" + name, identify(run / name) in audit["receipts"])
            start, result, cost = (read(run / name) for name in ("start.json", "result.json", "cost.json"))
            check(
                f"{index}/whole_primary_cost_completed",
                result["completed"]
                and cost == case["cost"]
                and cost["completed"]
                and cost["primary_timing"]
                and not cost.get("intrusive_memory_profile", False),
            )
            baseline_run = baseline_runs[method]
            baseline = read(baseline_run / "start.json")
            baseline_campaign = baseline_run.relative_to(args.data_root).parts[0]
            campaign = run.relative_to(args.data_root).parts[0]
            baseline_signature, timed_signature = source(baseline_campaign, method), source(campaign, method)
            check(f"{index}/captured_worker_precision64", precision(baseline_run) == precision(run) == "64")
            agreement = configuration_checks(baseline, start, baseline_signature, timed_signature)
            for name, passed in agreement.items():
                check(f"{index}/" + name, passed)
            check(
                f"{index}/ranked_mesh_and_field",
                baseline["mesh"] == details[method]["raw"][0]["mesh"]
                and details[method]["raw"][0]["field"] in audit["receipts"],
            )
            for receipt in (baseline["mesh"], start["mesh"], details[method]["raw"][0]["field"]):
                verify(receipt)
            if method != "shell":
                verify(baseline["assets"])
                assets = read(Path(baseline["assets"]["path"]))
                for key in ("source", "elastic_mesh", "collision_mesh", "urdf"):
                    verify(assets[key])
            else:
                verify(baseline["physical_pad_geometry"])
            gpus.add(gpu_configuration(start["identity"]))
            config = start["args"]
            samples = [sample for sample in cost["samples"] if not sample["warmup"]]
            check(
                f"{index}/three_complete_warm_intervals",
                len(samples) == cost["measured_repeats"] == 3
                and all(sample["steps"] == 200 and sample["all_stress_observations_valid"] for sample in samples),
            )
            check(
                f"{index}/declared_case_loads",
                config["loads"] == [0.1] * (4 if args.case == "normal" else 5)
                and config["friction"] == (0 if args.case == "normal" else 0.3)
                and config["gravity_x"] == (0 if args.case == "normal" else 1),
            )
            envs = config.get("envs", 1)
            latency = [1000 * sample["wall_seconds"] / sample["steps"] for sample in samples]
            check(f"{index}/recomputed_primary_latency", median(latency) == cost["median_latency_ms"])
            fields = case["fields"]
            conformance = [
                field["same_mesh_tensor_relative_l2"]
                for field in fields
                if field["same_mesh_tensor_relative_l2"] is not None
            ]
            check(
                f"{index}/original_endpoint_conformance",
                all(field["raw_peak_relative_difference"] <= 0.01 for field in fields)
                and all(value <= 0.005 for value in conformance),
            )
            mode, stride = config["stress_output"], config["stress_stride"]
            check(
                f"{index}/full_output_has_complete_tensor_controls", mode != "full" or len(conformance) == len(fields)
            )
            profile = memory.get((method, envs, mode)) if stride == 1 else None
            timing = {
                "case": args.case,
                "method": method,
                "envs": envs,
                "output_mode": mode,
                "stress_stride": stride,
                "dt_s": config["dt"],
                "precision_bits": 64,
                "radius_m": config.get("radius"),
                "source_reduction": config.get("source_reduction"),
                "linear_tolerance": config.get("tolerance"),
                "absolute_velocity_tolerance": config.get("velocity_tolerance"),
                "mesh_sha256": start["mesh"]["sha256"],
                "actual_mass_kg_per_environment": start.get("mass_kg", start.get("mass_kg_per_env")),
                "setup_seconds": start["setup_seconds"],
                "median_batch_step_ms": median(latency),
                "minimum_batch_step_ms": min(latency),
                "maximum_batch_step_ms": max(latency),
                "median_aggregate_env_steps_per_second": median(envs * 1000 / value for value in latency),
                "warm_intervals": 3,
                "independent_process_repeats": False,
                "reference_case_time_s": error_time,
                "timed_start_s": min(sample["level"] for sample in samples) * 0.2,
                "timed_end_s": (max(sample["level"] for sample in samples) + 1) * 0.2,
                "maximum_endpoint_tensor_conformance_l2": max(conformance) if conformance else None,
                "maximum_endpoint_raw_peak_conformance": max(field["raw_peak_relative_difference"] for field in fields),
                "sampled_memory_separate_profile_mib": profile["sampled_gpu_process_peak_mib"] if profile else None,
                "separate_memory_profile_run": profile["run"] if profile else None,
                "memory_maximum_sample_gap_s": profile["maximum_sample_interval_seconds"] if profile else None,
                "memory_sample_count": profile["sample_count"] if profile else None,
                "separate_cpu_max_rss_mib": profile["cpu_process_max_rss_mib"] if profile else None,
                "runtime_source_signature": timed_signature,
                "run": str(run),
                "admission_audit": row["admission_audit"],
                "error_source_frame": details[method]["raw"][0]["field"]["path"],
                "exact_error_measured_at_timed_endpoints": False,
                "cost_scope": "research workflow including mechanics, recovery, host output, mechanical audits and journal I/O",
            }
            timings.append(timing)
            for metric in error_rows:
                if metric["method"] == method and (mode == "full" or metric["metric"] == "raw_mises"):
                    associated.append(
                        {
                            **timing,
                            **{
                                key: metric[key]
                                for key in (
                                    "metric",
                                    "absolute_error",
                                    "unit",
                                    "reference_rms_or_peak",
                                    "relative_error",
                                    "conditional_lower",
                                    "conditional_upper",
                                )
                            },
                        }
                    )
            if index < 3:
                for key, changed in (("radius", 0.003), ("friction", 0.9), ("dt", 0.002)):
                    bad = copy.deepcopy(start)
                    bad["args"][key] = changed
                    check(
                        f"negative/{index}/changed_" + key,
                        not all(configuration_checks(baseline, bad, baseline_signature, timed_signature).values()),
                    )
                    negative_count += 1
                bad = copy.deepcopy(start)
                mass_key = "mass_kg" if "mass_kg" in bad else "mass_kg_per_env"
                bad[mass_key] *= 1.1
                check(
                    f"negative/{index}/changed_actual_mass",
                    not all(configuration_checks(baseline, bad, baseline_signature, timed_signature).values()),
                )
                check(
                    f"negative/{index}/changed_runtime_source",
                    not all(configuration_checks(baseline, start, baseline_signature, "0" * 64).values()),
                )
                negative_count += 2
        check("same_gpu_model_driver_capacity", len(gpus) == 1)
        check("complete_matched_timing_configurations", len(timings) == expected_count)
        check("all_applicable_metric_associations", len(associated) == (120 if args.case == "normal" else 36))
        check("fifteen_changed_configuration_rejections", negative_count == 15)
    except (OSError, KeyError, ValueError, AssertionError, StopIteration) as failure:
        error = str(failure)
        checks["complete_matched_cost_error_contract"] = False
    passed = bool(checks) and all(checks.values())
    if passed:
        for name, rows in (("timings", timings), ("reference-case-errors-and-cost", associated)):
            with (args.output / (name + ".csv")).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "case": args.case,
            "passed": passed,
            "checks": checks,
            "error": error,
            "matched_configurations": timings if passed else [],
            "reference_case_error_cost_rows": associated if passed else [],
            "preserved_every_step_failures": failures,
            "runtime_source_manifests": sources,
            "receipts": list(receipts.values()),
            "whole_goal_complete": False,
            "scope": "Qualified reference-case error measurements and separately own-admitted research-workflow timings share exact geometry, material assets, mass, physical/solver parameters and byte-identical Genesis runtime inventories. Output/cadence/batch and physical history horizons are explicit. Endpoint field/peak conformance is retained separately; it is not a new continuum error bound or an exact error measurement at each timed endpoint. Max-only rows associate only raw Mises magnitude error; no complete-field error is assigned to unexported tensors. Sparse memory is unavailable, and sampled every-step memory comes from separate process profiles. Warm blocks are not independent process repeats. Default-collider production/module throughput never inherits pad errors.",
        },
    )
    if not passed:
        raise SystemExit("Matched cost/error contract failed; no joined rows admitted")


if __name__ == "__main__":
    main()
