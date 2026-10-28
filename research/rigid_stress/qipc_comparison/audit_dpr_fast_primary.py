"""Audit original paired throughput receipts without importing scientific libraries."""

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
from pathlib import Path

from .artifacts import file_id, write_json


def argument(arguments, name, default=None):
    return arguments[arguments.index(name) + 1] if name in arguments else default


def checked_json(identity):
    path = Path(identity["path"])
    if file_id(path) != identity:
        raise ValueError(f"Changed recorded artifact: {path}")
    return json.loads(path.read_text())


def inspect_row(group, task, directory, source, revision):
    definition_path = Path(argument(task["args"], "--task"))
    definition = json.loads(definition_path.read_text())
    arguments = definition["args"]
    scope = argument(arguments, "--scope")
    batch = int(argument(arguments, "--envs"))
    output = argument(arguments, "--output-mode")
    route = "pure_rigid"
    if scope != "policy-rigid":
        route = (
            "shell_planar32"
            if argument(arguments, "--stress-backend") == "shell"
            else {
                "adaptive": "volume_adaptive16",
                "angle": "volume_angle128",
                "edge": "volume_edge16_diagnostic",
            }[argument(arguments, "--pressure-quadrature")]
        )
    row = {
        "block": group,
        "name": task["name"],
        "scope": "policy" if scope == "policy-rigid" else scope,
        "batch": batch,
        "output": output,
        "route": route,
        "complete": False,
        "valid_transitions_per_second": None,
    }
    result_path = directory / f"group{group}" / task["name"] / "result.json"
    if not result_path.exists():
        return row
    result = json.loads(result_path.read_text())
    if not result.get("complete"):
        return row
    returncode_path = result_path.parent.parent / (task["name"] + ".returncode.txt")
    if not returncode_path.is_file():
        return row
    row.update(complete=True, result=file_id(result_path), wrapper_passed=result["passed"])
    row["returncode"] = int(returncode_path.read_text().strip())
    row["returncode_receipt"] = file_id(returncode_path)
    device = result["actual_device"] or {"uuid": "unavailable", "name": "unavailable"}
    row["gpu"] = device
    row["host"] = result["runtime"]["hostname"]
    inventory = list(csv.DictReader(io.StringIO(result["runtime"]["nvidia_smi_visible_inventory"])))
    matching = [
        {key.strip(): value.strip() for key, value in item.items()}
        for item in inventory
        if item[" uuid"].strip() == device["uuid"]
    ]
    row["driver"] = matching[0]["driver_version"] if len(matching) == 1 else None
    checks = {
        "task_exact": result["task"] == file_id(definition_path) and result["task_definition"] == definition,
        "stdlib_before_driver": result["checks"]["stdlib_before_driver"],
        "device_available": result["checks"]["actual_cuda_device_available"] and row["driver"] is not None,
        "declared_final_source": argument(arguments, "--source-revision") == revision,
        "captured_worker_exact": file_id(Path(result["worker"]["path"])) == result["worker"],
        "terminal_returncode": row["returncode"] == (0 if result["passed"] else 1),
    }
    row["checks"] = checks
    if result["benchmark"] is None:
        failure = result["failure_snapshot"]
        if failure is not None and failure.get("available"):
            payload = failure["payload"]
            if file_id(Path(payload["path"])) != payload:
                raise ValueError("Changed captured failed input.")
        error = result["driver_error"]
        row["failure"] = error
        row["failure_snapshot"] = failure
        failed_options = (
            [entry["options"] for entry in failure["entries"]] if failure and failure.get("available") else []
        )
        checks["actual_failed_route"] = bool(failed_options) and all(
            options["backend"] == "volume"
            and options["pressure_quadrature"] == "edge"
            and options["quadrature"] == 16
            and options["contact_model"] == "pressure"
            and options["load_transfer"] == "patch"
            and options["radius_model"] == "hertz"
            and options["pressure_profile"] == "hertz"
            and options["output_mode"] == output
            and options["box_edge_policy"] == "planar"
            for options in failed_options
        )
        checks["actual_failed_engine"] = error is not None and all(
            str(source / relative) in error["traceback"]
            for relative in ("examples/speed_benchmark/rigid_stress.py", "genesis/engine/solvers/rigid/rigid_solver.py")
        )
        row["expected_retained_negative"] = bool(
            route == "volume_edge16_diagnostic"
            and error is not None
            and error["type"] == "GenesisException"
            and "status=3" in error["message"]
            and failure is not None
            and failure.get("available")
            and result["failure_capture_error"] is None
        )
        row["execution_qualified"] = all(checks.values()) and row["expected_retained_negative"]
        return row
    benchmark = checked_json(result["benchmark"])
    row["benchmark"] = result["benchmark"]
    checks["source_files_exact"] = all(
        file_id(Path(name) if Path(name).is_absolute() else source / name)["sha256"] == digest
        for name, digest in benchmark["source_sha256"].items()
    )
    checks["source_revision"] = benchmark["source_revision"] == revision
    checks["scope_batch"] = benchmark["scope"] == scope and benchmark["envs"] == batch
    checks["same_actual_device"] = benchmark["gpu_uuid"] == device["uuid"]
    checks["matched_configuration"] = (
        benchmark["precision"] == "64"
        and benchmark["inverse_precision"] == "64"
        and benchmark["history"] == 0
        and benchmark["dt"] == 0.01
        and benchmark["substeps"] == 1
        and benchmark["seed"] == 510000
        and benchmark["egg_geometry"] == "mesh64"
        and not benchmark["cuda_graphs_enabled"]
    )
    checks["one_original_repeat"] = len(benchmark["repeats"]) == 1
    repeat = benchmark["repeats"][0]
    row["repeat"] = repeat
    attempts = repeat["steps"] * batch
    invalid = repeat["invalid_environment_steps"]
    seconds = repeat["seconds"]
    valid_rate = (attempts - invalid) / seconds
    checks["rate_accounting"] = (
        seconds >= 3.0
        and repeat["steps"] >= (300 if scope == "recovery" else 730)
        and math.isclose(repeat["transitions_per_second"], valid_rate, rel_tol=1e-12)
        and math.isclose(repeat["attempted_transitions_per_second"], attempts / seconds, rel_tol=1e-12)
    )
    checks["all_steps_valid"] = invalid == 0
    row["setup_seconds"] = benchmark["setup_seconds"]
    row["actual_options"] = benchmark["actual_stress_options"]
    row["native_buffer_bytes"] = 0 if route == "pure_rigid" else benchmark["native_buffer_bytes"]
    row["policy"] = benchmark["policy"]
    options = benchmark["actual_stress_options"]
    if route != "pure_rigid":
        checks["physical_defaults"] = (
            options["contact_model"] == "pressure"
            and options["load_transfer"] == "patch"
            and options["radius_model"] == "hertz"
            and options["pressure_profile"] == "hertz"
            and options["box_edge_policy"] == "planar"
            and options["output_mode"] == output
        )
        if route == "shell_planar32":
            checks["selected_route"] = (
                options["backend"] == "shell"
                and options["shell_loading"] == "planar"
                and options["loading_subdivisions"] == 32
                and options["shell_thickness"] == 0.0005
                and options["quadrature"] == 10
                and options["scatter_tasks_per_env"] == 128
                and benchmark["planar_integration"] == "radial"
                and benchmark["planar_query_requested"] == "blocks"
                and set(benchmark["effective_planar_query_modes"]) == {"blocks4"}
            )
        else:
            expected = {
                "volume_adaptive16": ("adaptive", 16),
                "volume_angle128": ("angle", 128),
                "volume_edge16_diagnostic": ("edge", 16),
            }[route]
            checks["selected_route"] = (
                options["backend"] == "volume" and (options["pressure_quadrature"], options["quadrature"]) == expected
            )
    if scope == "recovery":
        snapshot = benchmark["module_snapshot"]
        row["module_snapshot"] = snapshot
        row["physical_source_columns"] = benchmark["recovery_snapshot_check"]["physical_source_column_identities"]
        checks["common_raw_snapshot_preserved"] = all(
            snapshot[key]
            for key in ("restore_exact", "raw_inputs_exact_after_preparation", "raw_inputs_exact_after_measurement")
        )
        checks["zero_scene_advancement"] = benchmark["warmup_steps"] == 0 and snapshot["consumer_scene_tick"] == 0
    else:
        metrics = benchmark["trajectory_metrics"]["repeats"][0]
        row["trajectory_metrics"] = metrics
        checks["complete_metrics"] = metrics["totals"]["observed_steps"] == attempts and all(
            sum(metrics["per_environment"][key]) == value for key, value in metrics["totals"].items()
        )
        checks["valid_metrics"] = metrics["totals"]["invalid_stress_steps"] == 0
        checks["actual_policy"] = (
            benchmark["policy"]["layers"] == [26, 128, 128, 7]
            and benchmark["policy"]["seed"] == 99173
            and benchmark["policy"]["dtype"] == "float32"
            and benchmark["policy"]["weights_sha256"]
            == "37952a933b9ceb5bfcd86473abed687ee3fc7bce9409385f1567ef0bd4f7aa64"
            and benchmark["warmup_steps"] == 730
        )
    row["execution_qualified"] = result["passed"] and all(checks.values())
    if row["execution_qualified"]:
        row["valid_transitions_per_second"] = valid_rate
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    manifest = json.loads(args.manifest.read_text())
    if [len(group) for group in manifest] != [34, 34, 34]:
        raise ValueError("Expected all three original34-member paired blocks.")
    rows = [inspect_row(i, task, args.run, args.source, revision) for i, group in enumerate(manifest) for task in group]
    complete = all(row["complete"] for row in rows)
    if not complete and not args.allow_incomplete:
        raise ValueError("The original102-process matrix is still incomplete.")
    finished = [row for row in rows if row["complete"]]
    hardware = {
        str(i): sorted(
            {
                (row["host"], row["gpu"]["uuid"], row["gpu"]["name"], row["driver"])
                for row in finished
                if row["block"] == i
            }
        )
        for i in range(3)
    }
    hardware_passed = (
        all(len(values) == 1 for values in hardware.values())
        and len({(row["gpu"]["name"], row["driver"]) for row in finished}) == 1
    )
    source_groups = {}
    for batch in (4, 8192):
        modules = [row for row in finished if row["batch"] == batch and "module_snapshot" in row]
        source_groups[str(batch)] = {
            "rows": len(modules),
            "raw_snapshots": sorted({row["module_snapshot"]["raw_input_sha256"] for row in modules}),
            "all12_source_columns_equal": bool(modules)
            and all(row["physical_source_columns"] == modules[0]["physical_source_columns"] for row in modules),
        }
    pairs = []
    for scope in ("recovery", "policy"):
        for batch in (4, 8192):
            for output in ("max", "full"):
                for comparator in ("volume_edge16_diagnostic", "volume_angle128", "shell_planar32", "pure_rigid"):
                    if scope == "recovery" and comparator == "pure_rigid":
                        continue
                    ratios = []
                    for block in range(3):
                        selected = [
                            row
                            for row in rows
                            if row["block"] == block
                            and row["scope"] == scope
                            and row["batch"] == batch
                            and row["output"] == output
                            and row["route"] == "volume_adaptive16"
                        ]
                        controls = [
                            row
                            for row in rows
                            if row["block"] == block
                            and row["scope"] == scope
                            and row["batch"] == batch
                            and row["route"] == comparator
                            and (comparator == "pure_rigid" or row["output"] == output)
                        ]
                        if len(selected) == len(controls) == 1:
                            numerator, denominator = (
                                selected[0]["valid_transitions_per_second"],
                                controls[0]["valid_transitions_per_second"],
                            )
                            if numerator is not None and denominator is not None:
                                ratios.append({"block": block, "adaptive_over_comparator": numerator / denominator})
                    values = [item["adaptive_over_comparator"] for item in ratios]
                    pairs.append(
                        {
                            "scope": scope,
                            "batch": batch,
                            "output": output,
                            "comparator": comparator,
                            "ratios": ratios,
                            "complete_three_repeats": len(values) == 3,
                            "median": statistics.median(values) if len(values) == 3 else None,
                            "range": [min(values), max(values)] if len(values) == 3 else None,
                        }
                    )
    passed = (
        complete
        and hardware_passed
        and all(row["execution_qualified"] for row in finished)
        and all(
            len(group["raw_snapshots"]) == 1 and group["all12_source_columns_equal"] for group in source_groups.values()
        )
    )
    output = {
        "schema": "genesis.automatic-dpr-primary-audit.v1",
        "complete": complete,
        "passed": passed,
        "goal_complete": False,
        "source_revision": revision,
        "manifest": file_id(args.manifest),
        "audit_tool": file_id(Path(__file__)),
        "finished_rows": len(finished),
        "rows": rows,
        "hardware": hardware,
        "hardware_passed": hardware_passed,
        "common_module_inputs": source_groups,
        "paired_ratios": pairs,
        "scope": "All original paired repeats and failures retained. Failed warmups have no qualified rate. Module inputs are identical; live policy mechanics can differ through stress feedback. Shell and volume remain distinct physical models. Inference is not training.",
    }
    write_json(args.output, output)
    print(
        json.dumps(
            {"complete": complete, "passed": passed, "finished_rows": len(finished), "output": file_id(args.output)}
        )
    )


if __name__ == "__main__":
    main()
