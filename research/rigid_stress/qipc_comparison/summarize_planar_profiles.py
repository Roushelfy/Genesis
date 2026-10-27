"""Audit terminal planar profile receipts and aggregate their recorded trace events.

This standard-library tool retains every kernel and host API name. Duration sums
can overlap and describe intrusive traces rather than primary wall throughput.
"""

import argparse
import hashlib
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path

from .profile_planar_loading import CATEGORIES, _category

RUNTIME_CHECKS = (
    "native_valid",
    "step_valid",
    "finite_output",
    "original_unscaled_residual",
    "positive_loaded_radii",
    "original_force",
    "original_point_moment",
)
RESOURCES = (
    "registers per thread",
    "shared memory",
    "grid",
    "block",
    "est. achieved occupancy %",
    "blocks per SM",
    "warps per SM",
)
RIGID_CONSTRAINT_PREFIXES = (
    "_kernel_solve_graph_",
    "kernel_solve_graph_",
    "func_solve_init_",
    "func_solve_iter_",
    "func_solve_body_",
    "func_solve_decomposed_",
)
RIGID_DYNAMICS_PREFIXES = ("kernel_apply_links_external_wrench", "func_solve_mass_block_", "func_solve_mass_batch_")
STRESS_RESPONSE_PREFIXES = (
    "kernel_apply_inverse_",
    "kernel_solve_c",
    "kernel_solve_cooperative_",
    "kernel_direct_solve_",
)
FUSION_LIMITS = {
    "kernel_rigid_stress_post_": "Joint fused rigid/loading/recovery/output path. Its duration cannot be split into those stages.",
    "kernel_recover_": "Joint shell balance/surface response/stress output recovery. Query, fit and loading retain their separate kernels.",
    "category_absence": "No separate kernel event does not establish zero stage cost inside a fused path.",
    "duration_sums": "Events can overlap across streams and host nesting. Sums and nominal per-step sums are not wall times.",
    "launch_overhead": "Host API durations are reported directly. No launch cost is inferred by subtracting device time from wall time.",
    "other_rigid_control_and_runtime": "The inherited remainder also includes policy tensor/inference kernels in policy traces. It gives no isolated rigid-only or policy-only time.",
    "graph_mode_provenance": "Captured helpers export QD_GRAPH=0 before scientific imports. The policy benchmark records os.environ, combined with frozen Quadrants import-time environment semantics. These profile receipts do not directly read the imported Quadrants graph constant. Independent bounded graph screens retain that direct readback.",
}


def read_bytes(path):
    before = path.stat()
    value = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"Profile receipt changed while reading: {path}")
    return value, {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def read_json(path):
    value, identity = read_bytes(path)
    return json.loads(value), identity


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def arguments(tokens):
    result, index = {}, 0
    while index < len(tokens):
        key = tokens[index].removeprefix("--").replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            result[key] = tokens[index + 1]
            index += 2
        else:
            result[key] = True
            index += 1
    return result


def same_argument(actual, declared):
    if isinstance(actual, bool):
        return actual is declared
    if finite(actual):
        try:
            return actual == float(declared)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(declared)


def event_category(name, *, policy=False):
    if policy:
        if name.startswith(RIGID_CONSTRAINT_PREFIXES):
            return "rigid_constraint_solver"
        if name.startswith(RIGID_DYNAMICS_PREFIXES):
            return "rigid_dynamics_solve_or_external_wrench"
    if "kernel_rigid_stress_post_" in name:
        return "joint_fused_rigid_loading_recovery_output"
    category = _category(name)
    if policy and category == "response" and not name.startswith(STRESS_RESPONSE_PREFIXES):
        return "unattributed_solve_or_apply_kernel"
    return category


def aggregate_trace(trace, steps, issues, *, policy=False):
    """Aggregate complete events without attributing fused durations to hidden stages."""
    groups, totals, annotations = defaultdict(dict), defaultdict(lambda: [0, 0.0]), Counter()
    device_categories = defaultdict(lambda: [0, 0.0])
    cpu_pids, resource_sets = set(), defaultdict(dict)
    for event in trace.get("traceEvents", []):
        if (
            event.get("ph") == "M"
            and event.get("name") == "process_labels"
            and event.get("args", {}).get("labels") == "CPU"
        ):
            cpu_pids.add(event.get("pid"))
        if event.get("ph") != "X":
            continue
        category, name = event.get("cat"), event.get("name")
        if category == "user_annotation":
            annotations[name] += 1
        if category not in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_driver", "cuda_runtime", "cpu_op"):
            continue
        duration = event.get("dur")
        if not isinstance(name, str) or not finite(duration) or duration < 0:
            issues.append(f"Invalid complete {category} duration/name in trace")
            continue
        row = groups[category].setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "sum_duration_us": 0.0,
                "minimum_duration_us": duration,
                "maximum_duration_us": duration,
            },
        )
        row["calls"] += 1
        row["sum_duration_us"] += duration
        row["minimum_duration_us"] = min(row["minimum_duration_us"], duration)
        row["maximum_duration_us"] = max(row["maximum_duration_us"], duration)
        totals[category][0] += 1
        totals[category][1] += duration
        details = event.get("args", {})
        if category == "kernel":
            label = event_category(name, policy=policy)
            row["category"] = label
            device_categories[label][0] += 1
            device_categories[label][1] += duration
            resources = {key: details[key] for key in RESOURCES if key in details}
            resource_sets[name][json.dumps(resources, sort_keys=True)] = resources
        if category in ("gpu_memcpy", "gpu_memset"):
            count = details.get("bytes")
            row.setdefault("reported_bytes_sum", 0)
            row.setdefault("events_with_reported_bytes", 0)
            if isinstance(count, int) and count >= 0:
                row["reported_bytes_sum"] += count
                row["events_with_reported_bytes"] += 1
    output = {}
    for category, rows in groups.items():
        for name, row in rows.items():
            row["mean_event_duration_us"] = row["sum_duration_us"] / row["calls"]
            row["sum_duration_us_per_nominal_step"] = row["sum_duration_us"] / steps
            if category == "kernel":
                row["recorded_resource_variants"] = list(resource_sets[name].values())
        output[category] = sorted(rows.values(), key=lambda row: (-row["sum_duration_us"], row["name"]))
    return {
        "nominal_steps": steps,
        "cpu_process_ids_from_trace_labels": sorted(cpu_pids),
        "user_annotation_counts": dict(annotations),
        "complete_event_totals": {
            name: {
                "calls": values[0],
                "sum_duration_us": values[1],
                "sum_duration_us_per_nominal_step": values[1] / steps,
            }
            for name, values in sorted(totals.items())
        },
        "device_categories": [
            {
                "category": name,
                "calls": values[0],
                "sum_duration_us": values[1],
                "sum_duration_us_per_nominal_step": values[1] / steps,
            }
            for name, values in sorted(device_categories.items())
        ],
        "events_by_category_and_name": output,
        "trace_metadata": {key: value for key, value in trace.items() if key != "traceEvents"},
        "host_graph_launch_events": sum(
            row["calls"]
            for category in ("cuda_driver", "cuda_runtime")
            for row in output.get(category, [])
            if "graphlaunch" in row["name"].lower()
        ),
    }


def audit_task(campaign, group, task, source, analysis, revision, weights):
    row = {"group": group, "name": task["name"], "declared_task": task, "issues": [], "passed": False}
    issues = row["issues"]

    def require(condition, message):
        if not condition:
            issues.append(message)

    directory = campaign / f"group{group}"
    marker = directory / (task["name"] + ".returncode.txt")
    if not marker.is_file():
        issues.append("Missing terminal task returncode. Output tree remains unread.")
        return row
    raw, row["returncode_identity"] = read_bytes(marker)
    try:
        code = int(raw.strip())
    except ValueError:
        issues.append("Unparseable task returncode")
        return row
    row["returncode"] = code
    require(code == 0, "Task returncode is nonzero")
    options = arguments(task["args"])
    module = task["module"] == "research.rigid_stress.qipc_comparison.profile_planar_loading"
    row["kind"] = "module" if module else "policy"
    result_path = directory / task["name"]
    if module:
        result_path /= "result.json"
    result, row["result_identity"] = read_json(result_path)
    trace_path = result_path.parent / ("trace.json" if module else result.get("trace", "MISSING_TRACE"))
    trace, row["trace_identity"] = read_json(trace_path)
    workload, row["workload_script_identity"] = read_bytes(campaign / f"workload-{group}.sh")
    graph_export = b"export QD_GRAPH=0" in workload
    row["arguments"] = result.get("arguments")
    require(
        isinstance(row["arguments"], dict)
        and all(same_argument(row["arguments"].get(key), value) for key, value in options.items()),
        "Recorded arguments differ from frozen task arguments",
    )
    require(row["arguments"].get("output") == str(directory / task["name"]), "Recorded output path differs from task")
    row["declared_command"] = shlex.join(
        ["python", "-m", task["module"], *task["args"], "--output", str(directory / task["name"])]
    )
    row["source_revision"] = source.get("revision")
    row["source_path"] = source["source"]
    captured_commands = [
        line
        for line in workload.decode().splitlines()
        if line.startswith("if ")
        and f" -m {task['module']} " in line
        and f"--output {directory / task['name']} " in line
    ]
    row["captured_shell_invocations"] = captured_commands
    require(len(captured_commands) == 1, "No unique captured task shell invocation")
    row["graph_control"] = {
        "captured_helper_export_QD_GRAPH0": graph_export,
        "benchmark_environment_graph_flag": result.get("cuda_graphs_enabled"),
        "direct_imported_constant_readback": None,
        "scope": "Pre-import helper export plus frozen Quadrants import-time environment semantics. Policy flag comes from os.environ(QD_GRAPH), not an imported-constant readback.",
    }
    require(graph_export, "Captured helper does not explicitly disable CUDA graphs")
    steps = int(options.get("steps", 0)) if module else int(options.get("trace_steps", 0))
    require(steps == 10, "Expected exactly10 nominal trace steps")
    if steps <= 0:
        return row
    stats = aggregate_trace(trace, steps, issues, policy=not module)
    row["trace_summary"] = stats
    require(stats["host_graph_launch_events"] == 0, "Trace contains graph-launch host API events")
    require(len(stats["cpu_process_ids_from_trace_labels"]) == 1, "Trace has no unique CPU process PID")
    require(stats["complete_event_totals"].get("kernel", {}).get("calls", 0) > 0, "No recorded CUDA kernel calls")
    row["envs"] = int(options["envs"])
    row["output_mode"] = options["output_mode"]
    row["variant"] = (
        "planar32"
        if options.get("shell_loading") == "planar"
        else "volume"
        if options.get("stress_backend") == "volume"
        else "Q4"
    )
    require(int(options.get("warmup", 0)) == 730, "Profile warmup differs from730")
    require(
        result.get("conditions_sha256") == source["expected_conditions_sha256"],
        "Conditions hash differs from frozen planned input",
    )
    if module:
        require(
            result.get("complete") is True and result.get("passed") is True and result.get("primary_timing") is False,
            "Module completeness/runtime/scope declarations fail",
        )
        require(
            result.get("actual_native_package") == str(Path(source["source"]) / "genesis/__init__.py"),
            "Module imported native package differs from frozen source",
        )
        for stage in ("before_checks", "after_checks"):
            checks = result.get(stage, {})
            require(
                checks.get("passed") is True
                and all(checks.get("checks", {}).get(name) is True for name in RUNTIME_CHECKS),
                f"{stage} original runtime gates fail",
            )
            require(checks.get("loaded_contacts", 0) > 0, f"{stage} has no loaded contacts")
            for field, limit in (
                ("maximum_relative_force_error", 1e-8),
                ("maximum_relative_moment_error", 1e-8),
                ("maximum_residual_over_gate", 1),
            ):
                value = checks.get(field)
                require(finite(value) and 0 <= value <= limit, f"{stage} numeric {field} contradicts original gate")
        identity = result.get("identity", {})
        row["runtime_identity"] = identity
        inventory = identity.get("nvidia_smi_visible_inventory", "")
        row["gpu_uuids"] = re.findall(r"GPU-[a-fA-F0-9-]+", inventory)
        row["host"] = identity.get("hostname")
        row["slurm"] = identity.get("environment")
        require(
            bool(row["gpu_uuids"]) and identity.get("gpu_inventory_error") is None,
            "Module GPU UUID inventory unavailable",
        )
        require(bool(row["host"]), "Module host identity unavailable")
        require(identity.get("environment", {}).get("GENESIS_IMAGE_VER") == "1_26", "Unexpected module container image")
        worker = "research/rigid_stress/qipc_comparison/profile_planar_loading.py"
        expected = next((item.get("sha256") for item in analysis if item.get("path") == worker), None)
        require(
            expected is not None and result.get("actual_worker_sha256") == expected,
            "Module worker hash differs from captured analysis inventory",
        )
        row["recorded_worker_sha256"] = result.get("actual_worker_sha256")
        annotation = "PlanarLoading.module"
        row["snapshot_check"] = result.get("snapshot_check")
        row["runtime_checks"] = {stage: result.get(stage) for stage in ("before_checks", "after_checks")}
        row["routes"] = {stage: result.get(stage) for stage in ("routes_before", "routes_after", "terminal_task_work")}
        observed = stats["events_by_category_and_name"]["kernel"]
        recorded = {item["name"]: (item["sum_duration_us"], item["calls"]) for item in result.get("device_kernels", [])}
        require(
            recorded == {item["name"]: (item["sum_duration_us"], item["calls"]) for item in observed},
            "Recorded module kernel summary differs from trace",
        )
        require(all(item["calls"] == 10 for item in observed), "Module kernel does not occur10times")
        require(
            {item["category"]: item["sum_duration_us"] for item in result.get("device_categories", [])}
            == {item["category"]: item["sum_duration_us"] for item in stats["device_categories"]},
            "Recorded module category summary differs from trace",
        )
        row["actual_method"] = result.get("actual_method")
        row["actual_anchor_modes"] = result.get("effective_anchor_modes")
        row["actual_planar_query_modes"] = result.get("effective_planar_query_modes")
    else:
        require(task["module"] == "examples.speed_benchmark.rigid_stress", "Unexpected policy worker module")
        require(result.get("source_revision") == revision, "Policy benchmark source revision mismatch")
        require(
            result.get("cuda_graphs_enabled") is False, "Policy benchmark graph environment flag is enabled/unknown"
        )
        require(result.get("policy", {}).get("weights_sha256") == weights, "Actual policy weights mismatch")
        policy = result.get("policy", {})
        require(
            policy.get("dtype") == "float32"
            and finite(policy.get("warmup_reference_max_error_rad"))
            and finite(policy.get("control_error_budget_rad"))
            and policy["warmup_reference_max_error_rad"] <= policy["control_error_budget_rad"],
            "Policy FP32/FP64 warmup control metadata fail",
        )
        require(result.get("scope") == "policy", "Policy trace scope mismatch")
        trace_metadata = result.get("trace_metadata", {})
        require(
            trace_metadata.get("steps") == 10
            and trace_metadata.get("start_tick") == 600
            and trace_metadata.get("end_tick") == 610,
            "Policy trace interval differs from600..610",
        )
        require(trace_metadata.get("invalid_environment_steps") == 0, "Policy trace has invalid environment steps")
        expected_resets = row["envs"] // 4
        require(
            trace_metadata.get("resets") == expected_resets,
            "Policy trace reset environment count differs from the recorded control wave",
        )
        repeats = result.get("repeats", [])
        require(len(repeats) == 1 and repeats[0].get("steps") == 600, "Policy pre-trace workflow is incomplete")
        require(
            all(
                item.get("invalid_environment_steps") == 0
                and item.get("scatter_overflow_calls") == 0
                and item.get("failed_steps_per_environment") == [0] * row["envs"]
                for item in repeats
            ),
            "Policy pre-trace invalid/overflow/per-env counters fail",
        )
        row["trace_interval"] = trace_metadata
        row["pretrace_workflow_counters"] = [
            {
                key: item.get(key)
                for key in (
                    "steps",
                    "invalid_environment_steps",
                    "scatter_overflow_calls",
                    "failed_steps_per_environment",
                    "resets",
                    "planar_routes",
                )
            }
            for item in repeats
        ]
        row["policy"] = result.get("policy")
        row["actual_command"] = result.get("command")
        row["host"], row["pid"], row["slurm_job_id"] = result.get("host"), result.get("pid"), result.get("slurm_job_id")
        row["gpu_uuids"] = [result.get("gpu_uuid")]
        require(
            stats["cpu_process_ids_from_trace_labels"] == [row["pid"]],
            "Policy actual PID differs from trace CPU process PID",
        )
        require(
            isinstance(result.get("gpu_uuid"), str) and result["gpu_uuid"].startswith("GPU-"),
            "Policy GPU UUID unavailable",
        )
        row["physical_assets"] = result.get("physical_assets")
        row["recorded_source_sha256"] = result.get("source_sha256")
        expected_files = {item["path"]: item for item in source.get("files", [])}
        mismatches = []
        for name, digest in (result.get("source_sha256") or {}).items():
            path = Path(name)
            relative = str(path.relative_to(source["source"])) if path.is_absolute() else name
            if expected_files.get(relative, {}).get("sha256") != digest:
                mismatches.append(name)
        require(
            bool(row["recorded_source_sha256"]) and not mismatches,
            "Policy source file hashes differ from frozen capture",
        )
        require(
            bool(row["physical_assets"])
            and all(
                expected_files.get(name, {}).get("sha256") == item.get("sha256")
                and expected_files.get(name, {}).get("bytes") == item.get("bytes")
                for name, item in row["physical_assets"].items()
            ),
            "Policy physical asset hashes differ from frozen capture",
        )
        row["actual_method"] = result.get("method_selected")
        row["actual_anchor_modes"] = result.get("effective_anchor_modes")
        row["actual_planar_query_modes"] = result.get("effective_planar_query_modes")
        row["actual_fused_pipeline"] = result.get("effective_fused_pipeline")
        row["actual_rigid_post_fusion"] = result.get("effective_rigid_post_fusion")
        annotation = "rigid_stress_trace_step"
    require(stats["user_annotation_counts"].get(annotation) == 10, "Trace does not contain10 recorded step annotations")
    if row["variant"] == "planar32":
        require(row["actual_planar_query_modes"] == ["blocks4"], "Planar profile did not execute blocks4 query")
        require(
            int(options.get("quadrature", 0)) == 10
            and int(options.get("loading_subdivisions", 0)) == 32
            and int(options.get("tasks_per_env", options.get("scatter_tasks_per_env", 0))) == 128,
            "Planar selected degree/subdivisions/capacity mismatch",
        )
        if not module:
            for repeat in result["repeats"]:
                routes = repeat.get("planar_routes", {})
                delta = routes.get("per_environment_delta", {})
                require(
                    routes.get("full_interval_counts_available") is True
                    and routes.get("environment_resets_inside_timer") == 0
                    and delta.get("overflow_steps") == [0] * row["envs"]
                    and delta.get("material_rejections") == [0] * row["envs"],
                    "Planar pre-trace route counters are incomplete or rejected/overflowed",
                )
    if row["variant"] != "volume":
        require(
            row["actual_anchor_modes"] == ["cooperative_cuda_packed"],
            "Shell profile did not execute cooperative packed anchors",
        )
    row["pid_from_trace"] = stats["cpu_process_ids_from_trace_labels"]
    if module:
        row["pid"] = row["pid_from_trace"][0] if len(row["pid_from_trace"]) == 1 else None
        row["pid_source"] = "Trace CPU process_labels metadata"
    else:
        row["pid_source"] = "Actual benchmark PID, matched to trace CPU process_labels metadata"
    require(row.get("actual_method") == "inverse", "Profile selected an unexpected elastic recovery method")
    if row["variant"] == "volume":
        require(
            row.get("actual_fused_pipeline") == [True]
            and row.get("actual_rigid_post_fusion") is True
            and int(options.get("quadrature", 0)) == 128
            and options.get("pressure_quadrature") == "angle"
            and any(
                "kernel_rigid_stress_post_" in item["name"] for item in stats["events_by_category_and_name"]["kernel"]
            ),
            "Volume matched angle128 joint rigid/stress post route unavailable",
        )
    require(read_bytes(marker)[1] == row["returncode_identity"], "Terminal returncode changed while aggregating")
    row["passed"] = not issues
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--policy-weights-sha256", required=True)
    args = parser.parse_args()
    source, source_identity = read_json(args.campaign / "source.json")
    tasks, tasks_identity = read_json(args.campaign / "tasks.json")
    analysis, analysis_identity = read_json(args.campaign / "analysis-files.json")
    issues = []
    if source.get("revision") != args.source_revision:
        issues.append("Campaign source revision differs from requested source")
    if source.get("patch_sha256") != hashlib.sha256(b"").hexdigest():
        issues.append("Frozen campaign has an engine patch")
    first_conditions = Path(arguments(tasks[0][0]["args"])["conditions"])
    _, conditions_identity = read_bytes(first_conditions)
    source["expected_conditions_sha256"] = conditions_identity["sha256"]
    rows = []
    for group, definitions in enumerate(tasks):
        for task in definitions:
            try:
                rows.append(
                    audit_task(
                        args.campaign, group, task, source, analysis, args.source_revision, args.policy_weights_sha256
                    )
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                rows.append({"group": group, "name": task.get("name"), "passed": False, "issues": [str(error)]})
    counts = Counter(row.get("kind", "unread") for row in rows)
    if len(rows) != 14 or counts != {"module": 8, "policy": 6}:
        issues.append("Final profile requires exactly14workers:8module and6policy")
    expected_cases = Counter(
        [
            ("module", batch, mode, variant)
            for batch, mode in ((4, "max"), (1024, "max"), (8192, "max"), (8192, "full"))
            for variant in ("Q4", "planar32")
        ]
        + [("policy", batch, "max", variant) for batch in (4, 1024) for variant in ("Q4", "planar32", "volume")]
    )
    if (
        Counter((row.get("kind"), row.get("envs"), row.get("output_mode"), row.get("variant")) for row in rows)
        != expected_cases
    ):
        issues.append("Final profile worker case matrix differs from the14 planned roles")
    pairs = []
    for group in range(4):
        items = [row for row in rows if row["group"] == group]
        identities = [row.get("snapshot_check", {}).get("physical_source_identity", {}) for row in items]
        keys = ("sha256", "columns", "shape", "dtype", "loaded_slot_mask_sha256", "ordered_original_anchor_sha256")
        same = len(items) == 2 and all(
            identities[0].get(key) is not None and identities[0].get(key) == identities[1].get(key) for key in keys
        )
        pairs.append(
            {
                "group": group,
                "tasks": [row["name"] for row in items],
                "passed": same,
                "physical_source_identities": identities,
            }
        )
        if not same:
            issues.append(f"Module group{group} Q4/planar physical inputs differ")
    all_terminal = len(rows) == 14 and all(row.get("returncode") is not None for row in rows)
    complete = all_terminal and not issues and all(row["passed"] for row in rows)
    payload = {
        "schema": "planar-final-profile-summary-v19",
        "complete": complete,
        "passed": complete,
        "all_workers_terminal": all_terminal,
        "source_revision": args.source_revision,
        "workers": len(rows),
        "row_count": len(rows),
        "counts": dict(counts),
        "issues": issues,
        "rows": rows,
        "module_physical_input_pairs": pairs,
        "source_receipt": source_identity,
        "task_manifest": tasks_identity,
        "analysis_inventory": analysis_identity,
        "conditions_identity": conditions_identity,
        "policy_weights_sha256": args.policy_weights_sha256,
        "kernel_category_rules": CATEGORIES,
        "policy_category_overrides_before_generic_response": {
            "rigid_constraint_solver": RIGID_CONSTRAINT_PREFIXES,
            "rigid_dynamics_solve_or_external_wrench": RIGID_DYNAMICS_PREFIXES,
            "known_stress_response": STRESS_RESPONSE_PREFIXES,
            "unknown_generic_solve_apply": "unattributed_solve_or_apply_kernel",
            "scope": "Policy traces only. Module categories retain the captured profile driver's original classification.",
        },
        "fusion_and_accounting_limits": FUSION_LIMITS,
        "primary_timing": False,
        "spatial_qualification_assessed": False,
        "scope": "Read-only standard-library aggregation of14terminal intrusive profiles. Module replays10recoveries on a validated physical snapshot. Policy traces10moving steps after600workflow steps and include a reset wave (1/256environment resets forB4/B1024). All raw kernel, copy and host API events remain in externally hashed traces. No hidden fused substage or launch cost is inferred.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: payload[key] for key in ("complete", "passed", "workers", "row_count", "counts", "issues")}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
