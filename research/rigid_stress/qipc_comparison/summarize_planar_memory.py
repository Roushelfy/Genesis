"""Audit complete workflow memory receipts using only the Python standard library.

Every planned worker remains in the JSON/CSV, including partial and missing runs.
Rates belong to intrusive memory measurements; spatial accuracy has separate receipts.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .measure_cpu12_workflow_memory import _native_inventory
from .summarize_planar_benchmarks import explicit_arguments, file_identity, is_same_argument

WRAPPER_CHECKS = (
    "driver_completed",
    "complete_fixed_horizon",
    "repetition_count_matches_task",
    "complete_workflow_scope",
    "own_pid_NVIDIA_available",
    "same_gpu",
    "stdlib_collector_before_GPU_imports",
    "task_input_unchanged",
    "exposed_native_stress_inventory",
)
PER_ENVIRONMENT_GROUPS = (
    "planar_source_and_task_state_per_environment",
    "material_bitmap_and_diagnostics_per_environment",
    "original_fallback_scatter_per_environment",
    "original_contacts_per_environment",
    "response_and_output_per_environment",
)
SHARED_GROUPS = ("planar_geometry_shared", "original_surface_and_curvature_shared")


@dataclass
class MemoryAudit:
    campaign: str
    group: int
    name: str
    issues: list[str] = field(default_factory=list)
    row: dict = field(default_factory=dict)

    def require(self, condition, message):
        if not condition:
            self.issues.append(message)


def read_json(path, issues):
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        issues.append(f"Cannot read {path}: {error}")
        return None


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def matching_identity(recorded, path):
    actual = file_identity(path)
    return isinstance(recorded, dict) and all(recorded.get(key) == actual[key] for key in ("bytes", "sha256"))


def sampler_receipt(audit, result, benchmark, directory):
    """Cross-check the import-boundary ready event and first preferred-PID NVIDIA match."""
    memory, polling = result.get("memory", {}), result.get("polling", {})
    mapping, ready = result.get("pid_mapping", {}), memory.get("ready", {})
    pid, preferred, gpu = benchmark.get("pid"), mapping.get("preferred_pid"), benchmark.get("gpu_uuid")
    audit.require(isinstance(pid, int) and pid > 0 and mapping.get("container_pid") == pid, "Benchmark PID mismatch")
    audit.require(isinstance(preferred, int) and preferred > 0, "Missing preferred outermost NVIDIA PID")
    audit.require(memory.get("pid_mapping") == mapping, "Sampler/wrapper PID mappings differ")
    audit.require(
        ready.get("event") == "ready"
        and ready.get("target_parent_pid") == pid
        and isinstance(ready.get("pid"), int)
        and ready["pid"] > 0
        and ready["pid"] != pid
        and ready.get("scientific_modules_loaded") == []
        and result.get("scientific_modules_before_sampler") == [],
        "Collector was not ready before scientific imports in a separate process",
    )
    audit.require(
        memory.get("available") is True
        and memory.get("collector_returncode") == 0
        and memory.get("start_error") is None
        and memory.get("parse_errors") == []
        and finite_nonnegative(memory.get("sampled_peak_device_bytes"))
        and memory.get("sampled_peak_device_bytes", 0) > 0,
        "Own-process sampled NVIDIA peak unavailable or collector failed",
    )
    peaks = memory.get("sampled_peak_bytes_by_gpu", {})
    audit.require(
        bool(gpu)
        and set(peaks) == {gpu}
        and peaks.get(gpu) == memory.get("sampled_peak_device_bytes")
        and polling.get("matched_gpu_uuids") == [gpu],
        "NVIDIA samples do not match the benchmark GPU",
    )
    audit.require(
        polling.get("sampled_peak_is_lower_bound") is True
        and polling.get("sampling_outside_parent_GIL") is True
        and polling.get("poll_count", 0) > 1
        and memory.get("matched_poll_count", 0) > 0
        and all(
            finite_nonnegative(polling.get(key))
            for key in (
                "observed_interval_min_seconds",
                "observed_interval_median_seconds",
                "observed_interval_max_seconds",
            )
        )
        and finite_nonnegative(memory.get("requested_interval_seconds")),
        "Polling interval/lower-bound receipts incomplete",
    )
    intervals = [
        polling.get(key)
        for key in (
            "observed_interval_min_seconds",
            "observed_interval_median_seconds",
            "observed_interval_max_seconds",
        )
    ]
    audit.require(
        all(finite_nonnegative(value) for value in intervals)
        and 0 < intervals[0] <= intervals[1] <= intervals[2]
        and polling.get("poll_count") == memory.get("observed_poll_count")
        and memory.get("matched_poll_count", 0) + memory.get("unmatched_poll_count", 0)
        == memory.get("observed_poll_count"),
        "Observed polling intervals/counts are inconsistent",
    )
    raw = directory / "nvidia-process-memory.jsonl"
    if not raw.is_file() and isinstance(memory.get("raw_log"), str):
        raw = Path(memory["raw_log"])
    first, match = None, None
    scanned = 0
    try:
        with raw.open() as stream:
            first = json.loads(next(stream))
            for line in stream:
                scanned += 1
                record = json.loads(line)
                if record.get("matches"):
                    match = record
                    break
    except (OSError, StopIteration, UnicodeDecodeError, json.JSONDecodeError) as error:
        audit.issues.append(f"Sampler readiness/first-match readback unavailable: {error}")
    audit.require(first == ready, "Raw first collector event differs from its readiness receipt")
    audit.require(
        isinstance(match, dict)
        and match.get("selected_pid") == preferred
        and match.get("returncode") == 0
        and all(sample.get("pid") == preferred and sample.get("gpu_uuid") == gpu for sample in match["matches"]),
        "Raw NVIDIA first match does not establish preferred PID ownership on this GPU",
    )
    return {
        "sampled_NVIDIA_peak_bytes_lower_bound": memory.get("sampled_peak_device_bytes"),
        "requested_poll_interval_seconds": memory.get("requested_interval_seconds"),
        "actual_polling": polling,
        "first_ready_event": first,
        "first_matching_seconds": memory.get("first_matching_seconds"),
        "last_matching_seconds": memory.get("last_matching_seconds"),
        "preferred_NVIDIA_pid": preferred,
        "container_pid": pid,
        "raw_poll_log": str(raw),
        "raw_lines_read_through_first_match": scanned + 1,
        "poll_readback_scope": "First ready and first preferred-PID match independently checked; complete peak/interval accounting is retained in the original collector receipt.",
    }


def inventory_receipt(audit, result, benchmark):
    """Recompute equal-pointer accounting and retain named allocation groups."""
    try:
        computed = _native_inventory(benchmark)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        audit.issues.append(f"Invalid exposed native-buffer inventory: {error}")
        return {}
    recorded, native = result.get("native_stress_buffer_audit", {}), result.get("native_memory", {})
    audit.require(
        recorded.get("passed") is True
        and isinstance(recorded.get("checks"), dict)
        and bool(recorded["checks"])
        and all(value is True for value in recorded["checks"].values())
        and computed["passed"] is True,
        "Exposed native stress inventory failed accounting checks",
    )
    for key in ("groups", "declared_aliases", "native_stress_bytes", "logical_stress_bytes_before_alias_dedup"):
        audit.require(recorded.get(key) == computed[key], f"Native inventory readback differs: {key}")
    if not computed["pure_rigid"]:
        audit.require(
            native.get("native_buffer_bytes") == computed["native_stress_bytes"]
            and native.get("native_logical_buffer_bytes") == computed["logical_stress_bytes_before_alias_dedup"],
            "Wrapper native memory totals differ from exposed inventory",
        )
    groups = {}
    for name, values in computed["groups"].items():
        groups[name] = {key: value for key, value in values.items() if key != "buffers"}
        groups[name]["buffer_count"] = len(values["buffers"])
        groups[name]["classification"] = (
            "named shared geometry"
            if name in SHARED_GROUPS
            else "named per-environment group; bytes/env includes any fixed scalar counters"
            if name in PER_ENVIRONMENT_GROUPS
            else "other/unclassified sharing; no all-shared claim"
        )
    return {
        "equal_pointer_deduplicated_stress_bytes": computed["native_stress_bytes"],
        "logical_stress_bytes": computed["logical_stress_bytes_before_alias_dedup"],
        "groups": groups,
        "named_shared_geometry_bytes": sum(
            groups.get(name, {}).get("declared_unique_bytes", 0) for name in SHARED_GROUPS
        ),
        "other_unclassified_stress_bytes": groups.get("other_stress_geometry_and_operators", {}).get(
            "declared_unique_bytes", 0
        ),
        "declared_equal_pointer_aliases": computed["declared_aliases"],
        "deduplication_scope": "Equal exposed tensor-view data pointers only. Missing backing-storage bases, sizes and byte ranges prevent certification of offset views sharing storage.",
        "process_scope": "Exposed stress buffers only; rigid/collider/controller/reset snapshots, JIT/CUDA contexts and allocator residency are measured separately at process level.",
        "pure_rigid_stress_buffers_only": computed["pure_rigid"],
        "other_native_memory": {
            key: value
            for key, value in native.items()
            if key not in ("native_buffer_bytes", "native_logical_buffer_bytes")
        },
    }


def source_receipt(audit, result, benchmark, manifest, analysis, revision):
    audit.require(manifest.get("revision") == revision, "Captured engine revision mismatch")
    audit.require(manifest.get("patch_sha256") == hashlib.sha256(b"").hexdigest(), "Captured native engine has a patch")
    audit.require(
        result.get("source_revision") == benchmark.get("source_revision") == revision, "Worker source pin mismatch"
    )
    files = {row["path"]: row["sha256"] for row in manifest.get("files", [])}
    reported = benchmark.get("source_sha256", {})
    audit.require(isinstance(reported, dict) and bool(reported), "Missing actual source hashes")
    if not isinstance(reported, dict):
        return
    covered = set()
    for name, value in reported.items():
        matches = [(path, digest) for path, digest in files.items() if name == path or name.endswith("/" + path)]
        digest = value.get("sha256") if isinstance(value, dict) else value
        audit.require(len(matches) == 1 and matches[0][1] == digest, f"Actual source hash differs: {name}")
        if len(matches) == 1:
            covered.add(matches[0][0])
    benchmark_file = "examples/speed_benchmark/rigid_stress.py"
    required = {
        name for name in files if name.startswith("genesis/engine/solvers/rigid/stress/") and name.endswith(".py")
    }
    required.update(
        (
            benchmark_file,
            "genesis/options/rigid_stress.py",
            "examples/rigid/franka_egg_stress.py",
            "examples/rigid/egg_stress_controller.py",
        )
    )
    audit.require(required <= covered, "Actual source hash coverage is incomplete")
    audit.require(
        benchmark_file in files
        and isinstance(result.get("actual_worker"), dict)
        and result["actual_worker"].get("sha256") == files[benchmark_file],
        "Actual benchmark module hash mismatch",
    )
    audit.require(result.get("source_sha256") == reported, "Wrapper/benchmark source hash receipts differ")
    for key, filename in (
        ("worker", "measure_cpu12_workflow_memory.py"),
        ("sampler_worker", "measure_cpu12_memory.py"),
    ):
        matches = [row["sha256"] for row in analysis if row["path"].endswith("/" + filename)]
        audit.require(
            len(matches) == 1 and result.get(key, {}).get("sha256") == matches[0], f"Captured {key} hash mismatch"
        )


def audit_task(campaign, group, task, manifest, analysis, revision, weights):
    audit = MemoryAudit(str(campaign), group, task.get("name", "<missing-name>"))
    name = audit.name
    audit.require(Path(name).name == name and name not in (".", ".."), "Task name is not a safe directory basename")
    directory = campaign / f"group{group}" / name
    audit.row = {"campaign": str(campaign), "group": group, "name": name, "directory": str(directory)}
    try:
        code = int(directory.with_name(name + ".returncode.txt").read_text().strip())
    except (OSError, ValueError) as error:
        code = None
        audit.issues.append(f"Missing/invalid terminal return code: {error}")
    audit.require(code == 0, f"Worker did not terminate with RC0: {code}")
    audit.row["returncode"] = code
    result = read_json(directory / "result.json", audit.issues)
    benchmark = read_json(directory / "benchmark.json", audit.issues)
    if not isinstance(result, dict) or not isinstance(benchmark, dict):
        audit.issues.append("Missing or malformed wrapper/benchmark JSON object")
        return audit
    audit.row.update(
        result_file=file_identity(directory / "result.json"), benchmark_file=file_identity(directory / "benchmark.json")
    )
    audit.require(result.get("complete") is True and result.get("passed") is True, "Wrapper is incomplete or failed")
    if result.get("complete") is not True:
        return audit
    for key in ("memory", "polling", "pid_mapping", "native_memory", "native_stress_buffer_audit", "preparation"):
        if not isinstance(result.get(key), dict):
            audit.issues.append(f"Missing or malformed wrapper metadata: {key}")
            return audit
    if not isinstance(benchmark.get("policy"), dict):
        audit.issues.append("Missing actual policy metadata")
        return audit
    checks = result.get("checks", {})
    audit.require(
        isinstance(checks, dict)
        and all(checks.get(key) is True for key in WRAPPER_CHECKS)
        and all(value is True for value in checks.values()),
        "Wrapper checks are incomplete or false",
    )
    audit.require(
        all(result.get(key) is None for key in ("driver_error", "memory_error", "benchmark_parse_error"))
        and result.get("benchmark_recovered_from_exception") is False,
        "Driver/collector failed or benchmark was recovered from an exception",
    )
    audit.require(result.get("primary_timing") is False, "Memory worker is mislabeled as primary timing")
    audit.require(result.get("genesis_image_version") == "1_26", "Worker image is not1_26")
    audit.require(
        matching_identity(result.get("benchmark"), directory / "benchmark.json"), "Benchmark file hash mismatch"
    )
    try:
        wrapper_args = explicit_arguments(task["args"])
        task_path = Path(wrapper_args["task"])
        definition = read_json(task_path, audit.issues)
        audit.require(
            task.get("module") == "research.rigid_stress.qipc_comparison.measure_cpu12_workflow_memory",
            "Unexpected memory wrapper module",
        )
        if not isinstance(definition, dict) or definition.get("module") != "examples.speed_benchmark.rigid_stress":
            raise ValueError("Task definition does not invoke the frozen benchmark")
        expected = explicit_arguments(definition["args"])
        audit.require(definition == result.get("task_definition"), "Saved/current task definitions differ")
        audit.require(matching_identity(result.get("task"), task_path), "Task definition file hash mismatch")
        actual = benchmark.get("arguments", {})
        if not isinstance(actual, dict):
            raise TypeError("Parsed benchmark arguments are missing")
        audit.require(actual == result.get("benchmark_arguments"), "Wrapper/benchmark parsed arguments differ")
        for option, value in expected.items():
            audit.require(is_same_argument(value, actual.get(option)), f"Actual argument mismatch: {option}")
    except (KeyError, TypeError, ValueError) as error:
        audit.issues.append(f"Task argument contract invalid: {error}")
        return audit
    for key, value in (("steps", 600), ("warmup", 730), ("repetitions", 1), ("minimum_seconds", 0)):
        audit.require(is_same_argument(expected.get(key), value), f"Fixed protocol mismatch: {key}")
    audit.require(expected.get("source_revision") == revision, "Task does not explicitly bind the requested source")
    pure = expected.get("scope") == "policy-rigid"
    backend = expected.get("stress_backend")
    variant = (
        "pure_rigid"
        if pure
        else "volume"
        if backend == "volume"
        else "planar"
        if expected.get("shell_loading") == "planar"
        else "Q4"
    )
    audit.require(expected.get("scope") in ("policy", "policy-rigid"), "Unexpected complete workflow scope")
    audit.require(expected.get("output_mode") in ("max", "full"), "Missing max/full output flag")
    batch = actual.get("envs")
    audit.require(isinstance(batch, int) and batch > 0, "Invalid batch size")
    audit.require(
        benchmark.get("scope") == expected.get("scope") and benchmark.get("envs") == batch,
        "Benchmark scope/batch metadata mismatch",
    )
    audit.require(benchmark.get("output_mode") == expected.get("output_mode"), "Benchmark max/full metadata mismatch")
    audit.require(benchmark.get("cuda_graphs_enabled") is False, "Actual CUDA graph mode is not OFF")
    fused = [] if pure else [backend == "volume" and expected.get("unfused_pipeline") is not True]
    anchor = (
        []
        if pure
        else [
            "serial_fused_pipeline"
            if fused == [True]
            else "cooperative_cuda_packed"
            if expected.get("cooperative_anchor") is True
            else "serial"
        ]
    )
    query = (
        []
        if pure
        else [
            "blocks4"
            if variant == "planar" and expected.get("planar_query") == "blocks"
            else "cells"
            if variant == "planar"
            else "not_used"
        ]
    )
    audit.require(benchmark.get("effective_fused_pipeline") == fused, "Actual fused pipeline route mismatch")
    audit.require(benchmark.get("effective_anchor_modes") == anchor, "Actual anchor route mismatch")
    audit.require(benchmark.get("effective_planar_query_modes") == query, "Actual planar query route mismatch")
    policy = benchmark.get("policy", {})
    audit.require(policy.get("weights_sha256") == weights, "Actual policy weights SHA256 mismatch")
    audit.require(
        policy.get("seed") == 99173
        and policy.get("layers") == [26, 128, 128, 7]
        and policy.get("dtype") == "float32"
        and policy.get("residual_scale_rad") == 1e-4
        and policy.get("control_error_budget_rad") == 1e-9
        and finite_nonnegative(policy.get("warmup_reference_max_error_rad"))
        and policy.get("warmup_reference_max_error_rad", math.inf) <= 1e-9,
        "Policy inference contract mismatch",
    )
    repeats = benchmark.get("repeats", [])
    audit.require(isinstance(repeats, list) and len(repeats) == 1, "Expected exactly one repetition")
    audit.require(
        result.get("expected_steps") == 600 and result.get("expected_repetitions") == 1,
        "Wrapper horizon/repetition expectations differ",
    )
    audit.require(result.get("repeat_receipts") == repeats, "Wrapper/benchmark repetition receipts differ")
    for repeat in repeats:
        audit.require(
            repeat.get("steps") == 600
            and finite_nonnegative(repeat.get("seconds"))
            and repeat.get("seconds", 0) > 0
            and repeat.get("invalid_environment_steps") == 0
            and repeat.get("scatter_overflow_calls") == 0,
            "Incomplete horizon or invalid/overflow work in repetition",
        )
        failed = repeat.get("failed_steps_per_environment")
        audit.require(
            failed is None
            if pure
            else isinstance(failed, list) and len(failed) == batch and all(v == 0 for v in failed),
            "Failed-environment counters missing or nonzero",
        )
    audit.require(
        result.get("reported_invalid_environment_steps") == [0]
        and result.get("total_reported_invalid_environment_steps") == 0,
        "Wrapper invalid counters are not zero",
    )
    prep = result.get("preparation", {})
    audit.require(prep.get("warmup_steps") == benchmark.get("warmup_steps") == 730, "Warmup receipt mismatch")
    warmup = benchmark.get("warmup_quality", [])
    counts = (
        [
            value.get("nonfinite_count")
            for record in warmup
            if isinstance(record, dict)
            for value in record.values()
            if isinstance(value, dict)
        ]
        if isinstance(warmup, list)
        else []
    )
    nonfinite = sum(counts) if counts and all(isinstance(value, int) and value >= 0 for value in counts) else None
    audit.require(
        benchmark.get("warmup_diagnostics") == "ranges_and_nonfinite_counts"
        and isinstance(warmup, list)
        and bool(warmup)
        and nonfinite == 0,
        "Warmup diagnostic samples are missing or contain nonfinite values",
    )
    audit.require(finite_nonnegative(prep.get("setup_seconds")), "Setup timer missing or nonfinite")
    audit.require(
        all(benchmark.get(key) == value for key, value in prep.items()), "Wrapper/benchmark preparation receipts differ"
    )
    rss = result.get("peak_process_rss_bytes")
    audit.require(
        finite_nonnegative(rss) and rss > 0 and rss == result.get("process_max_rss_kib", -1) * 1024,
        "Process RSS high-water receipt invalid",
    )
    source_receipt(audit, result, benchmark, manifest, analysis, revision)
    inventory = inventory_receipt(audit, result, benchmark)
    audit.row.update(
        variant=variant,
        batch=batch,
        output_mode=benchmark.get("output_mode"),
        output_scope="unused stress flag in pure rigid control" if pure else "stress sampled max/full field output",
        scope=benchmark.get("scope"),
        gpu_uuid=benchmark.get("gpu_uuid"),
        host=benchmark.get("host"),
        pid=benchmark.get("pid"),
        source_revision=benchmark.get("source_revision"),
        policy_weights_sha256=policy.get("weights_sha256"),
        cuda_graphs_enabled=benchmark.get("cuda_graphs_enabled"),
        cuda_graph_mode_provenance="Benchmark process-environment receipt. The frozen helper exports QD_GRAPH=0 before scientific imports; frozen Quadrants import-time environment semantics establish GraphOff. This field does not directly read the imported _GRAPH_ENABLED constant; bounded graph-control screens carry that separate evidence.",
        actual_fused_pipeline=benchmark.get("effective_fused_pipeline"),
        actual_anchor_modes=benchmark.get("effective_anchor_modes"),
        actual_query_modes=benchmark.get("effective_planar_query_modes"),
        peak_process_RSS_high_water_bytes=rss,
        memory=sampler_receipt(audit, result, benchmark, directory),
        native_stress=inventory,
        preparation={key: value for key, value in prep.items() if key != "warmup_quality"},
        preparation_scope=result.get("preparation_scope"),
        intrusive_repetition_diagnostics=repeats,
        warmup_sampled_nonfinite_count=nonfinite,
        warmup_counter_scope="Warmup diagnostics are sampled every10steps. Zero-invalid claims use complete reported600step repeat counters; actual-policy accuracy qualification separately checks every warmup/audit step.",
        primary_timing=False,
        spatial_qualification_established=False,
        instrumentation_scope="All nested rates are intrusive memory diagnostics and excluded from primary speedup headlines.",
    )
    return audit


def summarize(campaigns, revision, weights, accuracy_receipts=()):
    audits, campaign_rows, issues = [], [], []
    for campaign in campaigns:
        errors = []
        groups = read_json(campaign / "tasks.json", errors)
        manifest = read_json(campaign / "source.json", errors)
        analysis = read_json(campaign / "analysis-files.json", errors)
        if not isinstance(groups, list) or not isinstance(manifest, dict) or not isinstance(analysis, list):
            issues.extend([*errors, f"Campaign manifests unavailable: {campaign}"])
            continue
        issues.extend(errors)
        seen = set()
        for group_index, group in enumerate(groups):
            if not isinstance(group, list) or not group:
                issues.append(f"Empty/malformed task group: {campaign}/group{group_index}")
                continue
            group_audits = []
            for task in group:
                if not isinstance(task, dict):
                    issues.append(f"Malformed task: {campaign}/group{group_index}")
                    continue
                key = (group_index, task.get("name"))
                if key in seen:
                    issues.append(f"Duplicate expected worker: {campaign}/{key}")
                seen.add(key)
                audit = audit_task(campaign, group_index, task, manifest, analysis, revision, weights)
                audits.append(audit)
                group_audits.append(audit)
            gpus = {audit.row.get("gpu_uuid") for audit in group_audits if audit.row.get("gpu_uuid")}
            if len(gpus) > 1:
                for audit in group_audits:
                    audit.issues.append("Sequential same-group workers used different GPUs")
        campaign_rows.append(
            {
                "campaign": str(campaign),
                "expected_workers": len(seen),
                "tasks": file_identity(campaign / "tasks.json"),
                "source": file_identity(campaign / "source.json"),
                "analysis": file_identity(campaign / "analysis-files.json"),
            }
        )
    pids = set()
    for audit in audits:
        identity = (audit.row.get("host"), audit.row.get("pid"))
        if identity[1] is not None:
            audit.require(identity not in pids, "Worker host/PID reused; fresh process not established")
            pids.add(identity)
        audit.row.update(qualified=not audit.issues, issues=audit.issues)
    accuracy = []
    for path in accuracy_receipts:
        errors = []
        value = read_json(path, errors)
        accuracy.append(
            {
                "file": file_identity(path) if path.is_file() else {"path": str(path)},
                "issues": errors,
                "recorded_qualification": value.get("all_terminal_passed") if isinstance(value, dict) else None,
                "scope": "Separate actual-policy accuracy qualification; memory workers supply no spatial proof.",
            }
        )
    return {
        "schema": "genesis.planar-memory-summary.v1",
        "source_revision": revision,
        "policy_weights_sha256": weights,
        "expected_workers": sum(row["expected_workers"] for row in campaign_rows),
        "qualified_workers": sum(not audit.issues for audit in audits),
        "overall_qualified": bool(audits) and not issues and all(not audit.issues for audit in audits),
        "campaign_issues": issues,
        "campaigns": campaign_rows,
        "rows": [audit.row for audit in audits],
        "primary_timing": False,
        "spatial_qualification_established": False,
        "separate_accuracy_receipts": accuracy,
        "scope": "Fresh complete-policy process memory, constructor/setup/JIT/warmup and600steps. Wrapper pass plus zero-invalid/source/weights/actual GraphOff/routes and own-PID GPU checks; all planned partial/missing workers retained.",
        "native_memory_scope": "Equal-pointer-deduplicated exposed stress buffers. Named geometry groups are shared; named per-environment groups report total/B including fixed counters. Other operators remain sharing-unclassified. Offset views into backing storage cannot be certified from exposed pointer aliases.",
        "process_peak_scope": "RSS is resource.ru_maxrss high-water bytes. NVIDIA own-process peak is a polling lower bound including CUDA/JIT/allocator residency; requested and observed poll intervals retained.",
        "preparation_scope": "setup_seconds ends after qd.sync and covers whole scene/model/proxy constructor work after Genesis initialization. Component build timers include elastic model JIT; surface timer describes the original surface. Proxy preparation is not separately isolated. First-use runtime JIT during warmup remains inside process memory sampling.",
        "timing_scope": "All repeated rates/seconds are intrusive diagnostics; excluded from primary speedup headlines. Pure rigid max/full flags carry no stress-output meaning.",
        "graph_mode_provenance": "Benchmark cuda_graphs_enabled records process environment, not a direct imported Quadrants constant readback. GraphOff follows the pre-import helper export and frozen dependency import semantics. Dedicated bounded graph screens independently read the imported constant.",
    }


def write_csv(path, rows):
    fields = [
        "campaign",
        "group",
        "name",
        "qualified",
        "variant",
        "batch",
        "output_mode",
        "gpu_uuid",
        "peak_process_RSS_high_water_bytes",
        "sampled_NVIDIA_peak_bytes_lower_bound",
        "native_equal_pointer_bytes",
        "native_logical_bytes",
        "named_shared_geometry_bytes",
        "other_unclassified_stress_bytes",
        "setup_seconds",
        *[name + "_bytes" for name in SHARED_GROUPS],
        *[name + "_bytes_per_env" for name in PER_ENVIRONMENT_GROUPS],
        "issues",
        "primary_timing",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            native = row.get("native_stress", {})
            output = {key: row.get(key) for key in fields}
            output.update(
                sampled_NVIDIA_peak_bytes_lower_bound=row.get("memory", {}).get(
                    "sampled_NVIDIA_peak_bytes_lower_bound"
                ),
                native_equal_pointer_bytes=native.get("equal_pointer_deduplicated_stress_bytes"),
                native_logical_bytes=native.get("logical_stress_bytes"),
                named_shared_geometry_bytes=native.get("named_shared_geometry_bytes"),
                other_unclassified_stress_bytes=native.get("other_unclassified_stress_bytes"),
                setup_seconds=row.get("preparation", {}).get("setup_seconds"),
                issues="; ".join(row["issues"]),
                primary_timing=False,
            )
            for name in PER_ENVIRONMENT_GROUPS:
                output[name + "_bytes_per_env"] = (
                    native.get("groups", {}).get(name, {}).get("declared_unique_bytes_per_environment")
                )
            for name in SHARED_GROUPS:
                output[name + "_bytes"] = native.get("groups", {}).get(name, {}).get("declared_unique_bytes")
            writer.writerow(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--accuracy-receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic", action="store_true", help="Return zero for a partial/failed summary; qualification stays false."
    )
    args = parser.parse_args()
    if (
        re.fullmatch("[0-9a-f]{40}", args.source_revision) is None
        or re.fullmatch("[0-9a-f]{64}", args.weights_sha256) is None
    ):
        parser.error("Bind full source-revision40hex and weights-sha25664hex.")
    if args.output.suffix != ".json":
        parser.error("Use a .json output path; the adjacent .csv is written separately.")
    campaigns = [path.resolve() for path in args.campaign]
    if len(set(campaigns)) != len(campaigns):
        parser.error("Campaigns must be distinct.")
    payload = summarize(campaigns, args.source_revision, args.weights_sha256, args.accuracy_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    write_csv(csv_path, payload["rows"])
    payload["csv"] = file_identity(csv_path)
    payload["summarizer"] = file_identity(Path(__file__).resolve())
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "overall_qualified": payload["overall_qualified"],
                "expected_workers": payload["expected_workers"],
                "qualified_workers": payload["qualified_workers"],
                "output": str(args.output),
            }
        )
    )
    if not payload["overall_qualified"] and not args.diagnostic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
