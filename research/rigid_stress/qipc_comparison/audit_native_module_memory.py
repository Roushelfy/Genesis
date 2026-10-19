"""Bind separate sampled module memory to its exact admitted replay configuration."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .artifacts import file_id, runtime_identity, write_json
from .native_module_replay_audit import source_receipt
from .process_memory_receipts import audit_polling


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--primary", type=Path, required=True)
    cli.add_argument("--primary-audit", type=Path, required=True)
    cli.add_argument("--memory", type=Path, required=True)
    cli.add_argument("--require-independent-sampler", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, rows = [], [], []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        assert passed, name

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    def terminal(folder, name=None):
        path = folder / ((name + ".returncode.txt") if name else "returncode.txt")
        receipts.append(file_id(path))
        check("terminal/" + str(path), path.read_text().strip() == "0")

    primary_source, primary_workers = source_receipt(args.primary, receipts)
    memory_source, memory_workers = source_receipt(args.memory, receipts)
    check("same_exact_engine_and_assets", primary_source == memory_source)
    primary_worker_map = {r["path"]: r for r in primary_workers}
    memory_worker_map = {r["path"]: r for r in memory_workers}
    control = "research/rigid_stress/qipc_comparison/native_production_controls.py"
    check("same_recording_and_replay_worker", primary_worker_map[control] == memory_worker_map[control])
    terminal(args.primary_audit.parent.parent)
    audit_plan = read(args.primary_audit.parents[2] / "tasks.json")
    check("primary_audit_single_declared_task", len(audit_plan) == 1 and len(audit_plan[0]) == 1
          and audit_plan[0][0]["name"] == args.primary_audit.parent.name
          and audit_plan[0][0]["module"] == "research.rigid_stress.qipc_comparison.native_module_replay_audit")
    audit = read(args.primary_audit)
    check("primary_audit_passed", audit["passed"] and len(audit["rows"]) == 4)
    check("primary_audit_bound_engine", audit["engine_revision"] == primary_source["revision"])
    for ident in audit["receipts"]:
        check("primary_receipt/" + ident["path"], file_id(ident["path"]) == ident)
    primary_tasks = read(args.primary / "tasks.json")
    memory_tasks = read(args.memory / "tasks.json")
    check("four_modes_in_one_group", len(primary_tasks) == len(memory_tasks) == 1)
    check("four_modes_in_each_campaign", len(primary_tasks[0]) == len(memory_tasks[0]) == 4)
    primary_rows = {r["mode"]: r for r in audit["rows"]}
    terminal(args.primary / "group0")
    terminal(args.memory / "group0")
    for task, profile_task in zip(primary_tasks[0], memory_tasks[0]):
        name = task["name"]
        check(name + "/same_mode", profile_task["name"] == name)
        terminal(args.primary / "group0", name)
        terminal(args.memory / "group0", name)
        primary = read(args.primary / "group0" / name / "result.json")
        profile = read(args.memory / "group0" / name / "result.json")
        check(name + "/all_profile_checks", profile["passed"] and all(profile["checks"].values()))
        check(name + "/separate_profile", profile["primary_timing"] is False)
        task_receipt = profile["task"]
        check(name + "/task_hash", file_id(task_receipt["path"]) == task_receipt)
        memory_task = read(Path(task_receipt["path"]))
        expected_task = json.loads(json.dumps(task))
        expected_task["args"][expected_task["args"].index("--repetitions") + 1] = "1"
        check(name + "/only_repetition_count_differs", memory_task == expected_task)
        recorded = profile["recording_and_replay"]
        check(name + "/profile_record_hash", file_id(recorded["path"]) == recorded)
        diagnostic = read(Path(recorded["path"]))
        for key in (
            "method", "output_mode", "envs", "steps", "warmup", "stress_stride", "dt_s", "substeps",
            "conditions", "stress_configuration", "effective_rigid_post_fusion",
        ):
            check(name + "/matching/" + key, primary[key] == diagnostic[key])
        check(name + "/diagnostic_conformance", diagnostic["passed"] and all(diagnostic["checks"].values()))
        for key in ("complete_history_check_passed", "partial_resets_replayed"):
            check(name + "/" + key, primary["replay"][key] and diagnostic["replay"][key])
        for key in ("recorded_device_bytes", "buffers", "stress_reset_schedule"):
            check(name + "/replay_matching/" + key, primary["replay"][key] == diagnostic["replay"][key])
        for key in ("trajectory",):
            check(name + "/diagnostic_array/" + key, file_id(diagnostic[key]["path"]) == diagnostic[key])
            receipts.append(diagnostic[key])
        ident = diagnostic["replay"]["input_history"]
        check(name + "/diagnostic_input_history", file_id(ident["path"]) == ident)
        receipts.append(ident)
        repeats = diagnostic["replay"]["repeats"]
        check(name + "/complete_diagnostic_horizon", len(repeats) == 1 and repeats[0]["updates"] == 1200
              and repeats[0]["invalid_environment_updates"] == 0)
        samples = profile["memory"]["samples"]
        times = [s["seconds"] for s in samples]
        gaps = [b - a for a, b in zip(times[:-1], times[1:])]
        gpu = diagnostic["gpu_uuid"].removeprefix("GPU-")
        check(name + "/finite_memory_samples", len(samples) > 1 and all(
            math.isfinite(s["seconds"]) and s["seconds"] >= 0 and s["process_mib"] > 0 for s in samples))
        check(name + "/monotone_samples", all(g > 0 for g in gaps))
        check(name + "/same_profile_gpu", all(s["gpu_uuid"].removeprefix("GPU-") == gpu for s in samples))
        peak = max(s["process_mib"] for s in samples)
        check(name + "/sampled_peak_recomputed", profile["memory"]["peak_mib"] == peak)
        coverage = None
        if args.require_independent_sampler:
            coverage = audit_polling(profile["memory"], profile["profile_pid"], diagnostic["gpu_uuid"])
        row = primary_rows[name]
        check(name + "/primary_row_bound", row["result"] == file_id(args.primary / "group0" / name / "result.json"))
        rows.append({
            "mode": name,
            "batch": 8,
            "median_env_updates_per_s": row["median_env_updates_per_s"],
            "min_env_updates_per_s": row["min_env_updates_per_s"],
            "max_env_updates_per_s": row["max_env_updates_per_s"],
            "median_update_batch_ms": row["median_update_batch_ms"],
            "sampled_process_peak_mib": peak,
            "sample_count": len(samples),
            "first_sample_seconds": times[0],
            "last_sample_seconds": times[-1],
            "maximum_observed_sample_gap_seconds": max(gaps),
            "median_observed_sample_gap_seconds": statistics.median(gaps),
            "process_max_rss_kib": profile["process_max_rss_kib"],
            "recorded_device_input_bytes": row["recorded_device_bytes"],
            "primary_gpu": primary["gpu_name"],
            "primary_gpu_uuid": primary["gpu_uuid"],
            "profile_gpu": diagnostic["gpu_name"],
            "profile_gpu_uuid": diagnostic["gpu_uuid"],
            "same_physical_gpu_as_primary": primary["gpu_uuid"].removeprefix("GPU-") == gpu,
            "physical_error": None,
            "independent_polling": coverage,
        })
    with (args.output / "module-throughput-memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({**r, "independent_polling": json.dumps(r["independent_polling"], sort_keys=True)} for r in rows)
    write_json(args.output / "result.json", {
        "identity": runtime_identity(require_gpu_inventory=False),
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "rows": rows,
        "receipts": receipts,
        "engine_revision": primary_source["revision"],
        "source_files_rehashed_per_campaign": len(primary_source["files"]),
        "captured_workers_rehashed": {"primary": len(primary_workers), "memory": len(memory_workers)},
        "physical_reference_qualified": False,
        "capacity_limit_measured": False,
        "independent_sampler_required": args.require_independent_sampler,
        "scope": "B8 complete1200-frame changing collider-history replay, every-step max/full, three warmed primary repeats; separate matched fresh-process memory runs include recording, validation, resident histories and replay. Memory peaks are sampled observations; first/last sample times and observed gaps are not bounds on unseen peaks. No dynamics/policy rate, production-scene memory, independent-process timing dispersion or physical-pad error is inferred.",
    })


if __name__ == "__main__":
    main()
