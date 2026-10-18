"""Collect new timing/memory/profile measurements with their actual scopes and failures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def configuration(campaign, group, task):
    folder = campaign / f"group{group}" / task["name"]
    result_path = folder / "result.json"
    task_rc = folder.parent / f"{task['name']}.returncode.txt"
    row = {
        "campaign": str(campaign),
        "configuration": task["name"],
        "group": group,
        "task_returncode": int(task_rc.read_text()) if task_rc.is_file() else None,
        "log_exists": (folder.parent / f"{task['name']}.log").is_file(),
        "result": None,
    }
    if not result_path.is_file():
        return row
    result = json.loads(result_path.read_text())
    repeats = result.get("repeats", [])
    rates = [repeat["aggregate_env_steps_per_second"] for repeat in repeats]
    latency = [repeat["batch_step_ms"] for repeat in repeats]
    numerically_admitted = [
        repeat["valid_aggregate_env_steps_per_second"]
        for repeat in repeats
        if repeat["completed_requested_history"]
        and repeat["final_output_finite_checked_after_timer"]
        and repeat["valid_aggregate_env_steps_per_second"] is not None
    ]
    row.update(
        {
            "identity": file_id(result_path),
            "args": result.get("args"),
            "hardware": result.get("identity"),
            "build_seconds": result.get("build_seconds"),
            "warmup_seconds": result.get("warmup_seconds"),
            "scope": repeats[0]["scope"] if repeats else result.get("scope"),
            "process_memory": result.get("process_memory"),
            "median_aggregate_env_steps_per_second": np.median(rates).item() if rates else None,
            "minimum_aggregate_env_steps_per_second": min(rates) if rates else None,
            "maximum_aggregate_env_steps_per_second": max(rates) if rates else None,
            "median_batch_step_ms": np.median(latency).item() if latency else None,
            "median_numerically_valid_rate": np.median(numerically_admitted).item()
            if len(numerically_admitted) == len(repeats) and repeats
            else None,
            "raw_repeats": repeats,
            "reference_qualified": False,
        }
    )
    return row


def profile(path, output):
    trace = json.loads(path.read_text())
    kernels = defaultdict(lambda: [0, 0.0])
    cpu = defaultdict(lambda: [0, 0.0])
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or event.get("dur", -1) < 0:
            continue
        table = kernels if event.get("cat") == "kernel" else cpu if event.get("cat") == "cpu_op" else None
        if table is not None:
            table[event["name"]][0] += 1
            table[event["name"]][1] += event["dur"]
    total = sum(value[1] for value in kernels.values())
    kernel_rows = [
        {
            "name": name,
            "calls": value[0],
            "summed_duration_us": value[1],
            "fraction_of_summed_kernel_duration": value[1] / total if total else None,
        }
        for name, value in sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)
    ]
    with output.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("name", "calls", "summed_duration_us", "fraction_of_summed_kernel_duration")
        )
        writer.writeheader()
        writer.writerows(kernel_rows)
    return {
        "trace": file_id(path),
        "kernel_table": file_id(output),
        "summed_kernel_duration_us": total,
        "top_kernels": kernel_rows[:20],
        "top_cpu_ops_inclusive": [
            {"name": name, "calls": value[0], "inclusive_duration_us": value[1]}
            for name, value in sorted(cpu.items(), key=lambda item: item[1][1], reverse=True)[:20]
        ],
        "scope": "Intrusive trace attribution. Summed kernel durations can overlap and are not scene latency; CPU-op durations are inclusive and are not added into a total. No primary throughput is derived from traces.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, profiles = [], []
    for campaign in args.campaigns:
        groups = json.loads((campaign / "tasks.json").read_text())
        for group, tasks in enumerate(groups):
            for task in tasks:
                rows.append(configuration(campaign, group, task))
                trace = campaign / f"group{group}" / task["name"] / "trace.json"
                # Only complete worker results admit a profile into this snapshot.
                if trace.is_file() and rows[-1].get("identity") is not None:
                    profiles.append(profile(trace, args.output / f"profile{len(profiles)}-kernels.csv"))
    write_json(
        args.output / "result.json",
        {
            "kind": "new-full-field-max-memory-and-profile-audit",
            "identity": runtime_identity(),
            "configurations": rows,
            "profiles": profiles,
            "scope": "Snapshot of completed worker outputs. Missing results remain pending/not-started or failed according to the retained task return codes/logs; no missing configuration is treated as successful. Primary timing, memory polling and profiles keep their original scopes. Rates are 10ms scene/control env-steps with explicit physical substeps. No physical accuracy qualification or fastest-in-band claim is inferred.",
        },
    )


if __name__ == "__main__":
    main()
