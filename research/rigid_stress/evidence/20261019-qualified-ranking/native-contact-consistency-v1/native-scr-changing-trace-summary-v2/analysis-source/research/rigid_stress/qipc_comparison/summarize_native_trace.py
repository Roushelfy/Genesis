"""Summarize complete named profiler scopes without turning intrusive times into rates."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--trace", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.trace.read_text())
    events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    names = ("native_scene_step", "SCR.contact_pipeline", "SCR.capacity_check_and_growth", "SCR.device_scalar_readback")
    scopes = []
    host_events = [e for e in events if e.get("cat") == "user_annotation"]
    count = sum(e["name"] == names[0] for e in host_events)
    assert count == 60, count
    for name in names:
        selected = [e for e in host_events if e["name"] == name]
        assert selected, name
        scopes.append(
            {
                "name": name,
                "calls": len(selected),
                "inclusive_host_us": sum(e["dur"] for e in selected),
                "host_us_per_scene_step": sum(e["dur"] for e in selected) / count,
            }
        )
    kernels, copies = defaultdict(list), defaultdict(list)
    for e in events:
        category = e.get("cat", "")
        if category == "kernel":
            kernels[e["name"]].append(e["dur"])
        if "memcpy" in e["name"].lower() or "synchron" in e["name"].lower():
            copies[category + "/" + e["name"]].append(e["dur"])
    kernel_rows = [
        {"name": k, "calls": len(v), "sum_gpu_us": sum(v), "gpu_us_per_scene_step": sum(v) / count}
        for k, v in kernels.items()
    ]
    kernel_rows.sort(key=lambda row: -row["sum_gpu_us"])
    copy_rows = [{"name": k, "calls": len(v), "sum_inclusive_us": sum(v)} for k, v in copies.items()]
    copy_rows.sort(key=lambda row: -row["sum_inclusive_us"])
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "trace_summary_completed": True,
            "scene_steps": count,
            "scopes": scopes,
            "kernels": kernel_rows,
            "copies_and_sync": copy_rows,
            "trace": file_id(args.trace),
            "original_worker_metadata_failed": True,
            "scope": "Original trace completed after scientific runtime checks, then its worker failed to JSON-encode a NumPy scalar in static config. This independent parse retains that failure, checks all60 observed steps and summarizes raw profiler events. Inclusive host scopes are nested and GPU/copy events can overlap; do not sum them or treat these observer times as admitted production throughput.",
        },
    )


if __name__ == "__main__":
    main()
