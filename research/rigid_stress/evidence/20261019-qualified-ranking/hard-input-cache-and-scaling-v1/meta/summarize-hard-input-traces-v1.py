"""Summarize captured Chrome events without importing the simulation stack."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(path):
    result = json.loads((path.parent / "result.json").read_text())
    events = json.loads(path.read_text())["traceEvents"]
    scopes = [event for event in events if event.get("cat") == "user_annotation" and event["name"].startswith("HertzInputs.")]
    launches = {}
    host_rows = defaultdict(list)
    for event in events:
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        containing = [scope for scope in scopes if scope["pid"] == event["pid"] and scope["tid"] == event["tid"]
                      and scope["ts"] <= event["ts"] and event["ts"] + event["dur"] <= scope["ts"] + scope["dur"]]
        name = min(containing, key=lambda scope: scope["dur"])["name"] if containing else "outside_scene_scope"
        host_rows[name].append(event)
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None:
            assert correlation not in launches, correlation
            launches[correlation] = name
    grouped = defaultdict(list)
    for event in events:
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            name = launches.get(event.get("args", {}).get("correlation"), "unattributed")
            grouped[name].append(event)
    rows = []
    for name in sorted(set(grouped) | set(host_rows)):
        kernels = [event for event in grouped[name] if event["cat"] == "kernel"]
        copies = [event for event in grouped[name] if event["cat"] != "kernel"]
        rows.append({
            "scope": name,
            "kernel_tasks_per_step": len(kernels) / result["steps"],
            "sum_kernel_duration_us_per_step": sum(event["dur"] for event in kernels) / result["steps"],
            "copy_or_memset_events_per_step": len(copies) / result["steps"],
            "sum_copy_or_memset_duration_us_per_step": sum(event["dur"] for event in copies) / result["steps"],
            "driver_runtime_counts": dict(Counter(event["name"] for event in host_rows[name])),
        })
    planar = [event for event in events if event.get("cat") == "kernel" and "kernel_hertz_planar" in event["name"] and "range_for" in event["name"]]
    durations = [event["dur"] for event in planar]
    assert len(durations) == result["steps"]
    return {
        "envs": result["envs"], "steps": result["steps"], "trace": identity(path),
        "scopes": rows,
        "planar_task_us": {"median": statistics.median(durations), "p95": statistics.quantiles(durations, n=100, method="inclusive")[94], "max": max(durations)},
        "planar_launch_metadata": planar[0]["args"],
        "memory": result["memory"],
        "new_shared_geometry_logical_bytes": result["new_shared_geometry_bytes"],
        "new_temporal_cache_logical_bytes_including_diagnostics": result["new_temporal_cache_bytes_including_diagnostics"],
        "host_scopes": result["inclusive_and_exclusive_host_scopes"],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    rows = [summarize(path) for path in sorted(args.campaign.glob("group*/trace/trace.json"))]
    assert rows
    with args.output.open("x") as stream:
        json.dump({"rows": rows, "scope": "Device events are attributed through correlation IDs to CUDA driver/runtime calls enclosed by the smallest named host scope on the same process/thread. GPU annotation parents are excluded to avoid double counting. Durations are sums of actual tasks, not critical-path or warmed inclusive wall time. Launch metadata occupancy is profiler-estimated, not a measured utilization counter. Logical field payload bytes exclude allocator rounding, other state and temporary allocations; device free-memory snapshots are not process memory peaks."}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
