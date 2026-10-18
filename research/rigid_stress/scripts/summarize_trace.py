"""Summarize a separate intrusive CUPTI trace without claiming unprofiled rates."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def interval_union_us(intervals):
    end = None
    lengths = []
    for start, finish in sorted(intervals):
        lengths.append(max(0.0, finish - max(start, end if end is not None else start)))
        end = max(finish, end if end is not None else finish)
    return math.fsum(lengths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    events = json.loads(raw)["traceEvents"]
    timed = [event for event in events if event.get("ph") == "X" and "dur" in event]
    kernels = [event for event in timed if event.get("cat") == "kernel"]
    origin = min((event["ts"] for event in kernels), default=0)
    copies = [event for event in timed if event.get("cat") in ("gpu_memcpy", "gpu_memset")]
    runtime = [event for event in timed if event.get("cat") in ("cuda_runtime", "cuda_driver")]
    # GPU annotations mirror each CPU range on every participating stream;
    # count the original CPU ranges once rather than adding the mirrors.
    steps = [
        event
        for event in timed
        if event.get("name") == "rigid_stress_trace_step" and event.get("cat") == "user_annotation"
    ]
    counts = Counter(event["name"] for event in runtime)
    runtime_totals = Counter()
    for event in runtime:
        runtime_totals[event["name"]] += event["dur"]
    kernel_intervals = [(event["ts"] - origin, event["ts"] - origin + event["dur"]) for event in kernels]
    kernel_union = interval_union_us(kernel_intervals)
    memcpy = [event for event in runtime if event["name"] == "cudaMemcpyAsync"]
    memcpy_intervals = [(event["ts"] - origin, event["ts"] - origin + event["dur"]) for event in memcpy]
    memcpy_union = interval_union_us(memcpy_intervals)
    # Inclusion/exclusion measures actual interval overlap, not a CPU cost model.
    memcpy_overlap = math.fsum((kernel_union, memcpy_union, -interval_union_us(kernel_intervals + memcpy_intervals)))
    step_intervals = [(event["ts"] - origin, event["ts"] - origin + event["dur"]) for event in steps]
    step_union = interval_union_us(step_intervals)
    step_overlap = math.fsum((kernel_union, step_union, -interval_union_us(kernel_intervals + step_intervals)))
    kernel_totals = Counter()
    for event in kernels:
        kernel_totals[event["name"]] += event["dur"]
    copy_counts, copy_bytes, missing_bytes = Counter(), Counter(), Counter()
    for event in copies:
        key = event["cat"] + ": " + event["name"]
        copy_counts[key] += 1
        size = event.get("args", {}).get("bytes")
        if isinstance(size, int) and size >= 0:
            copy_bytes[key] += size
        else:
            missing_bytes[key] += 1
    result = {
        "trace_path": str(args.trace),
        "aggregation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_bytes": len(raw),
        "step_count": len(steps),
        "step_wall_sum_ms": sum(event["dur"] for event in steps) / 1000,
        "kernel_count": len(kernels),
        "kernel_sum_ms": math.fsum(event["dur"] for event in kernels) / 1000,
        "kernel_interval_union_ms": kernel_union / 1000,
        "copy_or_memset_count": len(copies),
        "copy_or_memset_sum_ms": sum(event["dur"] for event in copies) / 1000,
        "explicit_copy_or_memset_bytes": {
            name: {"events": count, "known_bytes": copy_bytes[name], "events_without_byte_count": missing_bytes[name]}
            for name, count in copy_counts.items()
        },
        "runtime_counts": dict(counts),
        "runtime_duration_sum_ms": {name: value / 1000 for name, value in runtime_totals.items()},
        "runtime_memcpy_interval_union_ms": memcpy_union / 1000,
        "runtime_memcpy_gpu_kernel_intersection_ms": memcpy_overlap / 1000,
        "runtime_memcpy_fraction_overlapping_gpu_kernels": memcpy_overlap / memcpy_union if memcpy_union else None,
        "cpu_step_interval_union_ms": step_union / 1000,
        "cpu_step_gpu_kernel_intersection_ms": step_overlap / 1000,
        "top_kernel_total_ms": [[name, value / 1000] for name, value in kernel_totals.most_common(30)],
        "kernel_total_ms": [[name, value / 1000] for name, value in kernel_totals.most_common()],
        "note": "Intrusive trace includes controller, rigid, stress and scope-dependent policy/observation. "
        "Interval endpoints use a relative time origin to avoid rounding at large absolute timestamps. "
        "API/kernel intersections describe timestamp overlap, not CPU waiting versus useful execution. "
        "Do not add API durations to kernel durations or interpret their difference as CPU work. "
        "Interval union is trace occupancy, not unprofiled GPU utilization or training throughput. "
        "Copy bytes describe explicit device-transfer/memset events, not kernel DRAM traffic or bandwidth counters.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "runtime_counts"}, indent=2))


if __name__ == "__main__":
    main()
