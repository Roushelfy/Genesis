"""Summarize a separate intrusive CUPTI trace without claiming unprofiled rates."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def interval_union_us(intervals):
    end = None
    total = 0.0
    for start, finish in sorted(intervals):
        total += max(0.0, finish - max(start, end if end is not None else start))
        end = max(finish, end if end is not None else finish)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    events = json.loads(raw)["traceEvents"]
    timed = [event for event in events if event.get("ph") == "X" and "dur" in event]
    kernels = [event for event in timed if event.get("cat") == "kernel"]
    copies = [event for event in timed if event.get("cat") in ("gpu_memcpy", "gpu_memset")]
    runtime = [event for event in timed if event.get("cat") == "cuda_runtime"]
    steps = [event for event in timed if event.get("name") == "rigid_stress_trace_step"]
    counts = Counter(event["name"] for event in runtime)
    kernel_totals = Counter()
    for event in kernels:
        kernel_totals[event["name"]] += event["dur"]
    result = {
        "trace_path": str(args.trace),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_bytes": len(raw),
        "step_count": len(steps),
        "step_wall_sum_ms": sum(event["dur"] for event in steps) / 1000,
        "kernel_count": len(kernels),
        "kernel_sum_ms": sum(event["dur"] for event in kernels) / 1000,
        "kernel_interval_union_ms": interval_union_us([(event["ts"], event["ts"] + event["dur"]) for event in kernels])
        / 1000,
        "copy_or_memset_count": len(copies),
        "copy_or_memset_sum_ms": sum(event["dur"] for event in copies) / 1000,
        "runtime_counts": dict(counts),
        "top_kernel_total_ms": [[name, value / 1000] for name, value in kernel_totals.most_common(30)],
        "note": "Intrusive trace includes controller, rigid, stress and scope-dependent policy/observation. "
        "Interval union is trace occupancy, not unprofiled GPU utilization or training throughput.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "runtime_counts"}, indent=2))


if __name__ == "__main__":
    main()
