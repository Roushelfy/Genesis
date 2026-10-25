"""Collect retained GPU candidate receipts and synchronized benchmark summaries (stdlib only)."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def trace_summary(path):
    events = [row for row in json.loads(path.read_text())["traceEvents"] if row.get("ph") == "X"]
    steps = [row for row in events if row.get("cat") == "user_annotation" and row["name"] == "rigid_stress_trace_step"]
    count = len(steps)
    assert count > 0
    groups = defaultdict(list)
    for row in events:
        category, name = row.get("cat", ""), row["name"]
        if category == "kernel" or "memcpy" in name.lower() or "synchron" in name.lower() or "launch" in name.lower():
            groups[(category, name)].append(row["dur"])
    rows = [
        dict(
            category=key[0],
            name=key[1],
            calls=len(values),
            calls_per_step=len(values) / count,
            microseconds_per_step=sum(values) / count,
        )
        for key, values in groups.items()
    ]
    rows.sort(key=lambda row: -row["microseconds_per_step"])
    return dict(
        source=identity(path),
        steps=count,
        events=rows,
        category_counts=dict(Counter(row.get("cat", "") for row in events)),
        note="Intrusive observer timings; kernel events differ from host launches, and nested/overlapping events are not additive.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(run=str(args.run), campaigns=[], benchmarks=[], profiles=[], traces=[], numerics=[], memory=[])
    for campaign in sorted(args.run.iterdir()):
        submission = campaign / "submission.json"
        if not submission.is_file():
            continue
        receipt = dict(
            name=campaign.name,
            submission=json.loads(submission.read_text()),
            source_manifests=[identity(p) for p in sorted(campaign.glob("*source*.json"))],
            tasks=[],
        )
        for path in sorted(campaign.glob("group*/*returncode.txt")):
            receipt["tasks"].append(dict(path=str(path.relative_to(campaign)), returncode=int(path.read_text())))
        result["campaigns"].append(receipt)
        for path in sorted(campaign.glob("group*/*")):
            if not path.is_file() or path.suffix:
                continue
            data = json.loads(path.read_text())
            row = dict(campaign=campaign.name, name=path.name, source=identity(path))
            for key in [
                "source_revision",
                "scope",
                "envs",
                "stress_backend",
                "output_mode",
                "quadrature",
                "pressure_quadrature",
                "precision",
                "history",
                "atlas_regions",
                "atlas_transition",
                "atlas_output",
                "atlas_partition",
                "stress_response",
                "response_precision",
                "response_accumulation",
                "response_block_size",
                "response_output_tile",
                "response_screen_block",
                "condition_count",
                "conditions_sha256",
                "varied",
                "dt",
                "substeps",
                "host",
                "device_name",
                "scene_build_seconds",
                "warmup_seconds",
                "build_timings_including_JIT",
                "stress_memory",
                "memory",
                "memory_bytes",
                "peak_gpu_memory_bytes",
                "torch_memory_allocated_bytes",
                "torch_peak_allocated_bytes",
                "device_free_bytes",
                "trace_metadata",
            ]:
                if key in data:
                    row[key] = data[key]
            if "stress_backend" not in row:
                row["stress_backend"] = data.get("arguments", {}).get("stress_backend")
            # Preserve exact memory inventory keys without guessing the benchmark schema.
            row["memory_inventory"] = {k: v for k, v in data.items() if "memory" in k or "bytes" in k}
            by_kind = Counter()
            for buffer in data.get("native_buffers", []):
                if buffer.get("alias_of") is None:
                    by_kind[str(buffer.get("kind", "unclassified"))] += buffer.get("bytes", 0)
            row["native_buffer_bytes_by_kind"] = dict(by_kind)
            row["preparation"] = {
                k: v
                for k, v in data.items()
                if "build" in k
                or "warmup_seconds" in k
                or k in ["setup_seconds", "surface_quadrature_grid_seconds_including_JIT"]
            }
            row["gpu"] = data.get("gpu")
            row["stress_configurations"] = data.get("stress_configurations")
            if "repeats" in data:
                repeats = [
                    {k: v for k, v in repeat.items() if k != "failed_steps_per_environment"}
                    for repeat in data["repeats"]
                ]
                latency = [repeat["seconds"] / repeat["steps"] * 1000 for repeat in repeats]
                row.update(
                    repeats=repeats,
                    latency_ms=latency,
                    median_ms=statistics.median(latency),
                    admitted_environment_steps_per_second=statistics.median(
                        repeat["transitions_per_second"] for repeat in repeats
                    ),
                    attempted_environment_steps_per_second=statistics.median(
                        repeat["attempted_transitions_per_second"] for repeat in repeats
                    ),
                    admission_scope="Runtime validity only; independent numerical budgets require their separate controls.",
                    invalid_environment_steps=sum(repeat["invalid_environment_steps"] for repeat in repeats),
                )
                result["benchmarks"].append(row)
            else:
                row["profile"] = {k: v for k, v in data.items() if "profile" in k or "stage" in k or "kernel" in k}
                result["profiles"].append(row)
        for path in sorted(campaign.glob("group*/*.trace.json")):
            result["traces"].append(trace_summary(path))
        for path in sorted(campaign.glob("group*/*/result.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                # A still-running worker may be replacing its progress receipt.
                continue
            item = dict(campaign=campaign.name, name=path.parent.name, source=identity(path))
            if "sampled_peak_MiB" in data:
                item.update({key: value for key, value in data.items() if key != "rows"})
                benchmark = path.parent / "benchmark.json"
                if benchmark.is_file():
                    measured = json.loads(benchmark.read_text())
                    item["benchmark"] = identity(benchmark)
                    item["memory_inventory"] = {
                        key: value for key, value in measured.items() if "memory" in key or "bytes" in key
                    }
                    item["native_buffers"] = measured.get("native_buffers")
                    item["stress_configurations"] = measured.get("stress_configurations")
                result["memory"].append(item)
                continue
            rows = data.get("rows", [])
            errors = defaultdict(float)
            for row in rows:
                for key, value in row.get("errors", {}).items():
                    if isinstance(value, (int, float)):
                        errors[key] = max(errors[key], value)
                for key in ("load_error", "grouped_error", "grouped_load_relative", "force_error", "moment_error"):
                    if key in row:
                        errors[key] = max(errors[key], row[key])
            item.update({key: value for key, value in data.items() if key not in ("rows", "comparisons")})
            item["observation_count"] = len(rows)
            item["worst_errors"] = dict(errors)
            loaded = sum(row.get("loaded", 0) for row in rows)
            if loaded:
                item["loaded_prediction_hit_fraction"] = sum(row.get("loaded_hits", 0) for row in rows) / loaded
            blocks = [value for row in rows for value in row.get("evaluated_blocks", [])]
            if blocks:
                item["evaluated_blocks"] = dict(mean=statistics.mean(blocks), maximum=max(blocks))
            contributors = [row.get("contributors", row.get("maximum_contributors", 0)) for row in rows]
            if contributors:
                item["maximum_contributors"] = max(contributors)
                item["coverage_fallbacks"] = sum(row.get("coverage_fallbacks", 0) for row in rows)
            if "comparisons" in data:
                item["comparisons"] = [
                    dict(
                        quadrature=row["quadrature"],
                        passed=row["passed"],
                        worst_errors={key: max(values) for key, values in row["errors"].items()},
                    )
                    for row in data["comparisons"]
                ]
            result["numerics"].append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in result.items() if isinstance(v, list)}))


if __name__ == "__main__":
    main()
