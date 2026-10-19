"""Join separate native timing, conformance and sampled process-memory receipts."""

import argparse
import csv
import json
from pathlib import Path

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--memory-campaign", type=Path, required=True)
    cli.add_argument("--timing-campaign", type=Path, required=True)
    cli.add_argument("--timing-audit", type=Path, required=True)
    cli.add_argument("--conformance", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    memory_source = read(args.memory_campaign / "source.json")
    timing_source = read(args.timing_campaign / "source.json")
    assert memory_source["revision"] == timing_source["revision"] and len(memory_source["revision"]) == 40
    assert memory_source["files"] == timing_source["files"]
    for row in memory_source["files"]:
        actual = file_id(Path(memory_source["source"]) / row["path"])
        assert actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"]
    timing = read(args.timing_audit)
    conformance = read(args.conformance)
    assert timing["timing_receipts_passed"] and conformance["passed"]
    assert all(row["passed"] for row in conformance["checks"])
    batches = sorted({row["batch"] for row in timing["rows"]})
    assert batches == conformance["batches"]
    bound_keys = (
        "envs",
        "scope",
        "contact_model",
        "output_mode",
        "source_revision",
        "source_sha256",
        "dt",
        "substeps",
        "warmup_steps",
        "history",
        "precision",
        "level",
        "egg_geometry",
        "collision_method",
        "contact_patch",
        "intrinsic_load_partition",
        "seed",
        "varied",
        "conditions_sha256",
    )
    rows = []
    for group, batch in enumerate(batches):
        folder = args.memory_campaign / f"group{group}"
        assert (folder / "returncode.txt").read_text().strip() == "0"
        receipts.append(file_id(folder / "returncode.txt"))
        for primary in (row for row in timing["rows"] if row["batch"] == batch):
            mode = primary["mode"]
            root = folder / mode
            code = folder / (mode + ".returncode.txt")
            assert code.read_text().strip() == "0"
            receipts.append(file_id(code))
            profile = read(root / "result.json")
            benchmark = read(root / "benchmark.json")
            original = read(args.timing_campaign / f"group{group}" / (mode + ".json"))
            assert profile["passed"] and all(profile["checks"].values()) and profile["primary_timing"] is False
            assert profile["benchmark"] == file_id(root / "benchmark.json")
            assert profile["driver_error"] is None and profile["memory_error"] is None
            assert all(benchmark[k] == original[k] for k in bound_keys)
            assert benchmark["source_revision"] == memory_source["revision"]
            assert len(benchmark["repeats"]) == 1
            assert benchmark["repeats"][0]["steps"] == 1200
            assert benchmark["repeats"][0]["invalid_environment_steps"] == 0
            memory = profile["memory"]
            samples = memory["samples"]
            assert memory["enabled"] and samples
            assert all(r["gpu_uuid"] == benchmark["gpu_uuid"] and r["process_mib"] > 0 for r in samples)
            assert all(b["seconds"] > a["seconds"] for a, b in zip(samples, samples[1:]))
            assert memory["peak_mib"] == max(r["process_mib"] for r in samples)
            rows.append(
                {
                    "batch": batch,
                    "mode": mode,
                    "sampled_process_peak_mib": memory["peak_mib"],
                    "sample_count": len(samples),
                    "process_max_rss_kib": profile["process_max_rss_kib"],
                    "torch_peak_allocated_bytes": benchmark["torch_peak_allocated_bytes"],
                    "native_buffer_bytes": benchmark.get("native_buffer_bytes", 0),
                    "memory_gpu_uuid": benchmark["gpu_uuid"],
                    "primary_gpu_uuid": original["gpu_uuid"],
                    "same_physical_gpu_as_primary": benchmark["gpu_uuid"] == original["gpu_uuid"],
                    "primary_env_scene_steps_per_s": primary["median_env_steps_per_s"],
                }
            )
    with (args.output / "memory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": True,
            "rows": rows,
            "receipts": receipts,
            "conformance_checks": len(conformance["checks"]),
            "source_revision": memory_source["revision"],
            "source_files_rehashed": len(memory_source["files"]),
            "physical_error_qualified": False,
            "whole_goal_complete": False,
            "scope": "Configuration-matched separate timing, complete conformance and fresh-process GPU memory. Sampled NVIDIA process peaks include non-Torch allocations but can miss brief peaks at requested0.1s cadence plus command latency. Profile rates are intrusive diagnostics; displayed throughput is exclusively from the primary timing audit. This repeated-condition bank is not diverse-input capacity or physical accuracy qualification.",
        },
    )


if __name__ == "__main__":
    main()
