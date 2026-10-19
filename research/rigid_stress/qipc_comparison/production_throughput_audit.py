"""Audit fixed-horizon native production measurements without joining unmatched errors."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .artifacts import file_id, runtime_identity, write_json


SCOPES = (
    ("rigid", "rigid", "pressure", "max"),
    ("dpr-max", "live", "pressure", "max"),
    ("dpr-full", "live", "pressure", "full"),
    ("scr-max", "live", "intrinsic", "max"),
    ("scr-full", "live", "intrinsic", "full"),
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    source = read(args.campaign / "source.json")
    assert len(source["files"]) >= 858 and len(source["revision"]) == 40
    source_files = {r["path"]: r for r in source["files"]}
    for row in source["files"]:
        ident = file_id(Path(source["source"]) / row["path"])
        assert ident["sha256"] == row["sha256"] and ident["bytes"] == row["bytes"]
    patch = file_id(args.campaign / "source.patch")
    assert patch["sha256"] == source["patch_sha256"] and patch["bytes"] == 0
    receipts.append(patch)
    tasks = read(args.campaign / "tasks.json")
    assert len(tasks) == len(args.batches) and all(len(g) == 5 for g in tasks)
    assert len(set(args.batches)) == len(args.batches) and min(args.batches) > 0
    rows, comparisons = [], []
    bank_hashes = set()
    for group, batch in enumerate(args.batches):
        folder = args.campaign / f"group{group}"
        code = folder / "returncode.txt"
        assert code.read_text().strip() == "0"
        receipts.append(file_id(code))
        records = []
        for name, scope, model, mode in SCOPES:
            code = folder / (name + ".json.returncode.txt")
            assert code.read_text().strip() == "0"
            receipts.append(file_id(code))
            d = read(folder / (name + ".json"))
            assert (d["envs"], d["scope"], d["contact_model"], d["output_mode"]) == (batch, scope, model, mode)
            assert d["source_revision"] == source["revision"]
            for key, sha in d["source_sha256"].items():
                path = Path(key)
                relative = str(path.relative_to(source["source"])) if path.is_absolute() else str(path)
                assert source_files[relative]["sha256"] == sha, relative
            assert d["dt"] == 0.01 and d["substeps"] == 1 and d["warmup_steps"] == 900
            assert d["history"] == 0 and d["precision"] == "64" and d["level"] == 1
            assert d["egg_geometry"] == "mesh64" and d["collision_method"] == "mpr" and d["contact_patch"] == "off"
            assert d["intrinsic_load_partition"] == "body_source_prior" and d["seed"] == 623016 and d["varied"]
            assert d["trajectory_metrics"] is None and not d["arguments"]["trace"]
            assert d["arguments"]["minimum_seconds"] == 0
            bank = file_id(d["conditions"])
            assert bank["sha256"] == d["conditions_sha256"]
            bank_hashes.add(bank["sha256"])
            if bank not in receipts:
                receipts.append(bank)
            repeats = d["repeats"]
            assert len(repeats) == 3
            rates, latencies = [], []
            for r in repeats:
                assert r["steps"] == 1200 and r["seconds"] > 0 and r["invalid_environment_steps"] == 0
                assert math.isclose(r["transitions_per_second"], batch * 1200 / r["seconds"], rel_tol=1e-12)
                if scope == "live":
                    assert len(r["failed_steps_per_environment"]) == batch and not any(
                        r["failed_steps_per_environment"]
                    )
                rates.append(r["transitions_per_second"])
                latencies.append(1000 * r["seconds"] / 1200)
            row = {
                "batch": batch,
                "mode": name,
                "median_env_steps_per_s": statistics.median(rates),
                "min_env_steps_per_s": min(rates),
                "max_env_steps_per_s": max(rates),
                "median_scene_step_ms": statistics.median(latencies),
                "invalid_environment_steps": 0,
                "timed_environment_steps": batch * 3600,
                "gpu": d["gpu"],
                "gpu_uuid": d["gpu_uuid"],
                "native_buffer_bytes": d.get("native_buffer_bytes", 0),
                "torch_peak_allocated_bytes": d["torch_peak_allocated_bytes"],
                "setup_seconds": d["setup_seconds"],
                "physical_error": "pending for this collider/source/radius configuration",
                "complete_motion_output_conformance": "separate controls pending",
            }
            rows.append(row)
            records.append(d)
        assert len({d["gpu_uuid"] for d in records}) == 1
        assert len({d["conditions_sha256"] for d in records}) == 1
        for d in records[1:]:
            assert [r["resets"] for r in d["repeats"]] == [r["resets"] for r in records[0]["repeats"]]
        batch_rows = rows[-5:]
        for row in batch_rows[1:]:
            comparisons.append(
                {
                    "batch": batch,
                    "mode": row["mode"],
                    "added_scene_step_ms": row["median_scene_step_ms"] - batch_rows[0]["median_scene_step_ms"],
                    "same_gpu_rigid_baseline": True,
                }
            )
    assert len(bank_hashes) == 1
    with (args.output / "production.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "timing_receipts_passed": True,
            "rows": rows,
            "overheads": comparisons,
            "receipts": receipts,
            "source_files_rehashed": len(source["files"]),
            "warm_repeats_not_independent_process_repeats": True,
            "physical_reference_qualified": False,
            "whole_goal_complete": False,
            "scope": "Terminal fixed-horizon scene timing receipts and zero invalid-step counters. Complete separate conformance is not inferred. Native buffer bytes are logical allocations; Torch peak is not total process GPU peak. Different batch groups need not share a physical GPU. No policy throughput or physical-error claim.",
        },
    )


if __name__ == "__main__":
    main()
