"""Validate and summarize paired pressure/intrinsic benchmark metadata using stdlib."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def sampled_memory(path, gpu_uuid):
    totals = defaultdict(float)
    unknown = 0
    with path.open() as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        for raw in reader:
            row = {key.strip(): value.strip() for key, value in raw.items() if key is not None and value is not None}
            if row.get("gpu_uuid") != gpu_uuid:
                continue
            value = row["used_gpu_memory [MiB]"].split()[0]
            if value == "[N/A]":
                unknown += 1
                continue
            totals[row["timestamp"]] += float(value)
    if not totals:
        return {"samples": 0, "unknown": unknown, "sampled_peak_GiB": None}
    peak = max(totals, key=totals.get)
    return {
        "samples": len(totals),
        "unknown": unknown,
        "sampled_peak_GiB": totals[peak] / 1024,
        "peak_timestamp": peak,
        "interval_ms": 100,
        "sampled_lower_bound": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    contract = json.loads((args.root / "meta/contract-v1.json").read_text())
    snapshot = json.loads((args.root / "source-v1/snapshot.json").read_text())
    source_hashes = {row["path"]: row["sha256"] for row in snapshot["files"]}
    public = json.loads((args.root / "preflight-b32-v1/pressure-default-live/results/benchmark.json").read_text())[
        "stress_configurations"
    ][0]
    public = {key: value for key, value in public.items() if key != "mesh"}
    fixed = contract["fixed_physical"]
    records, failures, pending = [], [], []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if "source_manifest_sha256" not in summary:
            continue
        assert summary["source_manifest_sha256"] == contract["benchmark_manifest_sha256"], summary_path
        assert summary["conditions_sha256"] == contract["condition_bank_sha256"], summary_path
        requested = summary["arguments"]
        intrusive = requested["memory"] or requested["trace"]
        if not summary["all_requested_cases_finished"]:
            pending.append(str(summary_path.parent))
        for case in summary["cases"]:
            run = Path(case["run"])
            assert case["intrusive"] == intrusive
            if case["returncode"] != 0:
                log = run / "benchmark.log"
                tail = log.read_text(errors="replace")[-5000:] if log.exists() else "Benchmark did not start."
                before = run / "snapshot-before-verification.txt"
                after = run / "snapshot-after-verification.txt"
                assert before.read_bytes() == after.read_bytes()
                assert json.loads((run / "snapshot-inventory-after.json").read_text())["passed"]
                hardware = run / "hardware.csv"
                sampler = summary_path.parent / "samplers" / (case["case"] + ".csv")
                failures.append(
                    {
                        "cohort": summary_path.parent.name,
                        "case": case["case"],
                        "returncode": case["returncode"],
                        "envs": requested["envs"],
                        "intrusive": intrusive,
                        "log_tail": tail,
                        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                        "hardware_csv": hardware.read_text(),
                        "hardware_sha256": hashlib.sha256(hardware.read_bytes()).hexdigest(),
                        "sampler_sha256": hashlib.sha256(sampler.read_bytes()).hexdigest()
                        if sampler.exists()
                        else None,
                        "source_before_after_unchanged": True,
                        "snapshot_inventory_passed": True,
                        "source_revision": snapshot["base_revision"],
                        "classification": "out_of_memory"
                        if "out of memory" in tail.lower() or "OUT_OF_MEMORY" in tail
                        else "rigid_array_capacity_guard"
                        if "Jacobian shape" in tail and "too large" in tail
                        else "native_load_rejected"
                        if "Native stress contact recovery rejected a load" in tail
                        else "program_failure",
                        "successful_rollout_rate": None,
                    }
                )
                continue
            assert (run / "returncode.txt").read_text().strip() == "0"
            assert (run / "program-returncode.txt").read_text().strip() == "0"
            assert (run / "snapshot-before-verification.txt").read_bytes() == (
                run / "snapshot-after-verification.txt"
            ).read_bytes()
            assert json.loads((run / "snapshot-inventory-after.json").read_text())["passed"]
            assert (
                hashlib.sha256((run / "snapshot-files.sha256").read_bytes()).hexdigest()
                == contract["benchmark_manifest_sha256"]
            )
            path = run / "results/benchmark.json"
            data = json.loads(path.read_text())
            assert data["conditions_sha256"] == contract["condition_bank_sha256"] and data["condition_count"] == 8
            assert data["envs"] == requested["envs"] and data["warmup_steps"] == requested["warmup"]
            assert (
                data["precision"] == "64" and data["quadrature"] == 10 and data["dt"] == 0.01 and data["substeps"] == 1
            )
            assert (
                data["collision_method"] == "mpr"
                and data["contact_patch"] == "off"
                and data["egg_geometry"] == "mesh64"
            )
            assert data["output_mode"] == requested["output_mode"] and data["varied"] and data["seed"] == 623016
            assert data["scope"] == case["scope"]
            assert data["arguments"]["trace"] == requested["trace"] and not data["arguments"]["trajectory_metrics"]
            assert data["policy"]["warmup_reference_max_error_rad"] <= 1e-9
            source_root = Path(requested["source"]).resolve()
            for source, digest in data["source_sha256"].items():
                relative = Path(source)
                if relative.is_absolute():
                    relative = relative.relative_to(source_root)
                assert digest == source_hashes[relative.as_posix()], source
            profile = case["profile"]
            if profile != "shared":
                expected = {**public, "output_mode": requested["output_mode"]}
                if profile == "intrinsic":
                    expected.update(contact_model="intrinsic", intrinsic_load_partition="body_source_prior")
                elif profile == "pressure-unfused":
                    expected.update(fused_pipeline=False, fused_rigid_post=False)
                elif profile == "pressure-history4":
                    expected["history_size"] = 4
                elif profile == "pressure-scatter128":
                    expected["scatter_tasks_per_env"] = 128
                elif profile == "pressure-scatter256":
                    expected["scatter_tasks_per_env"] = 256
                elif profile in ("pressure-fast", "pressure-tuned"):
                    expected.update(
                        contact_wrench_reuse=True,
                        contact_moment_reuse=True,
                        packed_block_size=512,
                        coalesced_residual=True,
                    )
                    if profile == "pressure-tuned":
                        expected["scatter_tasks_per_env"] = 128
                actual = {key: value for key, value in data["stress_configurations"][0].items() if key != "mesh"}
                assert actual == expected, {
                    "case": str(path),
                    "different": {
                        key: (actual.get(key), expected.get(key))
                        for key in actual.keys() | expected.keys()
                        if actual.get(key) != expected.get(key)
                    },
                }
                assert data["tetrahedra"] == fixed["tetrahedra"] and data["dofs"] == fixed["dof"]
                if profile == "pressure-default" and requested["output_mode"] == "max":
                    assert data["public_default_stress"]
            repeats = data["repeats"]
            assert len(repeats) == requested["repetitions"]
            good = True
            rates, latency = [], []
            for row in repeats:
                assert row["steps"] == requested["steps"]
                attempted = data["envs"] * row["steps"]
                valid = attempted - row["invalid_environment_steps"]
                assert (
                    abs(valid / row["seconds"] - row["transitions_per_second"]) <= 1e-10 * row["transitions_per_second"]
                )
                if row["failed_steps_per_environment"] is not None:
                    assert len(row["failed_steps_per_environment"]) == data["envs"]
                    assert sum(row["failed_steps_per_environment"]) == row["invalid_environment_steps"]
                good = good and row["invalid_environment_steps"] == 0
                rates.append(row["transitions_per_second"])
                latency.append(1000 * row["seconds"] / row["steps"])
            row = {
                "cohort": summary_path.parent.name,
                "case": case["case"],
                "profile": profile,
                "scope": case["scope"],
                "envs": data["envs"],
                "output_mode": data["output_mode"],
                "intrusive": intrusive,
                "complete_valid_rollout": good,
                "headline_eligible": good
                and not intrusive
                and requested["repetitions"] == 3
                and requested["steps"] == 2400
                and requested["warmup"] == 900,
                "raw_rates": rates,
                "raw_step_ms": latency,
                "median_valid_envstep_s": statistics.median(rates) if good else None,
                "median_step_ms": statistics.median(latency) if good else None,
                "minimum_rate": min(rates),
                "maximum_rate": max(rates),
                "scatter_overflow_calls": [x["scatter_overflow_calls"] for x in repeats],
                "partial_reset_counts": [x["resets"] for x in repeats],
                "host": data["host"],
                "slurm_job_id": data["slurm_job_id"],
                "gpu": data["gpu"],
                "gpu_uuid": data["gpu_uuid"],
                "gpu_total_bytes": data["gpu_total_bytes"],
                "source_revision": data["source_revision"],
                "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_path": str(path),
                "raw_repeats": [
                    {key: value for key, value in repeat.items() if key != "failed_steps_per_environment"}
                    for repeat in repeats
                ],
                "public_default_stress": data["public_default_stress"],
                "native_load_model": data["load_model"] if profile != "shared" else "disabled",
                "stress_precision": data["precision"] if profile != "shared" else None,
                "stress_options": [
                    {key: value for key, value in options.items() if key != "mesh"}
                    for options in data.get("stress_configurations", [])
                ],
                "packages": json.loads((run / "environment.json").read_text())["packages"],
                "policy": data["policy"],
                "native_allocation_count": data.get("native_allocation_count"),
                "native_logical_buffer_bytes": data.get("native_logical_buffer_bytes"),
                "setup_seconds_including_build_JIT": data["setup_seconds"],
                "native_buffer_bytes": data.get("native_buffer_bytes"),
                "requested_effective": {
                    key: data.get(key)
                    for key in (
                        "fused_pipeline",
                        "surface_inverse",
                        "packed_surface_loads",
                        "cooperative_pressure",
                        "cooperative_scatter",
                        "cooperative_balance",
                        "cooperative_solve",
                        "history",
                        "method",
                        "method_selected",
                        "inverse_precision",
                        "rigid_post_fusion",
                        "effective_rigid_post_fusion",
                        "contact_wrench_reuse",
                        "effective_contact_wrench_reuse",
                        "contact_moment_reuse",
                        "effective_contact_moment_reuse",
                        "packed_block_size",
                        "effective_packed_block_sizes",
                        "coalesced_residual",
                        "scatter_tasks_per_env",
                        "effective_contact_block_size",
                    )
                },
            }
            if requested["memory"]:
                assert case["memory_sampler_alive_at_end"]
                memory = summary_path.parent / "samplers" / (case["case"] + ".csv")
                row["memory"] = sampled_memory(memory, data["gpu_uuid"])
                row["memory"]["raw_sha256"] = hashlib.sha256(memory.read_bytes()).hexdigest()
            records.append(row)
    for row in records:
        if row["scope"] not in ("live", "policy") or not row["complete_valid_rollout"]:
            continue
        scope = "rigid" if row["scope"] == "live" else "policy-rigid"
        baseline = [
            x for x in records if x["cohort"] == row["cohort"] and x["scope"] == scope and x["complete_valid_rollout"]
        ]
        if baseline:
            base = baseline[0]
            assert base["gpu_uuid"] == row["gpu_uuid"] and base["envs"] == row["envs"]
            ratio = row["median_step_ms"] / base["median_step_ms"]
            row.update(
                matched_baseline_step_ms=base["median_step_ms"],
                incremental_stress_step_ms=row["median_step_ms"] - base["median_step_ms"],
                step_time_ratio=ratio,
                throughput_loss_fraction=1 - 1 / ratio,
            )
    fields = [
        "cohort",
        "case",
        "profile",
        "scope",
        "envs",
        "output_mode",
        "intrusive",
        "headline_eligible",
        "median_valid_envstep_s",
        "median_step_ms",
        "matched_baseline_step_ms",
        "incremental_stress_step_ms",
        "step_time_ratio",
        "throughput_loss_fraction",
        "minimum_rate",
        "maximum_rate",
        "gpu_uuid",
        "raw_sha256",
        "raw_path",
    ]
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    result = {
        "scope": __doc__,
        "contract_sha256": hashlib.sha256((args.root / "meta/contract-v1.json").read_bytes()).hexdigest(),
        "public_default_options": public,
        "records": records,
        "failures": failures,
        "cohorts_still_incomplete": pending,
        "definitions": {
            "overflow_calls": "whole scatter passes using complete canonical fallback, not discarded environments",
            "step_time_ratio": "median stress step time divided by same-allocation matching rigid baseline",
            "throughput_loss_fraction": "1 - reciprocal(step_time_ratio)",
            "memory": "separate 100ms sampled process peak, GiB; sampled lower bound including setup",
        },
        "passed_metadata_checks": True,
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified_records": len(records), "failures": len(failures), "incomplete_cohorts": len(pending)}))


if __name__ == "__main__":
    main()
