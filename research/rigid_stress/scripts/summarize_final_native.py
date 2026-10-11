"""Collect completed final workloads, output costs, profiles and episode audits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from research.rigid_stress.scripts.summarize_native_audits import columns

BATCHES = (1024, 2048, 32768, 49152, 61440)
SEEDS = (510000, 623001)
SCOPES = ("rigid", "recovery", "live", "policy")


def read(path):
    payload = path.read_bytes()
    return json.loads(payload), {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def measurement(directory, scope, batch, seed):
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary, _ = read(summary_path)
    matches = [
        row for row in summary["measurements"] if (row["scope"], row["envs"], row["seed"]) == (scope, batch, seed)
    ]
    if not matches or matches[0]["status"] != "completed":
        return None
    run = matches[0]
    source, provenance = read(Path(run["result"]))
    repeats = source["repeats"]
    steps = sum(row["steps"] for row in repeats)
    seconds = sum(row["seconds"] for row in repeats)
    invalid = sum(row["invalid_environment_steps"] for row in repeats)
    rates = [row["transitions_per_second"] for row in repeats]
    memory_path = Path(run["memory_samples"])
    with memory_path.open() as handle:
        memory = [float(row["used_MiB"]) for row in csv.DictReader(handle) if row["uuid"] == source["gpu_uuid"]]
    return {
        **provenance,
        "scope": scope,
        "envs": batch,
        "seed": seed,
        "output_mode": source["output_mode"],
        "source_revision": source["source_revision"],
        "source_sha256": source["source_sha256"],
        "gpu": source["gpu"],
        "gpu_uuid": source["gpu_uuid"],
        "steps": steps,
        "seconds": seconds,
        "invalid_environment_steps": invalid,
        "attempted_environment_steps": batch * steps,
        "valid_env_steps_per_second": (batch * steps - invalid) / seconds,
        "batch_steps_per_second": steps / seconds,
        "repeat_valid_rates": rates,
        "repeat_std_valid_rate": float(np.std(rates, ddof=1)),
        "repeat_min_max_valid_rate": [min(rates), max(rates)],
        "resets_per_repeat": [row["resets"] for row in repeats],
        "scatter_overflow_calls_per_repeat": [row["scatter_overflow_calls"] for row in repeats],
        "reset_index_cache_at_end": source.get("reset_index_cache_at_end"),
        "native_buffer_bytes": source.get("native_buffer_bytes"),
        "saved_reset_state_bytes": source.get("saved_reset_state_bytes"),
        "torch_peak_allocated_bytes": source.get("torch_peak_allocated_bytes"),
        "gpu_total_bytes": source["gpu_total_bytes"],
        "device_free_bytes_at_end": source["device_free_bytes"],
        "sampled_whole_card_peak_MiB": max(memory) if memory else None,
        "memory_samples_path": str(memory_path),
        "memory_samples_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
        "policy": source["policy"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = {"summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    measurements, missing = [], []
    for batch in BATCHES:
        for seed in SEEDS:
            for scope in SCOPES:
                directory = args.run / f"final-native-v8-b{batch}-seed{seed}"
                row = measurement(directory, scope, batch, seed)
                if row is None:
                    missing.append({"envs": batch, "seed": seed, "scope": scope})
                else:
                    measurements.append(row)
    result["measurements"] = measurements
    result["missing_measurements"] = missing
    aggregate = []
    for batch in BATCHES:
        for scope in SCOPES:
            rows = [row for row in measurements if row["envs"] == batch and row["scope"] == scope]
            if len(rows) != len(SEEDS):
                continue
            seconds = sum(row["seconds"] for row in rows)
            steps = sum(row["steps"] for row in rows)
            invalid = sum(row["invalid_environment_steps"] for row in rows)
            rates = [rate for row in rows for rate in row["repeat_valid_rates"]]
            entry = {
                "envs": batch,
                "scope": scope,
                "valid_env_steps_per_second": (batch * steps - invalid) / seconds,
                "batch_steps_per_second": steps / seconds,
                "invalid_environment_steps": invalid,
                "repeat_min_max_valid_rate": [min(rates), max(rates)],
                "repeat_std_valid_rate": float(np.std(rates, ddof=1)),
                "sampled_whole_card_peak_MiB": max(row["sampled_whole_card_peak_MiB"] for row in rows),
            }
            aggregate.append(entry)
            print(batch, scope, json.dumps(entry), flush=True)
    result["two_seed_aggregates"] = aggregate
    result["same_seed_slowdown"] = []
    for row in measurements:
        if row["scope"] not in ("live", "policy"):
            continue
        rigid = next(
            (
                base
                for base in measurements
                if (base["envs"], base["seed"], base["scope"]) == (row["envs"], row["seed"], "rigid")
            ),
            None,
        )
        if rigid is not None:
            result["same_seed_slowdown"].append(
                {
                    "envs": row["envs"],
                    "seed": row["seed"],
                    "scope": row["scope"],
                    "step_time_increase_percent": 100
                    * (rigid["valid_env_steps_per_second"] / row["valid_env_steps_per_second"] - 1),
                    "throughput_decrease_percent": 100
                    * (1 - row["valid_env_steps_per_second"] / rigid["valid_env_steps_per_second"]),
                }
            )
    fullcost, missing_fullcost = [], []
    for batch in (1024, 32768):
        for seed in SEEDS:
            for scope in ("live", "policy"):
                paired = {}
                for mode in ("max", "full"):
                    directory = args.run / f"final-native-fullcost-v8-{scope}-{mode}-b{batch}-seed{seed}"
                    paired[mode] = measurement(directory, scope, batch, seed)
                    if paired[mode] is None:
                        missing_fullcost.append({"envs": batch, "seed": seed, "scope": scope, "mode": mode})
                if not all(paired.values()):
                    continue
                assert paired["max"]["gpu_uuid"] == paired["full"]["gpu_uuid"]
                fullcost.append(
                    {
                        "envs": batch,
                        "seed": seed,
                        "scope": scope,
                        "measurements": paired,
                        "step_time_increase_percent": 100
                        * (
                            paired["max"]["valid_env_steps_per_second"] / paired["full"]["valid_env_steps_per_second"]
                            - 1
                        ),
                        "additional_native_bytes": paired["full"]["native_buffer_bytes"]
                        - paired["max"]["native_buffer_bytes"],
                    }
                )
    result["matched_output_cost"] = fullcost
    result["missing_output_cost"] = missing_fullcost
    audits, missing_audits = [], []
    for batch in (1024, 2048, 32768, 61440):
        for seed in SEEDS:
            for scope in ("live", "policy"):
                path = args.run / f"native-final-episode-{scope}-b{batch}-seed{seed}-v8.json"
                if not path.exists():
                    missing_audits.append(str(path))
                    continue
                source, provenance = read(path)
                audits.append(
                    {
                        **provenance,
                        "envs": batch,
                        "seed": seed,
                        "scope": scope,
                        "source_sha256": source["source_sha256"],
                        "failed_environment_steps": source["failed_environment_steps"],
                        "outcomes": dict(zip(source["outcome_columns"], source["trajectory_only_totals"])),
                        "unfinished_episodes_at_end": sum(source["stages"][-1]["unfinished_episodes"]),
                        "work": columns(source["per_environment_work"], source["counter_columns"]),
                        "additional_work": columns(
                            source["additional_per_environment_work"], source["additional_work_columns"]
                        ),
                        "errors": columns(source["per_environment_error_maxima"], source["error_columns"]),
                        "actual_mu_range": source["actual_mu_range"],
                        "criteria": source["criteria"],
                        "tail_cost_scope": source["tail_cost_scope"],
                        "additional_work_scope": source["additional_work_scope"],
                    }
                )
    result["episode_audits"] = audits
    result["missing_episode_audits"] = missing_audits
    profiles, missing_profiles = [], []
    for batch in (1024, 32768):
        paths = [args.run / f"native-final-snapshot-profile-b{batch}-v8.json"]
        paths += [
            args.run / f"native-final-rollout-profile-{scope}-{mode}-b{batch}-v8.json"
            for scope in ("live", "policy")
            for mode in ("max", "full")
        ]
        for path in paths:
            if not path.exists():
                missing_profiles.append(str(path))
                continue
            source, provenance = read(path)
            keys = (
                ("stage_wall_ms", "pressure_subdivision_wall_ms", "profiler_note")
                if "stage_wall_ms" in source
                else ("summary", "invalid_environment_steps", "scope_note")
            )
            profiles.append(
                {**provenance, **{key: source[key] for key in keys}, "source_sha256": source["source_sha256"]}
            )
    result["profiles"] = profiles
    result["missing_profiles"] = missing_profiles
    result["complete"] = not any((missing, missing_fullcost, missing_audits, missing_profiles))
    result["scope_notes"] = [
        "Recovery repeats a frozen actual pre-integration snapshot; live and policy contain changing trajectories and timed public resets.",
        "Policy is FP32 inference plus observation and rollout, not full RL training.",
        "No invalid transition is removed from time; valid rates subtract invalid counts from attempted transitions.",
        "Memory CSV is sampled whole-card memory for the selected UUID, including setup/JIT; native buffers and Torch allocations are separate inventories.",
        "Per-environment work is not per-environment GPU latency. Intrusive nested profile intervals are not additive or throughput measurements.",
        "Two-seed aggregate is summed valid transitions / summed wall time; six-repeat spread combines temporal and workload-seed variation.",
    ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Complete",
        result["complete"],
        "missing",
        len(missing),
        len(missing_fullcost),
        len(missing_audits),
        len(missing_profiles),
        flush=True,
    )
    if args.require_complete and not result["complete"]:
        raise SystemExit("Final evidence is incomplete; see explicit missing lists.")


if __name__ == "__main__":
    main()
