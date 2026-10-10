"""Run matched native workloads and record whole-card memory through setup and rollout."""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def sample_memory(destination, done, period):
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    with destination.open("w") as output:
        writer = csv.writer(output)
        writer.writerow(["timestamp", "uuid", "used_MiB", "total_MiB", "utilization_percent", "power_W"])
        while not done.is_set():
            response = subprocess.run(command, text=True, capture_output=True, check=False, timeout=20)
            if response.returncode == 0:
                for row in csv.reader(response.stdout.splitlines()):
                    writer.writerow([item.strip() for item in row])
                output.flush()
            done.wait(period)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, nargs="+", required=True)
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("rigid", "recovery", "live", "policy", "policy-rigid"),
        default=["rigid", "recovery", "live", "policy"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[510000])
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory-period", type=float, default=0.5)
    parser.add_argument("--source-revision", default=os.environ.get("RIGID_STRESS_SOURCE_REVISION", "unrecorded"))
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--save-bank", type=Path)
    args, extra = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"source_revision": args.source_revision, "memory_period_seconds": args.memory_period, "measurements": []}
    for seed in args.seeds:
        for envs in args.envs:
            for scope in args.scopes:
                prefix = args.output / f"{scope}-b{envs}-seed{seed}"
                result_path = prefix.with_suffix(".json")
                command = [
                    sys.executable,
                    "examples/speed_benchmark/rigid_stress.py",
                    "--scope",
                    scope,
                    "--envs",
                    str(envs),
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--warmup",
                    str(args.warmup),
                    "--repetitions",
                    str(args.repetitions),
                    "--minimum-seconds",
                    "10",
                    "--level",
                    "1",
                    "--varied",
                    "--compact-log",
                    "--source-revision",
                    args.source_revision,
                    "--output",
                    str(result_path),
                    *extra,
                ]
                if args.bank is not None:
                    command += ["--conditions", str(args.bank)]
                if args.save_bank is not None:
                    if args.save_bank.exists():
                        command += ["--conditions", str(args.save_bank)]
                    else:
                        command += ["--save-conditions", str(args.save_bank)]
                done = threading.Event()
                memory_path = prefix.with_suffix(".memory.csv")
                sampler = threading.Thread(target=sample_memory, args=(memory_path, done, args.memory_period))
                sampler.start()
                print("Starting", scope, "B", envs, "seed", seed, flush=True)
                started = time.perf_counter()
                try:
                    with prefix.with_suffix(".log").open("w") as log:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                finally:
                    done.set()
                    sampler.join()
                measurement = {
                    "scope": scope,
                    "envs": envs,
                    "seed": seed,
                    "command": command,
                    "returncode": process.returncode,
                    "total_process_seconds": time.perf_counter() - started,
                    "result": str(result_path),
                    "memory_samples": str(memory_path),
                }
                if process.returncode == 0:
                    result = json.loads(result_path.read_text())
                    steps = sum(row["steps"] for row in result["repeats"])
                    seconds = sum(row["seconds"] for row in result["repeats"])
                    invalid = sum(row.get("invalid_environment_steps", 0) for row in result["repeats"])
                    measurement.update(
                        {
                            "repeats": result["repeats"],
                            "valid_env_steps_per_second": (envs * steps - invalid) / seconds,
                            "invalid_environment_steps": invalid,
                            "batch_steps_per_second": steps / seconds,
                            "gpu_uuid": result["gpu_uuid"],
                            "source_sha256": result["source_sha256"],
                            "policy_range": result["policy"],
                            "native_buffer_bytes": result.get("native_buffer_bytes"),
                        }
                    )
                    with memory_path.open() as memory:
                        selected = [
                            float(row["used_MiB"])
                            for row in csv.DictReader(memory)
                            if row["uuid"] == result["gpu_uuid"]
                        ]
                    measurement["sampled_whole_card_peak_MiB"] = max(selected) if selected else None
                    print("Completed", scope, envs, measurement["valid_env_steps_per_second"], flush=True)
                else:
                    print("Failed", scope, envs, "see", prefix.with_suffix(".log"), flush=True)
                summary["measurements"].append(measurement)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                if process.returncode != 0:
                    return


if __name__ == "__main__":
    main()
