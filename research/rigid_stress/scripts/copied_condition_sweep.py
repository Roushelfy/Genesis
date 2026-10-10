"""Run complete native live trajectories while repeating a frozen condition bank."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--scope", choices=("live", "policy", "rigid", "policy-rigid"), default="live")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bank = args.bank if args.bank is not None else args.output / "conditions1024.npz"
    summary = {"scope": args.scope, "condition_bank": str(bank), "measurements": []}
    for n_envs in args.envs:
        result_path = args.output / f"{args.scope}-b{n_envs}.json"
        command = [
            sys.executable,
            "examples/speed_benchmark/rigid_stress.py",
            "--scope",
            args.scope,
            "--envs",
            str(n_envs),
            "--level",
            "1",
            "--varied",
            "--warmup",
            "900",
            "--steps",
            "1200",
            "--repetitions",
            str(args.repetitions),
            "--minimum-seconds",
            "10",
            "--compact-log",
            "--output",
            str(result_path),
        ]
        if bank.exists():
            command += ["--conditions", str(bank)]
        else:
            assert n_envs == 1024, "Generate the 1024-condition bank before changing the batch size."
            command += ["--save-conditions", str(bank)]
        print("Starting", args.scope, "B=", n_envs, flush=True)
        start = time.perf_counter()
        with result_path.with_suffix(".log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        measurement = {
            "envs": n_envs,
            "command": command,
            "returncode": process.returncode,
            "total_process_seconds": time.perf_counter() - start,
            "result": str(result_path),
        }
        if process.returncode == 0:
            result = json.loads(result_path.read_text())
            measurement["transitions_per_second"] = (
                sum(row["steps"] for row in result["repeats"])
                * n_envs
                / sum(row["seconds"] for row in result["repeats"])
            )
            measurement["repeats"] = result["repeats"]
            measurement["stress_bytes"] = result.get("native_buffer_bytes")
            measurement["device_free_bytes"] = result["device_free_bytes"]
            print("Finished", n_envs, measurement["transitions_per_second"], "environment steps/s", flush=True)
        else:
            print("Failed", n_envs, "see", result_path.with_suffix(".log"), flush=True)
        summary["measurements"].append(measurement)
        summary["conditions_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest() if bank.exists() else None
        (args.output / "sweep-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if process.returncode != 0:
            break


if __name__ == "__main__":
    main()
