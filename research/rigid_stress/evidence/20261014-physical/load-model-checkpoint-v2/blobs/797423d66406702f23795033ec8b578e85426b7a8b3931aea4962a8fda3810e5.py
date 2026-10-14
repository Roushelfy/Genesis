"""Run the frozen configuration with complete scoped benchmarks and explicit data destinations."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Complete recorded contact input for the stress scope")
    parser.add_argument("--seed", type=int, default=610000)
    parser.add_argument("--scope", choices=("all", "rigid", "policy-rigid", "stress", "live", "policy"), default="all")
    args = parser.parse_args()
    scopes = ("rigid", "policy-rigid", "stress", "live", "policy") if args.scope == "all" else (args.scope,)
    if "stress" in scopes and (args.replay is None or not args.replay.is_file()):
        parser.error("The stress scope requires an existing --replay file")
    config = json.loads(args.config.read_text())
    common = config["benchmark_arguments"]
    if not isinstance(common, list) or not all(isinstance(argument, str) for argument in common):
        raise ValueError("Frozen benchmark_arguments must be a list of literal CLI arguments")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "passed": False,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "seed": args.seed,
        "commands": [],
    }
    manifest_path = args.output / f"suite-{args.scope}.json"
    for scope in scopes:
        module = "benchmark_replay" if scope == "stress" else "benchmark_live"
        artifact = args.output / scope / "result.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", f"research.rigid_stress.{module}", *common, "--scope", scope]
        if scope == "stress":
            command.extend(("--replay", str(args.replay)))
        else:
            command.extend(("--seed", str(args.seed)))
        command.extend(("--output", str(artifact)))
        log = artifact.with_suffix(".log")
        manifest["commands"].append({"scope": scope, "command": command, "log": str(log), "return_code": None})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["commands"][-1]), flush=True)
        with log.open("w") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        manifest["commands"][-1]["return_code"] = process.returncode
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"Benchmark {scope} rejected; inspect {log}")
    manifest["passed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passed": True, "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
