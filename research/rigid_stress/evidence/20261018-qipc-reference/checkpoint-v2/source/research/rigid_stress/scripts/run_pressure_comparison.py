"""Run paired native stress configurations inside one allocated GPU/container."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=(
            "pressure-default",
            "pressure-fast",
            "pressure-history4",
            "pressure-scatter128",
            "pressure-scatter256",
            "pressure-tuned",
            "pressure-unfused",
            "intrinsic",
        ),
        default=["pressure-default", "intrinsic"],
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=("rigid", "live", "policy-rigid", "policy"),
        default=["rigid", "live", "policy-rigid", "policy"],
    )
    parser.add_argument("--output-mode", choices=("max", "full"), default="max")
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.trace and args.scopes != ["live"]:
        parser.error("Moving trace runs require only --scopes live.")
    if args.memory and args.trace:
        parser.error("Memory and trace diagnostics use separate runs.")
    if args.output.exists():
        parser.error("Output must be a fresh directory.")
    source = args.source.resolve()
    bank = args.conditions.resolve()
    manifest = json.loads((source / "snapshot.json").read_text())
    bank_hash = hashlib.sha256(bank.read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    rows = []
    samplers = args.output / "samplers"
    for scope in args.scopes:
        profiles = args.profiles if scope in ("live", "policy") else ["shared"]
        for profile in profiles:
            name = profile + "-" + scope
            run = args.output / name
            flags = [
                "--scope",
                scope,
                "--envs",
                str(args.envs),
                "--level",
                "1",
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--repetitions",
                str(args.repetitions),
                "--minimum-seconds",
                "0",
                "--varied",
                "--seed",
                "623016",
                "--output-mode",
                args.output_mode,
                "--egg-geometry",
                "mesh64",
                "--collision-method",
                "mpr",
                "--contact-patch",
                "off",
                "--conditions",
                str(bank),
                "--source-revision",
                manifest["base_revision"],
                "--compact-log",
            ]
            if profile == "intrinsic":
                flags += [
                    "--contact-model",
                    "intrinsic",
                    "--intrinsic-load-partition",
                    "body_source_prior",
                    "--history",
                    "0",
                ]
            elif profile in ("pressure-scatter128", "pressure-scatter256"):
                flags += ["--scatter-tasks-per-env", "128" if profile == "pressure-scatter128" else "256"]
            elif profile in ("pressure-fast", "pressure-tuned"):
                flags += [
                    "--contact-wrench-reuse",
                    "on",
                    "--contact-moment-reuse",
                    "on",
                    "--packed-block-size",
                    "512",
                    "--coalesced-residual",
                    "on",
                ]
                if profile == "pressure-tuned":
                    flags += ["--scatter-tasks-per-env", "128"]
            elif profile == "pressure-unfused":
                flags += ["--unfused-pipeline", "--rigid-post-fusion", "off"]
            elif profile == "pressure-history4":
                flags += ["--history", "4"]
            elif profile == "pressure-default" and args.output_mode == "max":
                flags += ["--public-default-stress"]
            if args.trace:
                flags += ["--trace", "--trace-steps", "600"]
            command = [
                "bash",
                str(source / "research/rigid_stress/scripts/run_practical_native_benchmark.sh"),
                str(run),
                *flags,
            ]
            sampler = None
            sampler_stdout = sampler_stderr = None
            if args.memory:
                samplers.mkdir(exist_ok=True)
                sampler_stdout = (samplers / (name + ".csv")).open("wb")
                sampler_stderr = (samplers / (name + ".stderr")).open("wb")
                sampler = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory",
                        "--format=csv",
                        "--loop-ms=100",
                    ],
                    stdout=sampler_stdout,
                    stderr=sampler_stderr,
                )
            start = time.monotonic()
            print(json.dumps({"case": name, "command": command}), flush=True)
            result = subprocess.run(command, check=False)
            sampler_live = None
            if sampler is not None:
                sampler_live = sampler.poll() is None
                if sampler_live:
                    sampler.terminate()
                sampler.wait()
                sampler_stdout.close()
                sampler_stderr.close()
            rows.append(
                {
                    "case": name,
                    "profile": profile,
                    "scope": scope,
                    "command": command,
                    "returncode": result.returncode,
                    "wall_seconds": time.monotonic() - start,
                    "run": str(run),
                    "intrusive": args.memory or args.trace,
                    "memory_sampler_alive_at_end": sampler_live,
                }
            )
            assert hashlib.sha256(bank.read_bytes()).hexdigest() == bank_hash
            summary = {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "source_manifest_sha256": hashlib.sha256((source / "snapshot-files.sha256").read_bytes()).hexdigest(),
                "conditions_sha256": bank_hash,
                "cases": rows,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "all_requested_cases_finished": False,
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["all_requested_cases_finished"] = True
    summary["all_program_exits_zero"] = all(
        row["returncode"] == 0 and row["memory_sampler_alive_at_end"] is not False for row in rows
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["all_program_exits_zero"] else 1


if __name__ == "__main__":
    sys.exit(main())
