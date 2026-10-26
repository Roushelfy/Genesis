"""Run the reconstructed CPU atlas experiments sequentially with preserved logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    arguments: tuple[str, ...]


def stages(output: Path, scope: str) -> list[Stage]:
    prefix = "research.rigid_stress.atlas_study.cpu_repro."
    numerical = [
        Stage("geometry", "geometry_check", ("--output", str(output / "core_checks.json"))),
        Stage("selection", "selection_study", ("--output", str(output / "selection"))),
        Stage("boundary", "boundary_study", ("--output", str(output / "boundary"))),
        Stage("response", "response_study", ("--output", str(output / "response"))),
        Stage("audit", "audit", ("--output", str(output / "audit.json"))),
        Stage("pipeline_accuracy", "benchmark", (
            "--output", str(output / "pipeline_numerics.json"), "--orders", "4", "8",
            "--batches", "16", "--include-response", "--accuracy", "--numeric-only",
        )),
        Stage("history_accuracy", "history_study", (
            "--output", str(output / "history_results.json"), "--batches", "16", "64", "--frames", "80",
        )),
        Stage("history_operator_controls", "history_controls", (
            "--output", str(output / "history_operator_controls.json"),
        )),
        Stage("history_speed_sensitivity", "history_speed_study", (
            "--output", str(output / "history_speed_results.json"),
            "--cases", str(output / "history_speed_cases.jsonl"),
            "--main-results", str(output / "history_results.json"),
        )),
        Stage("history_slow_accuracy", "history_slow_study", (
            "--output", str(output / "history_slow_results.json"),
            "--speed-results", str(output / "history_speed_results.json"),
        )),
        Stage("condensation_accuracy", "condensation_study", ("--output", str(output / "condensation"))),
        Stage("stress_maps", "plot_stress", ("--output", str(output / "stress_maps"))),
    ]
    timing = [
        Stage("static_timing", "benchmark", (
            "--output", str(output / "benchmark.json"), "--orders", "4", "8", "--batches", "1", "16", "64",
            "--repetitions", "5", "--include-response", "--independent-control",
        )),
        Stage("history_timing", "history_benchmark", (
            "--output", str(output / "history_timings.json"), "--batches", "16", "64", "--frames", "80",
            "--repeats", "3",
        )),
        Stage("history_slow_timing", "history_benchmark", (
            "--output", str(output / "history_slow_timings.json"),
            "--batches", "64", "--frames", "80", "--order", "8", "--speed", "0.25",
            "--scopes", "complete", "--repeats", "3",
        )),
        Stage("condensation_timing", "condensation_study", (
            "--output", str(output / "condensation"), "--timing", "--repeats", "5",
        )),
    ]
    selected = numerical if scope == "numerics" else timing if scope == "timing" else numerical + timing
    return [Stage(stage.name, prefix + stage.module, stage.arguments) for stage in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("numerics", "timing", "all"), default="numerics")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory to preserve earlier evidence.")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    repository = Path(__file__).resolve().parents[3]
    receipts = []
    for stage in stages(output, args.scope):
        command = [sys.executable, "-m", stage.module, *stage.arguments]
        log = output / (stage.name + ".log")
        start = perf_counter()
        with log.open("w") as stream:
            completed = subprocess.run(command, cwd=repository, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        receipts.append({"name": stage.name, "command": command, "returncode": completed.returncode,
                         "wall_seconds": perf_counter() - start, "log": str(log)})
        (output / ("receipts_" + args.scope + ".json")).write_text(json.dumps(receipts, indent=2) + "\n")
        print(stage.name, completed.returncode, flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
