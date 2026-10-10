"""Repeat copied-condition candidates and audit the highest measured live rate."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_sweep(root: Path, bank: Path, name: str, scope: str, points: list[int]) -> list[dict]:
    command = [
        sys.executable,
        "research/rigid_stress/scripts/copied_condition_sweep.py",
        "--output",
        str(root / name),
        "--bank",
        str(bank),
        "--scope",
        scope,
        "--repetitions",
        "3",
        "--envs",
        *map(str, points),
    ]
    subprocess.run(command, check=True)
    measurements = json.loads((root / name / "sweep-summary.json").read_text())["measurements"]
    assert all(row["returncode"] == 0 for row in measurements)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    bank = args.run / "discovery/conditions1024.npz"
    region = run_sweep(args.run, bank, "final-region", "live", [1024, 16384, 24576, 32768])
    refined_path = args.run / "refinement/sweep-summary.json"
    refined = []
    while len(refined) < 3:
        if refined_path.exists():
            refined = json.loads(refined_path.read_text())["measurements"]
        if len(refined) < 3:
            time.sleep(10)
    baseline = next(row for row in region if row["envs"] == 24576)["transitions_per_second"]
    anchor = next(row for row in refined if row["envs"] == 24576)["transitions_per_second"]
    best = max(row["transitions_per_second"] for row in region)
    candidates = [
        row["envs"]
        for row in refined
        if row["envs"] > 32768
        and row["returncode"] == 0
        and row["transitions_per_second"] * baseline / anchor >= 0.97 * best
    ]
    if candidates:
        region += run_sweep(args.run, bank, "final-boundary", "live", candidates)
    selected = max(region, key=lambda row: row["transitions_per_second"])
    result = {"live_candidates": region, "selected": selected}
    (args.run / "final-selection.json").write_text(json.dumps(result, indent=2) + "\n")
    result["rigid"] = run_sweep(args.run, bank, "final-rigid", "rigid", [selected["envs"]])[0]
    audit = args.run / "final-validity.json"
    command = [
        sys.executable,
        "-m",
        "research.rigid_stress.native_validity_probe",
        "--scope",
        "live",
        "--envs",
        str(selected["envs"]),
        "--conditions",
        str(bank),
        "--warmup",
        "900",
        "--steps",
        "1200",
        "--output",
        str(audit),
    ]
    with audit.with_suffix(".log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result["audit_command"] = command
    result["audit"] = json.loads(audit.read_text())
    assert result["audit"]["failed_environment_steps"] == 0
    (args.run / "final-selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Final valid live peak:", selected["envs"], selected["transitions_per_second"], flush=True)


if __name__ == "__main__":
    main()
