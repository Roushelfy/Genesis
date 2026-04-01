"""Dispatcher: run all 12 G1 kimono simulations (6 datasets x 2 variants) sequentially.

For each combination, the dispatcher:
  1. Runs the simulation in no-gui mode (dumps every frame).
  2. Recovers all dumped frames and exports to USD.

Usage::

    uv run python IPC-Samples/python/Dancing/run_all_kimono.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "g1_kimono_batch.py"
REPO_ROOT = Path(__file__).resolve().parents[3]

NUM_DATASETS = 6


def run_cmd(args: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(args)}")
    print(f"{'=' * 70}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(args, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[dispatcher] FAILED: {label} (exit code {result.returncode}, {elapsed:.1f}s)")
        return False

    print(f"\n[dispatcher] OK: {label} ({elapsed:.1f}s)")
    return True


def main() -> None:
    python = sys.executable
    results: list[tuple[str, bool]] = []

    for dataset_idx in range(NUM_DATASETS):
        for no_lower in [False, True]:
            variant = f"d{dataset_idx}_{'no_lower' if no_lower else 'lower'}"

            base_args = [python, str(SCRIPT), "--dataset", str(dataset_idx)]
            if no_lower:
                base_args.append("--no-lower")

            sim_ok = run_cmd(
                base_args + ["--no-gui"],
                f"[sim] {variant}",
            )
            results.append((f"{variant}/sim", sim_ok))

            if not sim_ok:
                print(f"[dispatcher] skipping recover for {variant} due to sim failure")
                results.append((f"{variant}/recover", False))
                continue

            rec_ok = run_cmd(
                base_args + ["--recover"],
                f"[recover+export] {variant}",
            )
            results.append((f"{variant}/recover", rec_ok))

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for label, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:>6s}] {label}")

    failures = sum(1 for _, ok in results if not ok)
    print(f"\n  Total: {len(results)} tasks, {len(results) - failures} passed, {failures} failed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
