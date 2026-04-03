"""Dispatcher: settle cloth + export USD for completed kimono simulations.

For each variant, runs --settle 100 (recover last frame, run 100 extra frames
to let cloth reach equilibrium), then exports USD.

Usage::

    uv run python DemoAssets/costume/scripts/run_all_kimono.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "g1_kimono_batch.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_BASE = REPO_ROOT / "DemoAssets" / "costume" / "output" / "g1_kimono_batch"

SETTLE_FRAMES = 100
STALL_MINUTES = 20
POLL_SECONDS = 30
GPU_MEM_THRESHOLD = 0.90

TASKS: list[tuple[int, bool]] = [
    (0, True),
    (2, True),
    (4, True),
    (5, True),
    (1, True),  # no_lower
    (0, False),
    (2, False),
    (4, False),
    (5, False),  # lower (d1_lower abandoned)
]


def variant_name(dataset_idx: int, no_lower: bool) -> str:
    return f"d{dataset_idx}_{'no_lower' if no_lower else 'lower'}"


def dump_dir(vname: str) -> Path:
    return OUTPUT_BASE / vname / "dump" / "common" / "sim_engine.cpp"


def max_dump_frame(vname: str) -> int:
    d = dump_dir(vname)
    if not d.exists():
        return 0
    best = 0
    for p in d.glob("state.*.json"):
        try:
            n = int(p.stem.split(".")[1])
            best = max(best, n)
        except (IndexError, ValueError):
            pass
    return best


def gpu_memory_usage() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        parts = out.strip().split(",")
        if len(parts) >= 2:
            used, total = float(parts[0]), float(parts[1])
            return used / total if total > 0 else 0.0
    except Exception:
        pass
    return 0.0


def gpu_is_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return "python" in out.lower()
    except Exception:
        return False


def wait_gpu_free(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not gpu_is_busy():
            return True
        time.sleep(5)
    return False


def run_settle(dataset_idx: int, no_lower: bool) -> bool:
    """Run --settle N with stall/GPU monitoring. Returns success."""
    vname = variant_name(dataset_idx, no_lower)
    python = sys.executable
    args = [python, str(SCRIPT), "--dataset", str(dataset_idx), "--settle", str(SETTLE_FRAMES)]
    if no_lower:
        args.append("--no-lower")

    before = max_dump_frame(vname)
    expected_after = before + SETTLE_FRAMES
    print(f"\n{'=' * 70}")
    print(f"  [settle] {vname}  (frames {before} -> {expected_after})")
    print(f"  cmd: {' '.join(args)}")
    print(f"{'=' * 70}\n", flush=True)

    proc = subprocess.Popen(args, cwd=str(REPO_ROOT))
    last_dump = before
    last_progress = time.time()
    t0 = time.time()

    while proc.poll() is None:
        time.sleep(POLL_SECONDS)
        cur = max_dump_frame(vname)
        elapsed = time.time() - t0
        mem_frac = gpu_memory_usage()

        if mem_frac >= GPU_MEM_THRESHOLD:
            print(f"  [{vname}] GPU {mem_frac:.0%} >= {GPU_MEM_THRESHOLD:.0%}, killing ...", flush=True)
            proc.kill()
            proc.wait()
            wait_gpu_free()
            return False

        if cur > last_dump:
            last_dump = cur
            last_progress = time.time()
            print(f"  [{vname}] settle frame {cur}/{expected_after}, gpu={mem_frac:.0%}", flush=True)
        else:
            stall_min = (time.time() - last_progress) / 60
            if stall_min >= STALL_MINUTES:
                print(f"  [{vname}] STALLED {stall_min:.0f}min at frame {cur}, killing ...", flush=True)
                proc.kill()
                proc.wait()
                wait_gpu_free()
                return False

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [{vname}] settle FAILED (exit {proc.returncode}, {elapsed / 60:.1f}min)")
        wait_gpu_free()
        return False

    final = max_dump_frame(vname)
    print(f"  [{vname}] settle OK ({elapsed / 60:.1f}min, final frame {final})")
    return True


def main() -> None:
    results: dict[str, str] = {}

    print(f"\n{'#' * 70}")
    print(f"  Settle {SETTLE_FRAMES} extra frames + export USD for {len(TASKS)} variants")
    print(f"{'#' * 70}")

    for i, (di, nl) in enumerate(TASKS):
        vname = variant_name(di, nl)
        print(f"\n  >>> Task {i + 1}/{len(TASKS)}: {vname}", flush=True)

        ok = run_settle(di, nl)
        if ok:
            results[f"{vname}"] = "OK"
        else:
            results[f"{vname}"] = "FAILED"

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for key in sorted(results.keys()):
        print(f"  [{results[key]:>15s}] {key}")
    ok_count = sum(1 for v in results.values() if v == "OK")
    print(f"\n  Total: {len(results)} tasks, {ok_count} passed, {len(results) - ok_count} failed")
    if any(v != "OK" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
