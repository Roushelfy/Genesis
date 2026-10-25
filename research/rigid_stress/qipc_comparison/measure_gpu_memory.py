"""Sample visible-device memory through cold preparation and a complete benchmark.

This intrusive pass measures memory, not headline throughput. Device-used memory
includes driver/context allocations. Sampling interval is retained; a sampled
maximum is a lower bound on a transient instantaneous maximum.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def sample():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    rows = []
    for line in output.splitlines():
        uuid, used, total = line.split(",")
        rows.append(dict(uuid=uuid.strip(), used_MiB=int(used), total_MiB=int(total)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--output-mode", choices=("max", "full"), required=True)
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--source-revision", default="5c44042a4")
    args, forwarded = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "examples.speed_benchmark.rigid_stress",
        "--scope",
        "live",
        "--stress-backend",
        args.stress_backend,
        "--output-mode",
        args.output_mode,
        "--envs",
        str(args.envs),
        "--conditions",
        str(args.conditions),
        "--radius-model",
        "hertz",
        "--pressure-profile",
        "hertz",
        "--box-edge-policy",
        "planar",
        "--quadrature",
        "16",
        "--warmup",
        "600",
        "--steps",
        "600",
        "--repetitions",
        "1",
        "--minimum-seconds",
        "0",
        "--compact-log",
        "--source-revision",
        args.source_revision,
        *forwarded,
        "--output",
        str(args.output / "benchmark.json"),
    ]
    initial = sample()
    assert len(initial) == 1, "This memory control requires one visible allocated GPU."
    started = time.perf_counter()
    rows = [dict(seconds=0.0, **initial[0])]
    with (args.output / "benchmark.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            rows.append(dict(seconds=time.perf_counter() - started, **sample()[0]))
            time.sleep(0.2)
        status = process.wait()
    rows.append(dict(seconds=time.perf_counter() - started, **sample()[0]))
    result = dict(
        complete=True,
        returncode=status,
        command=command,
        sampling_sleep_seconds=0.2,
        initial_MiB=rows[0]["used_MiB"],
        sampled_peak_MiB=max(r["used_MiB"] for r in rows),
        observed_max_gap_seconds=max(b["seconds"] - a["seconds"] for a, b in zip(rows, rows[1:])),
        note=__doc__,
        rows=rows,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert status == 0, status


if __name__ == "__main__":
    main()
