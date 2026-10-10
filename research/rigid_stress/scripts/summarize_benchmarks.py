"""Summarize completed native repeats without discarding invalid transitions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        content = path.read_bytes()
        source = json.loads(content)
        repeats = source["repeats"]
        envs = source["envs"]
        steps = sum(row["steps"] for row in repeats)
        seconds = sum(row["seconds"] for row in repeats)
        invalid = sum(row["invalid_environment_steps"] for row in repeats)
        rates = [row["transitions_per_second"] for row in repeats]
        rows.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "scope": source["scope"],
                "envs": envs,
                "seed": source["seed"],
                "output_mode": source.get("output_mode"),
                "reset_mode": source.get("reset_mode"),
                "source_revision": source["source_revision"],
                "source_sha256": source["source_sha256"],
                "gpu_uuid": source["gpu_uuid"],
                "gpu": source["gpu"],
                "steps": steps,
                "seconds": seconds,
                "invalid_environment_steps": invalid,
                "valid_env_steps_per_second": (envs * steps - invalid) / seconds,
                "attempted_env_steps_per_second": envs * steps / seconds,
                "batch_steps_per_second": steps / seconds,
                "repeat_valid_rates": rates,
                "repeat_std_valid_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                "native_buffer_bytes": source.get("native_buffer_bytes"),
                "saved_reset_state_bytes": source.get("saved_reset_state_bytes"),
                "device_free_bytes_at_end": source["device_free_bytes"],
                "policy": source["policy"],
            }
        )
        print(path.name, rows[-1]["valid_env_steps_per_second"], "invalid", invalid, flush=True)
        del content, source
    args.output.write_text(
        json.dumps(
            {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "measurements": rows}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
