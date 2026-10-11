"""Compare closed prototype and production post-graph pairs on identical GPUs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    content = path.read_bytes()
    source = json.loads(content)
    rows = source["repeats"]
    seconds = sum(row["seconds"] for row in rows)
    steps = sum(row["steps"] for row in rows)
    invalid = sum(row["invalid_environment_steps"] for row in rows)
    rates = [row["transitions_per_second"] for row in rows]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "valid_env_steps_per_second": (source["envs"] * steps - invalid) / seconds,
        "batch_steps_per_second": steps / seconds,
        "invalid_environment_steps": invalid,
        "repeat_valid_rates": rates,
        "repeat_std_valid_rate": float(np.std(rates, ddof=1)),
        "gpu_uuid": source["gpu_uuid"],
        "source_revision": source["source_revision"],
        "source_sha256": source["source_sha256"],
        "effective_rigid_post_fusion": source.get("effective_rigid_post_fusion"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons, missing = [], []
    for kind, batches, scopes, variants in (
        ("prototype-v1", (1024, 32768), ("policy",), ("original", "fused")),
        ("production-v9", (1024, 2048), ("live", "policy"), ("off", "on")),
        ("production-v9", (32768, 61440), ("policy",), ("off", "on")),
    ):
        for batch in batches:
            for seed in (510000, 623001):
                for scope in scopes:
                    if kind == "prototype-v1":
                        paths = [
                            args.run / f"postgraph-{route}-{scope}-b{batch}-seed{seed}-v1.json" for route in variants
                        ]
                    else:
                        paths = [
                            args.run / f"native-postgraph-production-{route}-{scope}-b{batch}-seed{seed}-v9.json"
                            for route in variants
                        ]
                    if not all(path.exists() for path in paths):
                        missing.append({"kind": kind, "envs": batch, "seed": seed, "scope": scope})
                        continue
                    pair = [load(path) for path in paths]
                    assert pair[0]["gpu_uuid"] == pair[1]["gpu_uuid"]
                    assert pair[0]["source_sha256"] == pair[1]["source_sha256"]
                    if kind == "production-v9":
                        assert [row["effective_rigid_post_fusion"] for row in pair] == [False, True]
                    gain = 100 * (pair[1]["valid_env_steps_per_second"] / pair[0]["valid_env_steps_per_second"] - 1)
                    row = {
                        "kind": kind,
                        "envs": batch,
                        "seed": seed,
                        "scope": scope,
                        "measurements": dict(zip(variants, pair)),
                        "throughput_gain_percent": gain,
                    }
                    comparisons.append(row)
                    print(
                        kind,
                        batch,
                        seed,
                        scope,
                        gain,
                        "std",
                        [item["repeat_std_valid_rate"] for item in pair],
                        flush=True,
                    )
    args.output.write_text(
        json.dumps(
            {
                "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "comparisons": comparisons,
                "missing": missing,
                "complete": not missing,
                "scope": "900 warmup, three 2400-step measured changing trajectories on one physical GPU per matched pair. Prototype graph transform metadata and exact generated source accompany each case; production uses ordinary engine flags. No failed environment is excluded from time.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
