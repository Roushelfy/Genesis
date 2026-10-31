"""Compare captured automatic-shell fields, exact mechanical inputs and native memory scaling off the timing path."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.rigid_stress.qipc_comparison.artifacts import file_id, write_json


def relative(reference, candidate):
    return float(np.max(abs(candidate - reference)) / max(np.max(abs(reference)), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.baseline / "results.json").read_text())
    rows = []
    for path in args.candidate:
        if not (path / "results.json").is_file():
            rows.append(
                {
                    "candidate": str(path),
                    "completed": False,
                    "reason": "No accepted scene result; inspect preserved workload log.",
                }
            )
            continue
        candidate = json.loads((path / "results.json").read_text())
        row = {
            "baseline": file_id(args.baseline / "results.json"),
            "candidate": file_id(path / "results.json"),
            "maxima_relative": relative(
                np.array(max(reference["maximum_stress_Pa"])), np.array(max(candidate["maximum_stress_Pa"]))
            ),
        }
        if reference["arguments"]["envs"] == candidate["arguments"]["envs"]:
            with np.load(args.baseline / "contacts.npz") as old, np.load(path / "contacts.npz") as new:
                row["contact_inputs_relative"] = {
                    key: relative(old[key], new[key]) for key in ("position", "force", "radius")
                }
                row["contact_valid_identical"] = bool(np.array_equal(old["valid"], new["valid"]))
            for filename in ("warmup-fields.npz", "fields.npz"):
                if (args.baseline / filename).is_file() and (path / filename).is_file():
                    with np.load(args.baseline / filename) as old, np.load(path / filename) as new:
                        row[filename] = {key: relative(old[key], new[key]) for key in old.files}
        else:
            n0, n1 = reference["arguments"]["envs"], candidate["arguments"]["envs"]
            old = {item["name"]: item["bytes"] for item in reference["native_buffers"]}
            new = {item["name"]: item["bytes"] for item in candidate["native_buffers"]}
            assert old.keys() == new.keys()
            growth = {key: (new[key] - old[key]) / (n1 - n0) for key in old}
            shared = {key: old[key] - n0 * growth[key] for key in old}
            assert min(growth.values()) >= 0 and min(shared.values()) >= 0
            row["native_memory"] = {
                "shared_bytes": sum(shared.values()),
                "per_environment_bytes": sum(growth.values()),
                "by_buffer": {key: {"shared_bytes": shared[key], "per_environment_bytes": growth[key]} for key in old},
                "scope": "Pointer-deduplicated native tensor payloads fitted to the two measured batches; excludes allocator overhead, rigid solver storage and process runtime.",
            }
        rows.append(row)
    write_json(args.output / "comparison.json", rows)


if __name__ == "__main__":
    main()
