"""Check shared environments inside independently evolving input batches."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--small", type=Path, required=True)
    cli.add_argument("--large", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    small = json.loads((args.small / "result.json").read_text())
    large = json.loads((args.large / "result.json").read_text())
    assert small["execution_completed"] and large["execution_completed"]
    count = small["envs"]
    assert count < large["envs"] and small["steps"] == large["steps"]
    with np.load(args.small / "conditions.npz") as first, np.load(args.large / "conditions.npz") as second:
        assert first.files == second.files
        for key in first.files:
            np.testing.assert_array_equal(second[key][:count], first[key])
    with np.load(args.small / "trajectory.npz") as first, np.load(args.large / "trajectory.npz") as second:
        assert first.files == second.files
        for key in first.files:
            np.testing.assert_array_equal(second[key][:, :count], first[key])
    errors = {}
    for path in sorted(args.small.glob("step*.npz")):
        with np.load(path) as first, np.load(args.large / path.name) as second:
            assert first.files == second.files
            slots = first["source_count"].shape[1]
            for key in first.files:
                if key == "query_evaluations":
                    continue
                left, right = first[key], second[key]
                if key.startswith("hertz_"):
                    right = right.reshape((slots, large["envs"]) + right.shape[1:])[:, :count]
                    right = right.reshape(left.shape)
                else:
                    right = right[:count]
                if key == "curvature":
                    np.testing.assert_allclose(right, left, rtol=1e-11, atol=1e-9)
                elif key in ("center", "normal", "normal_residual", "gap", "force", "moment", "hertz_axes"):
                    np.testing.assert_allclose(right, left, rtol=1e-11, atol=1e-12)
                else:
                    np.testing.assert_array_equal(right, left)
                if np.issubdtype(left.dtype, np.floating):
                    errors[key] = max(errors.get(key, 0.0), float(np.max(np.abs(right - left), initial=0)))
    steps = 0
    with (
        gzip.open(args.small / "history.jsonl.gz", "rt") as first,
        gzip.open(args.large / "history.jsonl.gz", "rt") as second,
    ):
        for first_line, second_line in zip(first, second, strict=True):
            left, right = json.loads(first_line), json.loads(second_line)
            assert left["step"] == right["step"] == steps
            assert left["is_valid"] == right["is_valid"][:count]
            assert left["region_count"] == right["region_count"][:count]
            selected = [i for i, pair in enumerate(right["source_indices"]) if pair[0] < count]
            for key in ("source_indices", "source_status", "cache_hint_kind", "cache_hit"):
                if key in left:
                    assert left[key] == [right[key][i] for i in selected], (steps, key)
            for key, tolerance in (("curvature", 1e-9), ("normal_residual", 1e-12)):
                np.testing.assert_allclose([right[key][i] for i in selected], left[key], rtol=1e-11, atol=tolerance)
            steps += 1
    assert steps == small["steps"]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "small": file_id(args.small / "result.json"),
            "large": file_id(args.large / "result.json"),
            "shared_environments": count,
            "large_environments": large["envs"],
            "steps": steps,
            "snapshot_maximum_absolute_differences": errors,
            "scope": "Exact initial/controller conditions and motion for the shared environment prefix, exact every-step validity/source identity/region/cache status, and sampled tensors at the implementation-conformance tolerances. Total reset count is batch-wide and is not compared. Query evaluation counters are diagnostic. No physical stress claim.",
        },
    )


if __name__ == "__main__":
    main()
