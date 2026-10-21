"""Compare complete histories and sampled outputs after input implementation changes."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    first = json.loads((args.baseline / "result.json").read_text())
    second = json.loads((args.candidate / "result.json").read_text())
    assert first["execution_completed"] and second["execution_completed"]
    assert (first["steps"], first["envs"]) == (second["steps"], second["envs"])
    with np.load(args.baseline / "conditions.npz") as baseline, np.load(args.candidate / "conditions.npz") as candidate:
        assert baseline.files == candidate.files
        for key in baseline.files:
            np.testing.assert_array_equal(candidate[key], baseline[key])
    motion = {}
    with np.load(args.baseline / "trajectory.npz") as baseline, np.load(args.candidate / "trajectory.npz") as candidate:
        for key in baseline.files:
            np.testing.assert_array_equal(candidate[key], baseline[key])
            motion[key] = float(np.max(np.abs(candidate[key] - baseline[key])))
    with gzip.open(args.baseline / "history.jsonl.gz", "rt") as stream:
        baseline_history = [json.loads(line) for line in stream]
    with gzip.open(args.candidate / "history.jsonl.gz", "rt") as stream:
        candidate_history = [json.loads(line) for line in stream]
    assert len(baseline_history) == len(candidate_history) == first["steps"]
    history_differences = {"curvature": 0.0, "normal_residual": 0.0}
    for baseline, candidate in zip(baseline_history, candidate_history):
        for key in ["step", "reset_count", "is_valid", "source_indices", "source_status", "region_count"]:
            assert baseline[key] == candidate[key], (baseline["step"], key)
        for key, atol in [("curvature", 1e-9), ("normal_residual", 1e-12)]:
            left, right = np.array(baseline[key]), np.array(candidate[key])
            np.testing.assert_allclose(right, left, rtol=1e-11, atol=atol)
            history_differences[key] = max(history_differences[key], float(np.max(np.abs(left - right), initial=0)))
    snapshots = []
    for path in sorted(args.baseline.glob("step*.npz")):
        differences = {}
        with np.load(path) as baseline, np.load(args.candidate / path.name) as candidate:
            assert baseline.files == candidate.files
            for key in baseline.files:
                left, right = baseline[key], candidate[key]
                if key in ("curvature",):
                    np.testing.assert_allclose(right, left, rtol=1e-11, atol=1e-9)
                elif key in ("center", "normal", "normal_residual", "gap", "force", "moment", "hertz_axes"):
                    np.testing.assert_allclose(right, left, rtol=1e-11, atol=1e-12)
                else:
                    np.testing.assert_array_equal(right, left)
                if np.issubdtype(left.dtype, np.floating):
                    differences[key] = float(np.max(np.abs(left - right), initial=0))
            snapshots.append({"file": path.name, "maximum_absolute_differences": differences})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "baseline": file_id(args.baseline / "result.json"),
            "candidate": file_id(args.candidate / "result.json"),
            "steps": first["steps"],
            "envs": first["envs"],
            "exact_motion_differences": motion,
            "history_maximum_absolute_differences": history_differences,
            "snapshots": snapshots,
            "scope": "Exact conditions, motion and every-step validity/source identity/region counts; every-step source curvature/residual and all retained input tensors checked with declared numerical tolerances. No physical stress accuracy or complete stress-module claim.",
        },
    )


if __name__ == "__main__":
    main()
