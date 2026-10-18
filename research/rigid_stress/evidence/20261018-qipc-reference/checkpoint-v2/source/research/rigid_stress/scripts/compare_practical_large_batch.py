"""Compare complete saved moving fields from two qualified large-batch runs under Slurm."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    runs = (args.reference_run, args.candidate_run)
    inputs, summaries = {}, []
    for run in runs:
        assert (run / "program-returncode.txt").read_text().strip() == "0"
        assert (run / "returncode.txt").read_text().strip() == "0"
        assert (run / "engine-before.txt").read_bytes() == (run / "engine-after.txt").read_bytes()
        assert (run / "source-before.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
        path = run / "results/check/summary.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = json.loads(path.read_text())
        assert summary["passed"] and summary["invalid_live_steps"] == 0
        summaries.append(summary)
    for key in (
        "envs",
        "conditions_sha256",
        "initial_contact_block_size",
        "effective_contact_block_size",
        "output_mode",
        "observed_envs",
        "reset_envs",
        "warmup_steps",
        "following_live_steps",
    ):
        assert summaries[0][key] == summaries[1][key], key
    filenames = [sorted(path.name for path in (run / "results/check").glob("step*.npz")) for run in runs]
    assert filenames[0] == filenames[1] and filenames[0]
    checks = []
    for filename in filenames[0]:
        paths = [run / "results/check" / filename for run in runs]
        for path in paths:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as reference, np.load(paths[1]) as candidate:
            np.testing.assert_array_equal(candidate["envs"], reference["envs"])
            np.testing.assert_array_equal(candidate["phase"], reference["phase"])
            fields = {}
            for key in ("tensor_Pa", "von_mises_Pa", "peak_Pa"):
                first, second = candidate[key], reference[key]
                assert first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all()
                np.testing.assert_allclose(first, second, rtol=1e-4, atol=1e-3)
                delta = first - second
                fields[key] = {
                    "components": first.size,
                    "maximum_absolute_difference_Pa": float(np.abs(delta).max()),
                    "relative_L2_difference": float(np.linalg.norm(delta) / max(np.linalg.norm(second), 1e-300)),
                }
            checks.append({"snapshot": filename, "fields": fields})
    result = {
        "scope": __doc__,
        "reference_run": str(args.reference_run),
        "candidate_run": str(args.candidate_run),
        "envs": summaries[0]["envs"],
        "observed_envs": summaries[0]["observed_envs"],
        "original_comparison_bounds": {"rtol": 1e-4, "atol_Pa": 1e-3},
        "input_sha256": inputs,
        "checks": checks,
        "passed": True,
        "goal_completion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"snapshots": len(checks), "envs": summaries[0]["envs"], "passed": True}))


if __name__ == "__main__":
    main()
