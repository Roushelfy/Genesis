"""Validate a one-substep observer against the original ten-substep capped scene."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compare_fields import index_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--micro", type=Path, required=True)
    parser.add_argument("--kind", choices=("native", "shell"), required=True)
    parser.add_argument("--comparison-scope", choices=("macro-micro", "stress-sampling"), default="macro-micro")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    macro, micro = index_frames(args.macro), index_frames(args.micro)
    assert micro.keys() <= macro.keys()
    keys = (
        ["q", "link_position_m", "link_quaternion_wxyz", "stress_voigt_link_pa"]
        if args.kind == "native"
        else ["q", "vertices_world_m", "cauchy_world_pa"]
    )
    rows = []
    for tick, path in micro.items():
        with np.load(path) as actual, np.load(macro[tick]) as expected:
            differences = {}
            passed = True
            for key in keys:
                delta = actual[key] - expected[key]
                maximum = np.abs(delta).max().item()
                reference_norm = np.linalg.norm(expected[key])
                relative = np.linalg.norm(delta).item() / reference_norm if reference_norm > 0 else None
                if key in ("stress_voigt_link_pa", "cauchy_world_pa"):
                    # A1Pa RMS floor keeps an unstrained release state from
                    # dominating a relative comparison through cancellation.
                    rms = reference_norm / np.sqrt(expected[key].size)
                    admitted = relative <= 1e-4 if rms > 1 else maximum <= 1e-3
                else:
                    admitted = maximum <= 1e-8
                admitted = bool(admitted)
                passed = passed and admitted
                differences[key] = {
                    "maximum_absolute_difference": maximum,
                    "relative_l2_difference": relative,
                    "within_declared_tolerance": admitted,
                }
            rows.append({"time_s": tick * 0.01, "passed": passed, "differences": differences})
    success = all(row["passed"] for row in rows) and bool(rows)
    write_json(
        args.output / "result.json",
        {
            "kind": "held-controller-macro-micro-cadence-equivalence",
            "comparison_scope": args.comparison_scope,
            "identity": runtime_identity(),
            "macro": file_id(args.macro / "result.json"),
            "micro": file_id(args.micro / "result.json"),
            "observations": rows,
            "passed": success,
            "scope": (
                "Same physical/controller steps and prescribed input history, with different stress sampling. Every new keyframe is compared. "
                if args.comparison_scope == "stress-sampling"
                else "Same1ms physical step and unchanged10ms capped target/radius cadence. Separate Scene.step boundaries retain physical-step fields. "
            )
            + "Equivalence requires every saved common-time pose within1e-8 and every field within1e-4 relative above1Pa RMS, or1e-3Pa absolute at near-zero strain. Physical accuracy and primary throughput remain separate qualifications.",
        },
    )
    if not success:
        raise RuntimeError("Cadence equivalence did not pass; preserve the micro history as a separate variant")


if __name__ == "__main__":
    main()
