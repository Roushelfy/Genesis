"""Audit full/max live-pad cost preflights without treating them as primary timing."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, cases = [], []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    for method in ("dpr", "scr", "shell"):
        costs, histories, admitted = [], [], []
        for group, mode in enumerate(("full", "max")):
            path = args.run / f"group{group}" / method
            cost = read(path / "cost.json")
            assert cost["output_mode"] == mode and cost["completed"] and not cost["primary_timing"]
            assert cost["envs"] == 1
            audit = read(path.parent / (method + "-audit") / "result.json")
            if method == "shell":
                admitted.append(
                    len(audit["runs"]) == 1
                    and audit["runs"][0]["completed"]
                    and all(w["mechanical_window_passed"] for w in audit["runs"][0]["equilibrium_windows"])
                )
            else:
                admitted.append(audit["passed"])
            journal = path / "steps.jsonl"
            receipts.append(file_id(journal))
            history = [json.loads(line) for line in journal.read_text().splitlines()]
            if method == "shell":
                admitted[-1] &= all(
                    not any(r["solver_failure_flags"])
                    and r["equation_residual_relative"] <= 1e-3
                    and r["physical_pad"]["normal_law_and_transfer_passed"]
                    for r in history
                )
            histories.append(history)
            measured = [r for r in cost["samples"] if not r["warmup"]]
            assert len(measured) == 1 and measured[0]["steps"] == 200
            assert measured[0]["all_stress_observations_valid"]
            assert measured[0]["stress_observations"] == 200 // cost["stress_stride"]
            assert (measured[0]["field_host_bytes"] == 0) == (mode == "max")
            costs.append(cost)
        assert len(histories[0]) == len(histories[1]) == 400
        peaks = []
        for history in histories:
            peaks.append(np.array([r["stress"]["raw_vm_peak_pa"] for r in history if "stress" in r]))
        np.testing.assert_array_equal(
            [r["step"] for r in histories[0] if "stress" in r],
            [r["step"] for r in histories[1] if "stress" in r],
        )
        peak_error = float(np.max(np.abs(peaks[0] - peaks[1])) / max(np.max(np.abs(peaks[0])), 1e-30))
        is_shell = method == "shell"
        field_name = "cauchy_world_pa" if is_shell else "stress_pa"
        observed = args.run / "group0" / method / "frames" / ("level0.npz" if is_shell else "stress-level0.npz")
        if is_shell:
            baseline = args.data_root / "compliant-pad-fine-coarse-history-v1/group0/cycle/frames/level0.npz"
        else:
            index = 0 if method == "dpr" else 1
            baseline = (
                args.data_root / f"native-pad-full-source-stability-v1/group{index}/native/frames/stress-level0.npz"
            )
        receipts.extend((file_id(observed), file_id(baseline)))
        with np.load(observed) as saved:
            tensor = saved[field_name]
        with np.load(baseline) as saved:
            target = saved[field_name]
        assert tensor.shape == target.shape
        field_error = float(np.linalg.norm(tensor - target) / np.linalg.norm(target))
        checks = {
            "both_complete_histories_admitted": all(admitted),
            "full_max_observed_peak_agreement": peak_error <= 1e-8,
            "baseline_complete_field_agreement": field_error <= 0.005,
        }
        cases.append(
            {
                "method": method,
                "checks": checks,
                "full_max_peak_relative_difference": peak_error,
                "baseline_same_mesh_all_tensor_values_relative_l2": field_error,
                "costs": costs,
            }
        )
    passed = all(all(c["checks"].values()) for c in cases)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": passed,
            "cases": cases,
            "receipts": receipts,
            "primary_timing": False,
            "scope": "Matched B1 normal400-step preflight, first200 warmup, full/max modes. Each cost uses its own complete mechanical audit and all observed stress acceptance. Full/max peaks are checked across their common cadence; full fields are checked against the original same-mesh first equilibrium with the predeclared0.5% solver sensitivity target. This audit does not establish spatial accuracy, primary uninstrumented throughput, repeat dispersion or batched performance.",
        },
    )
    assert passed, "Cost preflight failed; retain per-case checks"


if __name__ == "__main__":
    main()
