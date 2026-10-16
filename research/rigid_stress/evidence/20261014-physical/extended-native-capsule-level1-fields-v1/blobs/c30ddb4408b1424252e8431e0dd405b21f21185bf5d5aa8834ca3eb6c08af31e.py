"""Compare complete final grasp fields and every retained original force gate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs, reports, fields, inputs = [], [], [], {}
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        directory = run / "solution" if (run / "solution/result.json").exists() else run / "results"
        configs.append(json.loads((directory / "configuration.json").read_text()))
        report = json.loads((directory / "result.json").read_text())
        assert report["complete_constrained_residual_N"] <= 1e-9
        assert report["complete_body_residual_N"] <= 1e-9
        assert max(r["complete_free_residual_n"] for r in report["accepted_steps"]) <= 1e-9
        reports.append(report)
        with np.load(directory / "fields.npz") as saved:
            fields.append({key: saved[key].copy() for key in saved.files})
        for name in ["fields.npz", "configuration.json", "result.json"]:
            f = directory / name
            inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    assert {k: v for k, v in configs[0].items() if k != "output"} == {
        k: v for k, v in configs[1].items() if k != "output"
    }
    assert [(r["phase"], r["step"]) for r in reports[0]["accepted_steps"]] == [
        (r["phase"], r["step"]) for r in reports[1]["accepted_steps"]
    ]
    for key in ["rest_positions", "surface_triangles", "applied_force_N", "body_component_permutation"]:
        np.testing.assert_array_equal(fields[0][key], fields[1][key])
    changes = {}
    for key in ["displacement", "contact_force_N", "body_tensor_Pa", "body_vm_Pa"]:
        old, new = fields[0][key], fields[1][key]
        delta, norm = float(np.linalg.norm(old - new)), float(np.linalg.norm(new))
        relative_budget = 1e-6 if key in ["body_tensor_Pa", "body_vm_Pa"] else 1e-5
        assert delta <= relative_budget * norm + 1e-12, (key, delta, norm)
        changes[key] = {"absolute_l2_difference": delta, "relative_l2_difference": delta / max(norm, 1e-12)}
    peak = abs(reports[0]["raw_body_peak_Pa"] / reports[1]["raw_body_peak_Pa"] - 1)
    assert peak < 1e-6
    result = {
        "scope": __doc__,
        "runs": [str(r) for r in args.run],
        "field_changes": changes,
        "raw_peak_relative_difference": peak,
        "raw_peak_positions_m": [r["raw_body_peak_position_m"] for r in reports],
        "retained_accepted_force_gates_checked_per_run": [len(r["accepted_steps"]) for r in reports],
        "max_complete_force_gate_N": max(
            r["complete_free_residual_n"] for report in reports for r in report["accepted_steps"]
        ),
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Final all-part displacement/contact loads and complete unaveraged body tensor/von Mises fields agree under identical physical inputs. Every retained accepted-step original1e-9N gate is checked, but intermediate full fields were not retained and their equivalence is not claimed. This is reference numerical qualification, not physical contact/mesh/history convergence or application/accuracy-cost acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
