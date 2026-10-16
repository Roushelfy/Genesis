"""Compare every saved inertial state and every unaveraged corner tensor after a solver change."""

import argparse
import json
from pathlib import Path

import numpy as np

from .skfem_reference import SkfemElasticReference
from .stable_neohookean_reference import StableNeoHookeanBulk
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = []
    reports = []
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        configs.append(json.loads((run / "results/configuration.json").read_text()))
        reports.append(json.loads((run / "results/summary.json").read_text()))
    assert {k: v for k, v in configs[0].items() if k != "output"} == {
        k: v for k, v in configs[1].items() if k != "output"
    }
    for report in reports:
        assert len(report["steps"]) == configs[0]["steps"]
        assert max(r["complete_dynamic_residual_N"] for r in report["steps"]) <= 1e-9
        assert max(r["complete_body_dynamic_residual_N"] for r in report["steps"]) <= 1e-9
    with (
        np.load(args.run[0] / "results/reference_geometry.npz") as a,
        np.load(args.run[1] / "results/reference_geometry.npz") as b,
    ):
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        reference = SkfemElasticReference(a["vertices"], a["tetrahedra"], 1e10, 0.3, 2000)
    bulk = StableNeoHookeanBulk(reference)
    rows = []
    maxima = {}
    for step in range(1, configs[0]["steps"] + 1):
        states = []
        for run in args.run:
            with np.load(run / f"results/step{step:04d}.npz") as saved:
                states.append({k: saved[k].copy() for k in saved.files})
        np.testing.assert_array_equal(states[0]["body_component_permutation"], states[1]["body_component_permutation"])
        count = reference.basis.N
        tensors = []
        peaks = []
        for state in states:
            vector = np.empty(count)
            vector[state["body_component_permutation"]] = state["displacement"][:count]
            tensor, vm, upper = bulk.stress_bounds(vector)
            tensors.append(tensor)
            peaks.append([float(vm.max()), float(upper.max())])
        record = {"step": step}
        for key in ["displacement", "velocity", "acceleration", "contact_force_N", "applied_force_N"]:
            for label, section in [(key, slice(None)), ("body_" + key, slice(0, count))]:
                a, b = states[0][key][section], states[1][key][section]
                error = float(np.linalg.norm(a - b))
                scale = float(np.linalg.norm(b))
                assert error <= 1e-5 * scale + 1e-12, (step, label, error, scale)
                relative = error / max(scale, 1e-12)
                maxima[label] = max(maxima.get(label, 0), relative)
                record[label + "_relative_change"] = relative
        tensor_error = float(np.linalg.norm(tensors[0] - tensors[1]) / np.linalg.norm(tensors[1]))
        assert tensor_error < 1e-6
        record["full_corner_tensor_relative_change"] = tensor_error
        record["raw_peak_lower_upper_Pa"] = peaks
        rows.append(record)
    result = {
        "scope": __doc__,
        "states_checked": len(rows),
        "body_tetrahedra": reference.mesh.nelements,
        "state_relative_maxima": maxima,
        "max_full_corner_tensor_relative_change": max(r["full_corner_tensor_relative_change"] for r in rows),
        "rows": rows,
        "unchanged_full_force_budget_N": 1e-9,
        "goal_completion": False,
        "limitations": "Same complete physical inputs and original full mapped/body dynamic force gates; all original field states and unaveraged corner tensors are compared. This qualifies reference numerical equivalence only, not physical convergence, full-arm matching or an accuracy-cost operating range.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
