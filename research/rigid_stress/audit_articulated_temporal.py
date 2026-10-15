"""Audit common-time full-arm contact fields with one unchanged closed initial state."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .audit_articulated_contact import audit_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    configurations = [json.loads((r / "results/configuration.json").read_text()) for r in args.run]
    durations = [c["dt"] * c["steps"] for c in configurations]
    np.testing.assert_allclose(durations, durations[0], rtol=0, atol=1e-14)
    for old, new in pairwise(configurations):
        assert old["dt"] > new["dt"]
        for key in old:
            if key not in ["dt", "steps", "mjcf", "output"]:
                assert old[key] == new[key], key
    closure, initial_velocity = None, None
    source_hashes, cases, histories, quadratures = [], [], [], []
    for index, (run, config) in enumerate(zip(args.run, configurations)):
        with np.load(run / "results/closure_state.npz") as saved:
            current = {k: saved[k].copy() for k in saved.files}
        if closure is not None:
            assert current.keys() == closure.keys()
            for key in closure:
                np.testing.assert_array_equal(current[key], closure[key])
        closure = current
        velocity_path = run / "results/initial_velocity.npz"
        if velocity_path.exists():
            with np.load(velocity_path) as saved:
                current_velocity = {k: saved[k].copy() for k in saved.files}
            if initial_velocity is not None:
                assert current_velocity.keys() == initial_velocity.keys()
                for key in initial_velocity:
                    np.testing.assert_array_equal(current_velocity[key], initial_velocity[key])
            initial_velocity = current_velocity
        else:
            assert not config.get("captured_initial_velocity", False)
        source_hashes.append(hashlib.sha256(Path(config["mjcf"]).read_bytes()).hexdigest())
        directory = args.output / f"case{index:02d}"
        directory.mkdir()
        case, history, weights = audit_run(run, directory)
        report = json.loads((run / "results/summary.json").read_text())
        assert report["fixed_arm_control"] is False
        assert (
            report["reused_closure_sha256"]
            == hashlib.sha256((Path(config["closure_run"]) / "results/closure_state.npz").read_bytes()).hexdigest()
        )
        assert all(x[0] >= 2 * config["dt"] for x in report["authored_equality_solref"])
        cases.append(case)
        histories.append(history)
        quadratures.append(weights)
    assert len(set(source_hashes)) == 1
    for weights in quadratures[1:]:
        np.testing.assert_array_equal(weights, quadratures[0])
    common_dt = configurations[0]["dt"]
    comparisons = []
    for first in range(len(cases) - 1):
        ratios = [round(common_dt / configurations[j]["dt"]) for j in [first, first + 1]]
        for j, ratio in zip([first, first + 1], ratios):
            assert abs(ratio * configurations[j]["dt"] - common_dt) < 1e-14
        rows = []
        for common_step in range(1, configurations[0]["steps"] + 1):
            indices = [common_step * ratio - 1 for ratio in ratios]
            old, new = [histories[j][i] for j, i in zip([first, first + 1], indices)]

            def norm(x):
                return float(np.sqrt(np.sum(quadratures[0] * np.sum(x * x, axis=(-2, -1)))))

            tensor_absolute = norm(old["tensor"] - new["tensor"])
            row = {
                "time_s": common_step * common_dt,
                "tensor_absolute_volume_L2_Pa_m32": tensor_absolute,
                "tensor_relative_volume_L2": tensor_absolute / norm(new["tensor"]),
                "energy_relative_change": float(abs(old["energy"] / new["energy"] - 1)),
            }
            for label, section in [("force_N", slice(0, 3)), ("moment_Nm", slice(3, 6))]:
                absolute = float(np.linalg.norm(old["body_wrench"][section] - new["body_wrench"][section]))
                denominator = float(np.linalg.norm(new["body_wrench"][section]))
                row[label + "_absolute_change"] = absolute
                row[label + "_relative_change"] = absolute / denominator if denominator else None
            peaks = [cases[j]["rows"][i]["raw_global_peak_lower_Pa"] for j, i in zip([first, first + 1], indices)]
            row["raw_peak_relative_change"] = abs(peaks[0] / peaks[1] - 1)
            row["raw_peak_absolute_change_Pa"] = abs(peaks[0] - peaks[1])
            for key, unit in [
                ("body_com_position", "m"),
                ("body_com_velocity", "m_s"),
                ("body_angular_velocity", "rad_s"),
            ]:
                row[key + "_absolute_change_" + unit] = float(np.linalg.norm(old[key] - new[key]))
            rows.append(row)
        comparisons.append({"coarse_case": first, "fine_case": first + 1, "common_times": rows})
    report = {
        "scope": __doc__,
        "cases": cases,
        "adjacent_comparisons": comparisons,
        "goal_completion": False,
        "limitations": "All steps retain complete original1e-9 body/joint/fresh-step budgets, common closure arrays and identical declared initial velocities. This tests the short hold at one coarse fixed-polyhedron/contact setting. It does not establish original approach/lift/support-history matching, spatial/contact/regularization convergence, resolved-load/native feedback separation or a physical accuracy/cost range.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "common_times_per_pair": len(comparisons[0]["common_times"])}))


if __name__ == "__main__":
    main()
