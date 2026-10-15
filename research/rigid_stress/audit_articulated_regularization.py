"""Separate contact-distance regularization from fixed-mesh full-arm contact histories."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .audit_rigid_egg_feedback import audit_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    configs = [json.loads((r / "results/configuration.json").read_text()) for r in args.run]
    for old, new in pairwise(configs):
        assert old["dhat"] > new["dhat"]
        for key in old:
            if key not in ["dhat", "output", "mjcf", "closure_run"]:
                assert old[key] == new[key], key
    assert len({hashlib.sha256(Path(c["mjcf"]).read_bytes()).hexdigest() for c in configs}) == 1
    cases, histories, quadratures = [], [], []
    for index, run in enumerate(args.run):
        output = args.output / f"case{index:02d}"
        output.mkdir()
        case, history, weights = audit_case(run, False, args.steps, output)
        case["dhat_m"] = configs[index]["dhat"]
        cases.append(case)
        histories.append(history)
        quadratures.append(weights)
    for weights in quadratures[1:]:
        np.testing.assert_array_equal(weights, quadratures[0])

    def norm(tensor):
        return float(np.sqrt(np.sum(quadratures[0] * np.sum(tensor * tensor, axis=(-2, -1)))))

    comparisons = []
    for coarse, fine in pairwise(range(len(cases))):
        rows = []
        for index, (old, new) in enumerate(zip(histories[coarse], histories[fine])):
            absolute = norm(old["reference_tensor"] - new["reference_tensor"])
            row = {
                "step": index + 1,
                "time_s": (index + 1) * configs[coarse]["dt"],
                "objective_nonlinear_tensor_absolute_volume_L2_Pa_m32": absolute,
                "objective_nonlinear_tensor_relative_change": absolute / norm(new["reference_tensor"]),
                "nonlinear_energy_relative_change": abs(old["nonlinear_energy_J"] / new["nonlinear_energy_J"] - 1),
                "raw_nonlinear_peak_relative_change": abs(
                    old["nonlinear_raw_peak_lower_Pa"] / new["nonlinear_raw_peak_lower_Pa"] - 1
                ),
            }
            for label, section in [("force_N", slice(0, 3)), ("moment_Nm", slice(3, 6))]:
                difference = float(
                    np.linalg.norm(old["patch_wrenches"][:, section] - new["patch_wrenches"][:, section])
                )
                denominator = float(np.linalg.norm(new["patch_wrenches"][:, section]))
                row["patch_" + label + "_absolute_change"] = difference
                row["patch_" + label + "_relative_change"] = difference / denominator if denominator else None
            rows.append(row)
        comparisons.append({"coarse_case": coarse, "fine_case": fine, "rows": rows})
    summary = {
        "scope": __doc__,
        "cases": cases,
        "adjacent_comparisons": comparisons,
        "goal_completion": False,
        "limitations": "Only barrier activation distance changes. Geometry, material, collision quadrature, friction, control, time step and held-initializer protocol remain fixed; each distance solves its own corresponding closure. Complete independent body/joint/fresh-step gates and each Coulomb bound are retained. This is one unresolved coarse spatial/contact setting, not spatial convergence, finite-footprint model validation, original approach/angular-history matching or a physical accuracy/cost operating range. Nonzero full regularized-contact torque is recorded, not projected away.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"cases": len(cases), "steps_per_case": args.steps, "final_changes": [c["rows"][-1] for c in comparisons]}
        )
    )


if __name__ == "__main__":
    main()
