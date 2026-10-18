"""Explain native interface rejection from the unmodified failed-step arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads((args.run / "start.json").read_text())
    radius = start["args"]["contact_radius"]
    path = args.run / "rejected-state.npz"
    rows = []
    with np.load(path) as saved:
        prefix = "stress.0.intrinsic."
        for env, contact in np.argwhere(saved[prefix + "contacts.valid"]):

            def value(name):
                return saved[prefix + name][env, contact]

            force = value("contacts.force")
            total = value("tractions.acceptance.force")
            moment = value("tractions.acceptance.moment")
            expected_moment = value("sources.free_moment")
            force_error = np.linalg.norm(total - force).item()
            moment_error = np.linalg.norm(moment - expected_moment).item()
            force_budget = 1e-8 * np.linalg.norm(force).item()
            moment_budget = force_budget * radius
            rows.append(
                {
                    "env": int(env),
                    "packed_contact": int(contact),
                    "auxiliary_aggregate_force_n": force.tolist(),
                    "retained_physical_source_prior_force_n": value("tractions.priors.force").tolist(),
                    "retained_group_force_scale_n": float(value("tractions.priors.group_scale")),
                    "reconstructed_force_n": total.tolist(),
                    "auxiliary_aggregate_free_moment_nm": expected_moment.tolist(),
                    "reconstructed_moment_nm": moment.tolist(),
                    "force_error_n": force_error,
                    "force_budget_n": force_budget,
                    "moment_error_nm": moment_error,
                    "moment_budget_nm": moment_budget,
                    "force_acceptance_passed": force_error <= force_budget,
                    "moment_acceptance_passed": moment_error <= moment_budget,
                    "local_friction_acceptance_passed": bool(value("tractions.acceptance.cone_valid")),
                    "traction_fit_status": int(value("tractions.status")),
                    "fit_force_error_normalized": float(value("tractions.force_error")),
                    "fit_moment_error_normalized": float(value("tractions.moment_error")),
                }
            )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "input": file_id(path),
            "rejection": json.loads((args.run / "rejected-state.json").read_text()),
            "contacts": rows,
            "scope": "Read-only evaluation of final wrench/cone acceptance in cooperative_scatter.py, using packed contact arrays. body_source_prior replaces auxiliary constraints by the net body wrench while retaining opposing physical priors; only the leader aggregate is constrained and its failure propagates to all members. No acceptance relaxation or reconstructed successful field.",
        },
    )


if __name__ == "__main__":
    main()
