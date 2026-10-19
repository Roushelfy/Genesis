"""Audit physical and material-frame momentum of complete transient P2 fields."""

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
    receipt = json.loads((args.run / "result.json").read_text())
    rows = []
    for path in sorted((args.run / "frames").glob("level*.npz")):
        with np.load(path) as saved:
            rest, displacement, contact, body, inertia, residual, points, forces = (
                saved[key]
                for key in (
                    "vertices_m",
                    "displacement_m",
                    "forces_n",
                    "body_forces_n",
                    "inertial_forces_n",
                    "equation_residual_n",
                    "source_points_m",
                    "source_forces_n",
                )
            )
            load = float(saved["load_n"])
        current = rest + displacement
        net = contact + body - inertia
        elastic = net + residual
        force_scale, moment_scale = max(load, 1e-3), max(load, 1e-3) * 0.03
        transfer_force = float(np.linalg.norm(contact.sum(axis=0) - forces.sum(axis=0)) / force_scale)
        transfer_moment = float(
            np.linalg.norm(np.cross(rest, contact).sum(axis=0) - np.cross(points, forces).sum(axis=0)) / moment_scale
        )
        force_balance = float(np.linalg.norm(net.sum(axis=0)) / force_scale)
        material_moment = float(np.linalg.norm(np.cross(rest, net).sum(axis=0)) / moment_scale)
        current_moment = float(np.linalg.norm(np.cross(current, net).sum(axis=0)) / moment_scale)
        linearized_internal_moment = float(np.linalg.norm(np.cross(current, elastic).sum(axis=0)) / moment_scale)
        rows.append(
            {
                "field": file_id(path),
                "load_n": load,
                "maximum_nodal_displacement_m": float(np.linalg.norm(displacement, axis=1).max()),
                "source_force_transfer_relative_error": transfer_force,
                "source_material_moment_transfer_relative_error": transfer_moment,
                "complete_force_balance_relative_error": force_balance,
                "material_moment_balance_relative_error": material_moment,
                "current_moment_balance_relative_error": current_moment,
                "linearized_internal_current_moment_relative_error": linearized_internal_moment,
                "source_transfer_passed": bool(max(transfer_force, transfer_moment) <= 1e-7),
                "physical_momentum_passed": bool(max(force_balance, current_moment) <= 1e-7),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "run": file_id(args.run / "result.json"),
            "history_admitted": receipt["mechanical_history_passed"],
            "frames": rows,
            "passed": bool(
                receipt["mechanical_history_passed"]
                and rows
                and all(row["source_transfer_passed"] and row["physical_momentum_passed"] for row in rows)
            ),
            "scope": "Complete nodal force and current-position moment balances, normalized by commanded load and load times0.03m, with unchanged1e-7 mechanical target. Material-frame balance alone cannot qualify finite-rotation mechanics. No pin, moment correction, force rescaling or displacement modification is applied.",
        },
    )


if __name__ == "__main__":
    main()
