"""Inspect saved first-grasp contact/slip states without advancing or relabeling a failed reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.reference / "result.json").read_text())
    config = receipt["config"]
    rows = []
    for path in sorted((args.reference / "frames").glob("step*.npz"))[-20:]:
        with np.load(path) as saved:
            position, force, velocity = saved["positions_m"], saved["contact_forces_n"], saved["velocities_m_per_s"]
            floor = position[:, 2] < config["d_hat"] * (1 + 1e-6)
            ground = force[floor].sum(axis=0)
            denominator = config["friction"] * abs(ground[2])
            rows.append(
                {
                    "time_s": saved["time_s"].item(),
                    "floor_vertices_in_band": int(floor.sum()),
                    "floor_resultant_n": ground.tolist(),
                    "other_contact_resultant_n": force[~floor].sum(axis=0).tolist(),
                    "floor_tangent_force_over_mu_normal": np.linalg.norm(ground[:2]).item() / denominator
                    if denominator > 1e-12
                    else None,
                    "maximum_floor_tangent_speed_m_per_s": np.linalg.norm(velocity[floor, :2], axis=1).max().item()
                    if floor.any()
                    else None,
                    "maximum_other_contact_speed_m_per_s": np.linalg.norm(
                        velocity[(~floor) & (np.linalg.norm(force, axis=1) > 1e-12)], axis=1
                    )
                    .max()
                    .item()
                    if ((~floor) & (np.linalg.norm(force, axis=1) > 1e-12)).any()
                    else None,
                }
            )
    write_json(
        args.output / "result.json",
        {
            "kind": "saved-unqualified-reference-slip-diagnosis",
            "identity": runtime_identity(),
            "receipt": file_id(args.reference / "result.json"),
            "history": rows,
            "scope": "Actual accepted-state nodal forces and velocities before the first failure; plane-band classification is geometric and does not replace per-source contact stencils.",
        },
    )


if __name__ == "__main__":
    main()
