"""Inspect coupled Shell contact histories and retained end-position geometry without changing solver state."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .common_metric import corotation
from .plates import equilibrium


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in args.runs:
        start = json.loads((run / "start.json").read_text())
        result = json.loads((run / "result.json").read_text())
        history = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        with np.load(run / "rest.npz") as saved:
            rest, mass = saved["vertices"], saved["masses_kg"]
        with np.load(start["args"]["mesh"]) as saved:
            solid = saved["vertices"]
        lower = solid[:, 2].min() - start["args"]["clearance"]
        upper = solid[:, 2].max() + start["args"]["clearance"]
        fields = []
        for path in sorted((run / "frames").glob("*.npz")):
            level = int(path.stem.removeprefix("level"))
            endpoint = [row for row in history if row["level"] == level][-1]
            with np.load(path) as saved:
                position = saved["vertices_world_m"]
                force = saved["contact_forces_world_n"]
                residual = saved["equation_residual_n"]
            rotation = corotation(rest, position, mass)
            centered = (position - mass @ position / mass.sum()) @ rotation
            elastic_difference = centered - (rest - mass @ rest / mass.sum())
            fields.append(
                {
                    "field": file_id(path),
                    "minimum_position_m": position.min(axis=0).tolist(),
                    "maximum_position_m": position.max(axis=0).tolist(),
                    "mass_center_displacement_m": (mass @ (position - rest) / mass.sum()).tolist(),
                    "maximum_displacement_m": float(np.linalg.norm(position - rest, axis=1).max()),
                    "rotation_rad": float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))),
                    "rotation_matrix": rotation.tolist(),
                    "maximum_corotated_displacement_m": float(np.linalg.norm(elastic_difference, axis=1).max()),
                    "lower_vertical_half_thickness_envelope_gap_m": float(position[:, 2].min() - 0.00025 - lower),
                    "upper_vertical_half_thickness_envelope_gap_m": float(
                        upper - endpoint["compression_coordinate_m"] - position[:, 2].max() - 0.00025
                    ),
                    "contact_force_l2_n": float(np.linalg.norm(force)),
                    "equation_residual_l2_n": float(np.linalg.norm(residual)),
                }
            )
        carried = [row for row in history if row["upper_reaction_n"] > 0.09]
        after = [
            row
            for row in history
            if carried and row["time_s"] > carried[0]["time_s"] and row["upper_reaction_n"] < 1e-8
        ]
        windows = []
        for level in sorted({row["level"] for row in history}):
            level_rows = [row for row in history if row["level"] == level]
            endpoint = level_rows[-1]
            window = [row for row in level_rows if row["time_s"] >= endpoint["time_s"] - 0.05 - 1e-10]
            checks = []
            inertia_bounds = []
            for row in window:
                observed = dict(row)
                residual_norm = row["equation_residual_relative"] * row["residual_force_scale_n"]
                observed["equation_residual_relative"] = residual_norm / row["contact_force_norm_n"]
                # Older journals did not retain the inertia resultant. Bound it from
                # m a = f_internal + f_contact - residual without assuming equilibrium.
                bound = np.linalg.norm(np.array(row["internal_force_resultant_n"]) + row["contact_force_resultant_n"])
                bound += np.sqrt(len(rest)) * residual_norm
                inertia_bounds.append(float(bound))
                if "egg_inertia_resultant_norm_n" not in observed:
                    observed["egg_inertia_resultant_norm_n"] = float(bound)
                check = equilibrium(observed, endpoint["command_force_n"])
                if start.get("physical_pad_geometry") is not None:
                    check["physical_pad_law_and_transfer"] = row["physical_pad"]["normal_law_and_transfer_passed"]
                checks.append(check)
            failures = sum(any(row["solver_failure_flags"]) for row in level_rows)
            windows.append(
                {
                    "level": level,
                    "command_force_n": endpoint["command_force_n"],
                    "samples": len(window),
                    "failed_steps_in_level": failures,
                    "maximum_equation_residual_contact_relative": max(
                        row["equation_residual_relative"] * row["residual_force_scale_n"] / row["contact_force_norm_n"]
                        for row in window
                    ),
                    "maximum_inertia_resultant_bound_n": max(inertia_bounds),
                    "maximum_plate_speed_m_per_s": max(abs(row["plate_speed_m_per_s"]) for row in window),
                    "maximum_kinetic_j": max(row["egg_kinetic_j"] for row in window),
                    "checks": {key: all(row[key] for row in checks) for key in checks[0]},
                    "mechanical_window_passed": bool(
                        result["completed"]
                        and len(window) >= 5
                        and window[-1]["time_s"] - window[0]["time_s"] >= 0.05 - 1e-10
                        and not failures
                        and all(all(row.values()) for row in checks)
                    ),
                    "physical_accuracy_qualified": False,
                }
            )
        rows.append(
            {
                "result": file_id(run / "result.json"),
                "completed": result["completed"],
                "first_carried_time_s": carried[0]["time_s"] if carried else None,
                "first_zero_reaction_after_carry_s": after[0]["time_s"] if after else None,
                "maximum_equation_residual_last_50ms": max(
                    row["equation_residual_relative"]
                    for row in history
                    if row["time_s"] >= history[-1]["time_s"] - 0.05 - 1e-10
                ),
                "fields": fields,
                "equilibrium_windows": windows,
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "runs": rows,
            "scope": "End-position envelope gaps use vertical half-thickness offsets, a conservative diagnostic rather than exact face-offset distances. Zero reaction with overlap requires contact/solver investigation; it cannot be admitted from a zero failure flag.",
        },
    )


if __name__ == "__main__":
    main()
