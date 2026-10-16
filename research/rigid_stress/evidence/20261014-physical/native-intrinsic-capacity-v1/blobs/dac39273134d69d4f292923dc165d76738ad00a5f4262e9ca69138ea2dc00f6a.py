"""Compare exact common physical times across fixed-hand rigid integration intervals."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from .audit_franka_reference import rotation
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    fixed_config, fixed_geometries = None, None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        frozen = {key: value for key, value in config.items() if key not in ("output", "substeps")}
        geometry = json.loads((run / "results/fixture-assets/manifest.json").read_text())
        if fixed_config is None:
            fixed_config, fixed_geometries = frozen, geometry
        assert frozen == fixed_config and geometry == fixed_geometries
        report = json.loads((run / "results/summary.json").read_text())
        rows = [
            row
            for row in report["captures"]
            if row["physical_substep"] == 0 and row["phase"] == "released_free_body_fixed_hand"
        ]
        assert len({(row["tick"], row["physical_substep"]) for row in rows}) == len(rows)
        cases.append((config["substeps"], run, rows))
    cases.sort()
    assert [case[0] for case in cases] == [2, 4, 8]
    assert all([row["tick"] for row in case[2]] == [row["tick"] for row in cases[0][2]] for case in cases)
    changes = []
    observables = [
        ("position", "egg_position_m", "m"),
        ("velocity", "egg_velocity_m_s", "m/s"),
        ("angular_velocity", "egg_angular_velocity_rad_s", "rad/s"),
        ("jaw_position", "jaw_position_m", "m"),
    ]
    for previous, current in pairwise(cases):
        assert all(
            [c["opposite_geom"] for c in a["contacts"]] == [c["opposite_geom"] for c in b["contacts"]]
            for a, b in zip(previous[2], current[2], strict=True)
        )
        record = {
            "previous_substeps": previous[0],
            "current_substeps": current[0],
            "common_physical_samples": len(current[2]),
            "first_time_s": 0.01 * current[2][0]["tick"],
            "last_time_s": 0.01 * current[2][-1]["tick"],
        }
        for name, key, units in observables:
            a, b = (np.array([row[key] for row in case[2]]) for case in (previous, current))
            absolute = float(np.linalg.norm(b - a))
            record[name] = {
                "l2_absolute_difference": absolute,
                "units": units,
                "l2_relative_difference": absolute / float(np.linalg.norm(b)),
                "maximum_sample_difference": float(np.linalg.norm(b - a, axis=1).max()),
            }
        a, b = (
            np.array([[c["normal_force_N"] for c in row["contacts"]] for row in case[2]])
            for case in (previous, current)
        )
        assert a.shape == b.shape and a.shape[1] == 2
        difference = float(np.linalg.norm(b - a))
        record["normal_force"] = {
            "l2_absolute_difference_N": difference,
            "l2_relative_difference": difference / float(np.linalg.norm(b)),
            "maximum_sample_difference_N": float(np.max(np.abs(b - a))),
        }
        angles = [
            float(
                np.arccos(
                    np.clip(
                        (np.trace(rotation(b["egg_quaternion_wxyz"]) @ rotation(a["egg_quaternion_wxyz"]).T) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
            for a, b in zip(previous[2], current[2], strict=True)
        ]
        record["maximum_orientation_difference_rad"] = max(angles)
        changes.append(record)
    summaries = [
        {
            "run": str(run),
            "integration_interval_s": 0.01 / substeps,
            "summary_sha256": hashlib.sha256((run / "results/summary.json").read_bytes()).hexdigest(),
            "last_common_time_frame": rows[-1],
        }
        for substeps, run, rows in cases
    ]
    output = {
        "scope": __doc__,
        "comparisons": changes,
        "cases": summaries,
        "goal_completion": False,
        "limitations": "Comparison uses physical_substep0 at identical macro ticks, without interpolation. Original controls, supports, geometry, material and total52s history remain fixed. Position relative differences include the common world offset; absolute translation and orientation differences must also be inspected. Temporal agreement is separate from contact regularization and physical model/deformation feedback. Persisting angular motion prevents static-contact acceptance; no stress accuracy or physical operating range is inferred.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"comparisons": changes}), flush=True)


if __name__ == "__main__":
    main()
