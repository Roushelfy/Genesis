"""Compare admitted DPR variants without conflating changed loads with physical accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manual", "radius", "hertz", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.manual, args.radius, args.hertz]
    starts = [json.loads((path / "start.json").read_text()) for path in paths]
    results = [json.loads((path / "result.json").read_text()) for path in paths]
    assert all(result["passed"] for result in results)
    assert len({result["accepted_frames"] for result in results}) == 1
    assert all(start["conditions"] == starts[0]["conditions"] for start in starts)
    assert len({start["quadrature"] for start in starts}) == 1
    observations = []
    for path in paths:
        with np.load(path / "observations.npz") as saved:
            observations.append({name: saved[name] for name in saved.files})
    for observation in observations[1:]:
        np.testing.assert_array_equal(observation["positions"], observations[0]["positions"])
    common = sorted(set.intersection(*[{file.name for file in path.glob("field-*.npz")} for path in paths]))
    assert common
    rows = []
    for name in common:
        fields = []
        for path in paths:
            with np.load(path / name) as saved:
                fields.append({key: saved[key] for key in saved.files})
        active = fields[0]["valid"]
        for field in fields[1:]:
            np.testing.assert_array_equal(field["valid"], active)
            for key in ("force", "normal", "position", "friction"):
                np.testing.assert_array_equal(field[key][active], fields[0][key][active])
        np.testing.assert_array_equal(fields[1]["radius"][active], fields[2]["radius"][active])
        reference = fields[0]["stress_tensor"]
        rows.append(
            {
                "field": name,
                "radius_vs_manual_unweighted_tensor_difference": float(
                    np.linalg.norm(fields[1]["stress_tensor"] - reference) / np.linalg.norm(reference)
                ),
                "hertz_vs_manual_unweighted_tensor_difference": float(
                    np.linalg.norm(fields[2]["stress_tensor"] - reference) / np.linalg.norm(reference)
                ),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "identity": runtime_identity(require_gpu_inventory=False),
            "sources": [file_id(path / "result.json") for path in paths],
            "accepted_frames": results[0]["accepted_frames"],
            "rigid_positions_exactly_equal": True,
            "common_keyframe_contact_inputs_exactly_equal": True,
            "automatic_radii_exactly_equal_across_profiles": True,
            "rows": rows,
            "peak_over_window_pa": [float(observation["peaks"].max()) for observation in observations],
            "scope": "Same rigid trajectory and source contact inputs; different prescribed finite load footprints. "
            "Stress differences against manual DPR are model differences, not physical errors. This admitted window "
            "does not certify a complete grasp/release cycle or the sharp-edge contact approximation.",
        },
    )


if __name__ == "__main__":
    main()
